### Packages import
import os
import gc
import time
import numpy as np
from time import strftime
import sys

import torch
from torchvision import models
from src.cuda_checker import cuda_torch_check, memory_checker


### My modules import
from src.data_loader import argObj, data_loaders_stimuli_fmri
from src import image_preprocessing
from src.feature_extraction import model_loader, fit_pca, pca_batch_calculator, extract_and_pca_features, extract_features_no_pca
from src.encoding import linear_regression, compute_perason_numpy
from src.evaluation_metrics import median_squared_noisenorm_correlation
from src.visualize import histogram, box_plot, noise_norm_corr_ROI, final_subj_corr_dataframe_boxplot_istograms

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from src.visualize import histogram, box_plot

### Cuda setup and check
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from torch.cuda.amp import autocast, GradScaler


# Select the device to run the model on
device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
# Check if cuda is available
device = torch.device(device)
cuda_torch_check()

train_percentage = 90 # X% of the training data will be used for training, (100-X)% for validation
transform = image_preprocessing.imagenet_V2_transform

batch_size = 16
pca_component = 1024
min_pca_batch_size = pca_component + 300 # pca_component * 2

compute_pca = True
architecture = "EndtoEnd" #@param ["EndtoEnd", "TwoStep"] {allow-input: true}
feature_model_type = "VGG19" #@param ["alexnet", "ZFNet", "resnet50", "vgg16","vgg19_bn" , "efficientnetb2", "efficientnetb2lib"]
model_layer = "features.43"
regression_type = "MLP" #@param ["linear", "ridge"]

save = False 

alpha_l = 1e5
alpha_r = 1e5
grid_search = False

### Path definition
if isinstance(model_layer, list):
    model_layer_full = '+'.join(model_layer)
else:
    model_layer_full = model_layer

datetime_id = strftime("(%Y-%m-%d_%H-%M)")
submission_name = f'{strftime("(%Y-%m-%d_%H-%M)")}-{architecture}_{feature_model_type}_{model_layer}-pca_{pca_component}-{regression_type}-alpha_{"{:.1e}".format(alpha_l)}'

data_home_dir = '../Datasets/Biomedical'
data_dir = '../Datasets/Biomedical/algonauts_2023_challenge_data'
# Used to save the prediction of saved model
parent_submission_dir = f'./files/submissions/{submission_name}'
if not os.path.isdir(parent_submission_dir) and save:
            os.makedirs(parent_submission_dir)
ncsnr_dir = '../Datasets/Biomedical/algonauts_ncsnr'
images_trials_dir = '../Datasets/Biomedical/algonauts_train_images_trials'

subj = 1

print('############################ Subject: ' + str(subj) + ' ############################ \n')
# Definining paths to data and submission directories ##
args = argObj(subj, data_home_dir, data_dir, parent_submission_dir, ncsnr_dir, images_trials_dir, save) 
# Obtain the indices of the training, validation and test data
idxs_train, idxs_val, idxs_test, train_imgs_paths, test_imgs_paths = args.images_idx_splitter(train_percentage)

# Defining the images data loaderds
data_loaders = data_loaders_stimuli_fmri(idxs_train, 
                                            idxs_val, 
                                            idxs_test, 
                                            train_imgs_paths, 
                                            test_imgs_paths,
                                            lh_fmri_path = args.lh_fmri,
                                            rh_fmri_path = args.rh_fmri)

train_dataloader_lh, val_dataloader_lh, test_dataloader_lh, train_dataloader_rh, val_dataloader_rh, test_dataloader_rh = data_loaders.images_fmri_dataloader(batch_size, transform)

lh_fmri_train, lh_fmri_val, rh_fmri_train, rh_fmri_val = data_loaders.fmri_splitter()

# Definizione della rete custom
class CustomNet(nn.Module):
    def __init__(self, num_outputs):
        super(CustomNet, self).__init__()
        
        # Caricamento del modello VGG-19 pre-addestrato
        vgg = models.vgg16_bn(pretrained=True)
        
        # Freeze the pre-trained layers
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Utilizziamo solo i layer di feature extraction senza gli ultimi layer completamente connessi
        self.features = nn.Sequential(*list(vgg.features.children()))
        
        # Aggiunta del layer di average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        # Aggiunta del layer completamente connesso per la regressione
        self.fc = nn.Linear(512 * 7 * 7, num_outputs)  # L'input size 512 è determinato dal numero di feature maps generate dall'ultimo layer di VGG-19
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# Test
# model = models.vgg16_bn(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.classifier = nn.Linear(512 * 7 * 7, 19004)
# model.to(device)
# model.train()

first_batch = next(iter(train_dataloader_lh))

# Creazione dell'istanza del modello
model = CustomNet(first_batch[1].shape[1])
model.to(device)
model.train()
# Stampa della struttura del modello
#print(model)

# Metrica da printare
def mean_pearson_corr(y_true, y_pred):
    pearson_corr = 0.0
    for i in range(y_true.shape[1]):
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        pearson_corr += corr
    return pearson_corr / y_true.shape[1]

learning_rate = 0.001
num_epochs = 10

# Definizione della loss function
criterion = nn.MSELoss()

# Selezionare solo i parametri del layer regressor per l'ottimizzazione
parameters = model.fc.parameters()

# Definizione dell'ottimizzatore
optimizer = optim.SGD(parameters, lr=learning_rate)

from tqdm import tqdm

for epoch in range(num_epochs):
    #model.train()
    total_loss = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_pearson_corr = 0.0
    total_batches = 0
    
    for images, labels in tqdm(train_dataloader_lh):
        #print(labels.shape)
        
        # Passaggio degli input attraverso la rete
        outputs = model(images.to(device))
        
        # Calcolo della loss
        loss = criterion(outputs, labels.to(device))
        
        # Backpropagation
        # Reset dei gradienti
        optimizer.zero_grad()
        loss.backward()
        
        # Clean memory
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        outputs = outputs.detach().cpu()
        torch.cuda.empty_cache()
        
        # Ottimizzazione dei parametri
        optimizer.step()
        
        # Calcolo delle metriche di accuratezza
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_rmse += mean_squared_error(labels.numpy(), outputs.numpy(), squared=False) * batch_size
        total_mae += mean_absolute_error(labels.numpy(), outputs.numpy()) * batch_size
        total_pearson_corr += mean_pearson_corr(labels.numpy(), outputs.numpy()) * batch_size
        total_batches += batch_size
    
    # Calcolo delle metriche di accuratezza medie
    average_loss = total_loss / total_batches
    average_rmse = total_rmse / total_batches
    average_mae = total_mae / total_batches
    average_pearson_corr = total_pearson_corr / total_batches
    
    # Stampa delle metriche di accuratezza
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, RMSE: {average_rmse:.4f}, MAE: {average_mae:.4f}, Pearson Corr: {average_pearson_corr:.4f}")

