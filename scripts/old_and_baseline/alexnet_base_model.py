# Packages 
import os
import time
start_time = time.time()

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models import AlexNet_Weights, VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, EfficientNet_B2_Weights
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import pearsonr as corr

# Functions and classes
# Platform definition
platform = 'jupyter_notebook'

# data_dir = '../../../Projects/Datasets/Biomedical/algonauts_2023_challenge_data'
# Data folder definition
data_dir = '../Datasets/Biomedical/algonauts_2023_challenge_data'
# Used to save the prediction of saved model
parent_submission_dir = './algonauts_2023_challenge_submission'
ncsnr_dir = '../Datasets/Biomedical/algonauts_ncsnr'
images_trials = '../Datasets/Biomedical/algonauts_train_images_trials'
        
class argObj:
  def __init__(self, data_dir, parent_submission_dir, subj, parent_ncsnr_dir = ncsnr_dir, images_trials_parent_dir = images_trials):
    # Define the dir where data is stored
    # 1 became 01
    self.subj = format(subj, '02') # '0numberofchars'
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

    # NCSNR
    self.parent_ncsnr_dir = parent_ncsnr_dir
    self.ncsnr_dir = os.path.join(self.parent_ncsnr_dir, 'subj'+self.subj)
    
    # SUBMISSION DIR
    self.parent_submission_dir = parent_submission_dir
    self.subject_submission_dir = os.path.join(self.parent_submission_dir,
        'subj'+self.subj)
    # Create the submission directory if not existing
    if not os.path.isdir(self.subject_submission_dir):
        os.makedirs(self.subject_submission_dir)

    # Train Images Trials 
    self.images_trials_parent_dir = images_trials_parent_dir
    self.images_trials_dir = os.path.join(self.images_trials_parent_dir, 'subj'+self.subj)

### Parameters ###
pca_component = 300
train_percentage = 90 # X% of the training data will be used for training, (100-X)% for validation
batch_mode = "dynamic" # "dynamic" or "static"
batch_size_min = 500 # Batch size has to be major than pca_component
batch_size_max = 550
save_predictions = True 
feature_model_type = "alexnet" #@param ["alexnet", "vgg16", "efficientnetb2", "efficientnetb2lib"]
regression_type = "linear" #@param ["linear", "ridge"]

if __name__ == "__main__":
            
    # Select the device to run the model on
    device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)
    
    # Check if GPU is available and torch is using it
    print("Check if GPU is available and if torch is using it ..")
    print("\n")
    print("Torch Cuda is available?")
    print(torch.cuda.is_available())
    print("Torch Cuda device count is :")
    print(torch.cuda.device_count())
    print("Torch Cuda current device is :")
    print(torch.cuda.current_device())
    print("Torch Cuda device is :")
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
    print("Pytorch version：")
    print(torch.__version__)
    print("CUDA Version: ")
    print(torch.version.cuda)
    print("cuDNN version is :")
    print(torch.backends.cudnn.version())
    print("\n")
    
    noise_norm_corr_dict = {}
    # For each subject, start the training process
    for subj in list(range(1, 9)):
        print('############## Subject: ' + str(subj) + ' ##############')
        ## Definining paths to data and submission directories ##
        args = argObj(data_dir, parent_submission_dir, subj)
        
        ## Load the fmri response for left and right hemispheres ##
        fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
        # Check the shapes of the data
        print('LH training fMRI data shape:', lh_fmri.shape)
        print('RH training fMRI data shape:', rh_fmri.shape)
        
        ## load the stimulus images ##
        train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
        test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')
        # Create lists will all training and test image file names, sorted
        train_img_list = os.listdir(train_img_dir)
        train_img_list.sort()
        test_img_list = os.listdir(test_img_dir)
        test_img_list.sort()
                
        ## Create the training, validation and test partitions indices ##
        rand_seed = 5 #@param
        np.random.seed(rand_seed)
        # train_percentage = 90
        # Calculate how many stimulus images correspond to 90% of the training data
        num_train = int(np.round(len(train_img_list) / 100 * train_percentage))
        # Shuffle all training stimulus images
        idxs = np.arange(len(train_img_list))
        np.random.shuffle(idxs)
        # Assign 90% of the shuffled stimulus images to the training partition,
        # and 10% to the test partition
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
        # No need to shuffle or split the test stimulus images
        idxs_test = np.arange(len(test_img_list))
        print('Total train images: ' + str(len(train_img_list)))
        print('Training stimulus images: ' + format(len(idxs_train)))
        print('Validation stimulus images: ' + format(len(idxs_val)))
        print('Test stimulus images: ' + format(len(idxs_test)))
        
        ## Create the training, validation and test partitions dataloaders ##
        # Preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize((256,256)), # resize the images to 224x224 pixels (256x256)
            transforms.CenterCrop(224), # IN THE OLDER SCRIPT: no center crop, only resize to 224x224
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            # normalize the images color channels
            # mean: [0.485, 0.456, 0.406] for the three channels
            # std: [0.229, 0.224, 0.225] for the three channels
        ])
        # Dataset class
        class ImageDataset(Dataset):
            def __init__(self, imgs_paths, idxs, transform):
                self.imgs_paths = np.array(imgs_paths)[idxs]
                self.transform = transform

            def __len__(self):
                return len(self.imgs_paths)

            def __getitem__(self, idx):
                # Load the image
                img_path = self.imgs_paths[idx]
                img = Image.open(img_path).convert('RGB')
                # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
                if self.transform:
                    img = self.transform(img).to(device)
                return img
        # Dataloader class
        # 3 6 -> 310 / other ones -> 300
        if batch_mode == 'static':
            if subj == 3 or subj == 6:
                batch_size = 310
            else: 
                batch_size = 300 # 300 #@param (310 for 3 and 6)
            print(batch_mode, "batch size: ", batch_size)
        if batch_mode == 'dynamic':
            # Batch size should never be less than pca components
            for batch_size in range(batch_size_min, batch_size_max + 1):
                if len(idxs_train) % batch_size >= batch_size_min:
                    # print("Train images: ", len(train_img_list))
                    # print("Remainder: ", len(train_img_list) % batch_size)
                    print(batch_mode, "Batch size: ", batch_size)
                    break

        # Get the paths of all image files
        train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
        test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

        # The DataLoaders contain the ImageDataset class
        train_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_train, transform), 
            batch_size=batch_size
        )
        val_imgs_dataloader = DataLoader(
            ImageDataset(train_imgs_paths, idxs_val, transform), 
            batch_size=batch_size
        )
        test_imgs_dataloader = DataLoader(
            ImageDataset(test_imgs_paths, idxs_test, transform), 
            batch_size=batch_size
        )
        
        ## Spli the fmri data into training and validation partitions ##
        lh_fmri_train = lh_fmri[idxs_train]
        lh_fmri_val = lh_fmri[idxs_val]
        rh_fmri_train = rh_fmri[idxs_train]
        rh_fmri_val = rh_fmri[idxs_val]
        del lh_fmri, rh_fmri
        
        if feature_model_type == 'alexnet':
            ## AlexNet ##
            model = torch.hub.load('pytorch/vision:v0.10.0', 
                                'alexnet', 
                                weights=AlexNet_Weights.IMAGENET1K_V1)
            model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
            model.eval() # set the model to evaluation mode, since you are not training it
            train_nodes, _ = get_graph_node_names(model)
            model_layer = "features.12" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
            feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
            for param in feature_extractor.parameters():
                param.requires_grad = False
        elif feature_model_type == 'vgg16':
            model = torch.hub.load('pytorch/vision:v0.10.0', 
                                'vgg16_bn', 
                                weights=VGG16_BN_Weights.IMAGENET1K_V1)
            model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
            model.eval() # set the model to evaluation mode, since you are not training it
            # for param in model.parameters():
            #      param.requires_grad = False
            train_nodes, _ = get_graph_node_names(model)
            model_layer = "flatten" #@param ["features.2", "features.5", "features.7", "features.10", "features.12", "features.14", "features.17", "features.19", "features.21", "features.24", "features.26", "features.28", "features.31", "features.33", "features.35", "classifier.2", "classifier.5", "classifier.7"] {allow-input: true}
            feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
            # for param in feature_extractor.parameters():
            #     param.requires_grad = False
            # param.detach
            # torch.cuda.empty_cache()
            # def feature_extractor_(x, feature_extractor):
            #     with torch.no_grad():
            #         return feature_extractor(x)
        elif feature_model_type == 'efficientnetb2':
            model = models.efficientnet_b2(weights = EfficientNet_B2_Weights.IMAGENET1K_V1)
            model.eval() # set the model to evaluation mode, since you are not training it
            model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
            model_layer = "features.8" #@param ['x', 'features.0', 'features.1.0.block.0', 'features.1.0.block.1', 'features.1.0.block.2', 'features.1.1.block.0', 'features.1.1.block.1', 'features.1.1.block.2', 'features.1.1.stochastic_depth', 'features.1.1.add', 'features.2.0.block.0', 'features.2.0.block.1', 'features.2.0.block.2', 'features.2.0.block.3', 'features.2.1.block.0', 'features.2.1.block.1', 'features.2.1.block.2', 'features.2.1.block.3', 'features.2.1.stochastic_depth', 'features.2.1.add', 'features.2.2.block.0', 'features.2.2.block.1', 'features.2.2.block.2', 'features.2.2.block.3', 'features.2.2.stochastic_depth', 'features.2.2.add', 'features.3.0.block.0', 'features.3.0.block.1', 'features.3.0.block.2', 'features.3.0.block.3', 'features.3.1.block.0', 'features.3.1.block.1', 'features.3.1.block.2', 'features.3.1.block.3', 'features.3.1.stochastic_depth', 'features.3.1.add', 'features.3.2.block.0', 'features.3.2.block.1', 'features.3.2.block.2', 'features.3.2.block.3', 'features.3.2.stochastic_depth', 'features.3.2.add', 'features.4.0.block.0', 'features.4.0.block.1', 'features.4.0.block.2', 'features.4.0.block.3', 'features.4.1.block.0', 'features.4.1.block.1', 'features.4.1.block.2', 'features.4.1.block.3', 'features.4.1.stochastic_depth', 'features.4.1.add', 'features.4.2.block.0', 'features.4.2.block.1', 'features.4.2.block.2', 'features.4.2.block.3', 'features.4.2.stochastic_depth', 'features.4.2.add', 'features.4.3.block.0', 'features.4.3.block.1', 'features.4.3.block.2', 'features.4.3.block.3', 'features.4.3.stochastic_depth', 'features.4.3.add', 'features.5.0.block.0', 'features.5.0.block.1', 'features.5.0.block.2', 'features.5.0.block.3', 'features.5.1.block.0', 'features.5.1.block.1', 'features.5.1.block.2', 'features.5.1.block.3', 'features.5.1.stochastic_depth', 'features.5.1.add', 'features.5.2.block.0', 'features.5.2.block.1', 'features.5.2.block.2', 'features.5.2.block.3', 'features.5.2.stochastic_depth', 'features.5.2.add', 'features.5.3.block.0', 'features.5.3.block.1', 'features.5.3.block.2', 'features.5.3.block.3', 'features.5.3.stochastic_depth', 'features.5.3.add', 'features.6.0.block.0', 'features.6.0.block.1', 'features.6.0.block.2', 'features.6.0.block.3', 'features.6.1.block.0', 'features.6.1.block.1', 'features.6.1.block.2', 'features.6.1.block.3', 'features.6.1.stochastic_depth', 'features.6.1.add', 'features.6.2.block.0', 'features.6.2.block.1', 'features.6.2.block.2', 'features.6.2.block.3', 'features.6.2.stochastic_depth', 'features.6.2.add', 'features.6.3.block.0', 'features.6.3.block.1', 'features.6.3.block.2', 'features.6.3.block.3', 'features.6.3.stochastic_depth', 'features.6.3.add', 'features.6.4.block.0', 'features.6.4.block.1', 'features.6.4.block.2', 'features.6.4.block.3', 'features.6.4.stochastic_depth', 'features.6.4.add', 'features.7.0.block.0', 'features.7.0.block.1', 'features.7.0.block.2', 'features.7.0.block.3', 'features.7.1.block.0', 'features.7.1.block.1', 'features.7.1.block.2', 'features.7.1.block.3', 'features.7.1.stochastic_depth', 'features.7.1.add', 'features.8', 'avgpool', 'flatten', 'classifier.0', 'classifier.1']
            feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
            for param in feature_extractor.parameters():
                param.requires_grad = False
            print(f'Feature extractor: {feature_model_type}, layer: {model_layer}')
        elif feature_model_type == 'efficientnetb2lib':
            model = EfficientNet.from_pretrained('efficientnet-b2')
            model = nn.Sequential(*list(model.children())[:-1])
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            def feature_extractor(x):
                with torch.no_grad():
                    x = model(x)
                    return x
        
        ## PCA ##
        def fit_pca(feature_extractor, dataloader):

            # Define PCA parameters
            pca = IncrementalPCA(n_components=pca_component, batch_size=batch_size)

            # Fit PCA to batch
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                print(_)
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                # Fit PCA to batch
                print(ft.shape)
                pca.partial_fit(ft.detach().cpu().numpy())
                torch.cuda.empty_cache()
            return pca
        
        ## Fit PCA to training data ##
        print(f'Fitting Incremental PCA ({pca_component} components) to training data...')
        pca = fit_pca(feature_extractor, train_imgs_dataloader)
        print("Comulative Explained variance ratio: ", sum(pca.explained_variance_ratio_))
        print("Number of components: ", pca.n_components_)
        
        ## Feature extraction ##
        def extract_features(feature_extractor, dataloader, pca):
            features = []
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                # Apply PCA transform
                ft = pca.transform(ft.cpu().detach().numpy())
                features.append(ft)
            return np.vstack(features)
        
        print('Extracting features from training, validation and test data...')
        features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
        features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
        features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)
        del model, pca, feature_extractor
        
        ## Fit the linear model ##
        # Fit linear regressions on the training data
        if regression_type == 'ridge':
            print('Fitting ridge regressions on the training data...')
            reg_lh = Ridge(alpha=1e5).fit(features_train, lh_fmri_train)
            reg_rh = Ridge(alpha=1e5).fit(features_train, rh_fmri_train)
        elif regression_type == 'linear':
            print('Fitting linear regressions on the training data...')
            reg_lh = LinearRegression().fit(features_train, lh_fmri_train)
            reg_rh = LinearRegression().fit(features_train, rh_fmri_train)
        # Use fitted linear regressions to predict the validation and test fMRI data
        print('Predicting fMRI data on the validation and test data...')
        lh_fmri_val_pred = reg_lh.predict(features_val)
        lh_fmri_test_pred = reg_lh.predict(features_test)
        rh_fmri_val_pred = reg_rh.predict(features_val)
        rh_fmri_test_pred = reg_rh.predict(features_test)
        
        # Test submission files
        if save_predictions:
            lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
            rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
            np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
            np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)
        
        ## Compute the correlation between the predicted and actual fMRI data ##
        print('Computing the correlation between the predicted and actual fMRI data...')
        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh_fmri_val_pred.shape[1])):
            lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0] # 0 per selezionare valore e non p-value

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh_fmri_val_pred.shape[1])):
            rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]
        
        ## Evaluate the model ##
        # NCSNR
        from nibabel.freesurfer.mghformat import load
        lh_ncsnr = load(os.path.join(args.ncsnr_dir, 'lh.ncsnr.mgh'))
        rh_ncsnr = load(os.path.join(args.ncsnr_dir, 'rh.ncsnr.mgh'))
        lh_ncsnr_all_vertices = lh_ncsnr.get_fdata()[:,0,0]
        rh_ncsnr_all_vertices = rh_ncsnr.get_fdata()[:,0,0]
        # fsaverage
        hemisphere = ['left', 'right'] #@param ['left', 'right'] {allow-input: true}
        # Load the brain surface map of all vertices
        roi_dir = os.path.join(args.data_dir, 'roi_masks',
            hemisphere[0][0]+'h.all-vertices_fsaverage_space.npy')
        lh_fsaverage_all_vertices = np.load(roi_dir)
        roi_dir = os.path.join(args.data_dir, 'roi_masks',
            hemisphere[1][0]+'h.all-vertices_fsaverage_space.npy')
        rh_fsaverage_all_vertices = np.load(roi_dir)
        # NCSNR for challenge vertices
        lh_ncsnr_challenge_vertices = lh_ncsnr_all_vertices[np.where(lh_fsaverage_all_vertices)[0]]
        rh_ncsnr_challenge_vertices = rh_ncsnr_all_vertices[np.where(rh_fsaverage_all_vertices)[0]]
        # TRIALS
        image_trial_number = np.load(os.path.join(args.images_trials_dir, 'train_images_trials.npy'))
        image_trial_number_val = image_trial_number[idxs_val]
        # Noise Ceiling
        A = len(image_trial_number_val[image_trial_number_val == 3])
        B = len(image_trial_number_val[image_trial_number_val == 2])
        C = len(image_trial_number_val[image_trial_number_val == 1])
        lh_noise_ceiling = (lh_ncsnr_challenge_vertices ** 2) / ((lh_ncsnr_challenge_vertices ** 2) + ((A/3 + B/2 + C/1) / (A + B + C)))
        rh_noise_ceiling = (rh_ncsnr_challenge_vertices ** 2) / ((rh_ncsnr_challenge_vertices ** 2) + ((A/3 + B/2 + C/1) / (A + B + C)))
        # Noise Normalized Squared Correlation
        lh_correlation[lh_correlation<0] = 0
        rh_correlation[rh_correlation<0] = 0
        lh_correlation = lh_correlation ** 2
        rh_correlation = rh_correlation ** 2
        lh_noise_ceiling[lh_noise_ceiling==0] = 1e-14
        rh_noise_ceiling[rh_noise_ceiling==0] = 1e-14
        lh_noise_norm_corr = np.divide(lh_correlation, lh_noise_ceiling)
        rh_noise_norm_corr = np.divide(rh_correlation, rh_noise_ceiling)
        lh_noise_norm_corr[lh_noise_norm_corr>1] = 1
        rh_noise_norm_corr[rh_noise_norm_corr>1] = 1
        
        noise_norm_corr_dict[f'lh_{subj}'] = lh_noise_norm_corr
        noise_norm_corr_dict[f'rh_{subj}'] = rh_noise_norm_corr
        print("Score -> Median Noise Normalized Squared Correlation Percentage (LH and RH)")
        print("LH subj",subj,"| Score: ",np.median(lh_noise_norm_corr)*100)
        print("RH subj",subj,"| Score: ",np.median(rh_noise_norm_corr)*100)

print("#########################")
print("#########################")
print("#########################")
print("#########################") 
print("Median Noise Normalized Squared Correlation Percentage (LH and RH) for each subject")    
for key, value in noise_norm_corr_dict.items():
    print("Subject ->",key,"| Score ->",np.median(value)*100)
    
concatenated_correlations = np.concatenate(list(noise_norm_corr_dict.values()))
print("#########################")
print("#########################")
print("#########################")
print("#########################")
print("Median Noise Normalized Squared Correlation Percentage on all subjects")
print("Concatenated Subjs | Score: ",np.median(concatenated_correlations)*100)

end_time = time.time()
total_time = end_time - start_time

print("Execution time: ", total_time/60, " min")