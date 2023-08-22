### Packages import
import os
import gc
import time
from time import strftime
import sys
start_time = time.time()

import numpy as np
import pandas as pd

import torch
from torchvision import models
from src.cuda_checker import cuda_torch_check, memory_checker

### My modules import
from src.data_loader import argObj, data_loaders_stimuli_fmri
from src import image_preprocessing
from src.feature_extraction import model_loader, fit_pca, pca_batch_calculator, extract_and_pca_features, extract_features_no_pca
from src.encoding import linear_regression
from src.evaluation_metrics import median_squared_noisenorm_correlation
from src.visualize import histogram, box_plot, noise_norm_corr_ROI, final_subj_corr_dataframe_boxplot_istograms

### Cuda setup and check
import torch
# Select the device to run the model on
device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
# Check if cuda is available
device = torch.device(device)
cuda_torch_check()

### Parameters definition
train_percentage = 90 # X% of the training data will be used for training, (100-X)% for validation
transform = image_preprocessing.imagenet_transform_alt

batch_size = 64
pca_component = 100
min_pca_batch_size = pca_component + 300 # pca_component * 2

compute_pca = True
feature_model_type = "alexnet" #@param ["alexnet", "ZFNet", "resnet50", "vgg16","vgg19_bn" , "efficientnetb2", "efficientnetb2lib"]
model_layer = "features.12"
regression_type = "linear" #@param ["linear", "ridge"]

save = True 

alpha_l = 1e5
alpha_r = 1e5
grid_search = False

### Path definition
if isinstance(model_layer, list):
    model_layer_full = '+'.join(model_layer)
else:
    model_layer_full = model_layer

datetime_id = strftime("(%Y-%m-%d_%H-%M)")
submission_name = f'{strftime("(%Y-%m-%d_%H-%M)")}-{feature_model_type}_{model_layer}-pca_{pca_component}-{regression_type}-alpha_{"{:.1e}".format(alpha_l)}'

### Path definition
# Data folder definition
data_home_dir = '../Datasets/Biomedical'
data_dir = '../Datasets/Biomedical/algonauts_2023_challenge_data'
# Used to save the prediction of saved model
parent_submission_dir = f'./files/submissions/{submission_name}'
if not os.path.isdir(parent_submission_dir) and save:
            os.makedirs(parent_submission_dir)
ncsnr_dir = '../Datasets/Biomedical/algonauts_ncsnr'
images_trials_dir = '../Datasets/Biomedical/algonauts_train_images_trials'

if __name__ == "__main__":
    print(submission_name + "\n")
    noise_norm_corr_dict = {}
    noise_norm_corr_ROI_dict = {} 
    for subj in list(range(1, 9)):
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
        
        train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = data_loaders.images_dataloader(batch_size, transform)
        
        model, feature_extractor = model_loader(feature_model_type, model_layer, device)
        
        if compute_pca:
            # Fit the PCA model
            pca_batch_size, n_stacked_batches = pca_batch_calculator(len(idxs_train),
                                                                    batch_size,
                                                                    min_pca_batch_size,
                                                                    pca_component)
            
            pca = fit_pca(feature_extractor,
                            train_imgs_dataloader,
                            pca_component,
                            n_stacked_batches,
                            pca_batch_size,
                            device)
            print("Comulative Explained variance ratio: ", sum(pca.explained_variance_ratio_))
            print("Number of components: ", pca.n_components_)
            
            print('## Extracting features from training, validation and test data...')
            features_train = extract_and_pca_features(feature_extractor, train_imgs_dataloader, pca, n_stacked_batches, device)
            features_val = extract_and_pca_features(feature_extractor, val_imgs_dataloader, pca, n_stacked_batches, device)
            features_test = extract_and_pca_features(feature_extractor, test_imgs_dataloader, pca, n_stacked_batches, device)
            
            # print("\n")
            # print('## Checking and Freeing  GPU memory...')
            # memory_checker()
            model.to('cpu') # sposto sulla ram
            feature_extractor.to('cpu') # sposto sulla ram
            del model, feature_extractor, pca, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader  # elimino dalla ram
            torch.cuda.empty_cache() # elimino la chache vram
            gc.collect() # elimino la cache ram
            # memory_checker()
        else:
            print('## Extracting features from training, validation and test data...')
            features_train = extract_features_no_pca(feature_extractor, train_imgs_dataloader, device)
            features_val = extract_features_no_pca(feature_extractor, val_imgs_dataloader, device)
            features_test = extract_features_no_pca(feature_extractor, test_imgs_dataloader, device)
            
            model.to('cpu') # sposto sulla ram
            feature_extractor.to('cpu') # sposto sulla ram
            del model, feature_extractor, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader  # elimino dalla ram
            torch.cuda.empty_cache() # elimino la chache vram
            gc.collect() # elimino la cache ram

        ## Fit the linear model ##
        print('\n ## Fit Encoder and Predict...')
        lh_fmri_train, lh_fmri_val, rh_fmri_train, rh_fmri_val = data_loaders.fmri_splitter()
        print('LH fMRI number of vertices:', lh_fmri_train.shape)
        print('RH fMRI number of vertices:', rh_fmri_train.shape)
        
        
        lh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_val_pred, rh_fmri_test_pred = linear_regression(regression_type, 
                                                                                                        features_train, 
                                                                                                        features_val, 
                                                                                                        features_test, 
                                                                                                        lh_fmri_train, 
                                                                                                        rh_fmri_train, 
                                                                                                        save,
                                                                                                        args.subject_test_submission_dir,
                                                                                                        alpha_l,
                                                                                                        alpha_r,
                                                                                                        grid_search)
        
        noise_norm_corr_dict[f'lh_{subj}'], noise_norm_corr_dict[f'rh_{subj}'] = median_squared_noisenorm_correlation(lh_fmri_val_pred, 
                                                                                                                        rh_fmri_val_pred,
                                                                                                                        lh_fmri_val,
                                                                                                                        rh_fmri_val,
                                                                                                                        args.data_dir,
                                                                                                                        args.ncsnr_dir,
                                                                                                                        args.images_trials_dir,
                                                                                                                        idxs_val)
        
        noise_norm_corr_ROI_dict[f'{subj}'] = noise_norm_corr_ROI(args.data_dir, 
                                                                  noise_norm_corr_dict[f'lh_{subj}'], 
                                                                  noise_norm_corr_dict[f'rh_{subj}'],
                                                                  save)
        
        print("\n Score -> Median Noise Normalized Squared Correlation Percentage (LH and RH)")
        print("LH subj",subj,"| Score: ",np.median(noise_norm_corr_dict[f'lh_{subj}'])*100)
        print("RH subj",subj,"| Score: ",np.median(noise_norm_corr_dict[f'rh_{subj}'])*100)

        ## Saving graphs showing accuracy score indipendently for each ROI, hemisphere of the subject ##
        if save:
            histogram(args.data_dir, noise_norm_corr_dict[f'lh_{subj}'], 
                        noise_norm_corr_dict[f'rh_{subj}'], 
                        f'{submission_name}_subj{subj}', 
                        save = args.subject_val_images_submission_dir)
            
            box_plot(args.data_dir, noise_norm_corr_dict[f'lh_{subj}'], 
                        noise_norm_corr_dict[f'rh_{subj}'], 
                        f'{submission_name}_subj{subj}', 
                        save = args.subject_val_images_submission_dir)
        
        
print("#########################")
print("##### FINAL RESULTS #####")
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

# Save the results in an output file and figures
if save:
    # Final single score on validation
    with open(os.path.join(parent_submission_dir, 'val_scores.txt'), 'a') as f:
        f.write(f'All_vertices: {np.median(concatenated_correlations)*100}')
    # Final subject/hemisphere-wise score on validation results save
    with open(os.path.join(parent_submission_dir, 'val_scores_subj_hemisphere.txt'), 'a') as f:
        for key, value in noise_norm_corr_dict.items():
            f.write(f'{key}: {np.median(value)*100}\n')
    # Final subject/hemisphere/ROI-wise score on validation results save
    noise_norm_corr_ROI_concatenated = pd.concat(noise_norm_corr_ROI_dict.values())
    key_values = [key for key in noise_norm_corr_ROI_dict.keys() for i in range(len(noise_norm_corr_ROI_dict[key]))]
    concatenated = noise_norm_corr_ROI_concatenated.assign(key=key_values)
    concatenated['formatted'] = concatenated.apply(lambda row: "Subj {} {} {}: {:.2f}".format(row['key'], row['Hemisphere'], row['ROIs'], row['Median Noise Normalized Encoding Accuracy']), axis=1)
    concatenated['formatted'].to_csv(os.path.join(parent_submission_dir,'val_scores_subj_hemisphere_ROI.txt'), index=False, header=False)
    # Final subject/hemisphere-wise boxplot and istograms on validation
    final_subj_corr_dataframe_boxplot_istograms(noise_norm_corr_dict,
                                                submission_name,
                                                parent_submission_dir)


end_time = time.time()
total_time = end_time - start_time

print("Execution time: ", total_time/60, " min")        
        
        
        
        
        
        
        
        
        
        
        
        