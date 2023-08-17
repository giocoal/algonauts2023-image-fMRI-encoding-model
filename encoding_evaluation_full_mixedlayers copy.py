### Packages import
import os
import gc
import time
from time import strftime
import sys
print(sys.maxsize > 2**32)
start_time = time.time()

import json

import numpy as np
import pandas as pd

import torch
from torchvision import models
from src.cuda_checker import cuda_torch_check, memory_checker

### My modules import
from src.data_loader import argObj, data_loaders_stimuli_fmri, transform_layers, load_roi_masks_challenge_from_list_of_ROI
from src import image_preprocessing
from src.feature_extraction import model_loader, fit_pca, pca_batch_calculator, extract_and_pca_features, extract_features_no_pca
from src.encoding import linear_regression, ridge_alpha_grid_search
from src.evaluation_metrics import median_squared_noisenorm_correlation
from src.visualize import histogram, box_plot, noise_norm_corr_ROI, final_subj_corr_dataframe_boxplot_istograms, color, noise_norm_corr_ROI_df, find_best_performing_layer, json_config_to_feature_extraction_dict

### Cuda setup and check
import torch

import argparse

# Define argument parser
def str_to_func(arg):
    switch = {
        "imagenet_transform_alt": image_preprocessing.imagenet_transform_alt,
        "some_other_transform": some_other_transform,
        "yet_another_transform": yet_another_transform,
    }
    return switch.get(arg, lambda: "Invalid transform")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Final script which predicts BOLD responses in a two-step fashion.')
    parser.add_argument('--run_mode', default='config', 
                        type=str, help='Run mode [\'argparse\', \'config\']')
    parser.add_argument('--device', default='cuda', 
                        type=str, help='Device to use in pytorch [\'cpu\', \'cuda\']')
    parser.add_argument('--transform', type=str, default='imagenet_transform_alt', 
                        help='Image transform to use')
    parser.add_argument('--train_percentage', default=90, 
                        type=int, help='Xx100 of the training data will be used for training, (100-X)x100 for validation')
    parser.add_argument('--batch_size', default=64, 
                        type=int, help='Batch size for training')
    parser.add_argument('--pca_component', default=100, 
                        type=int, help='PCA components')
    parser.add_argument('--min_pca_batch_size', default=400, 
                        type=int, help='Minimum PCA batch size')
    parser.add_argument('--compute_pca', default=True, 
                        type=bool, help='Whether to compute PCA or not')
    # parser.add_argument('--feature_model_type', default='alexnet', 
    #                     type=str, help='Type of feature model')
    # parser.add_argument('--model_layer', default='features.12', 
    #                     type=str, help='Model layer')
    parser.add_argument('--# ', default='mixed', 
                        type=str, help='Combining mode [\'single\', \'concat\', \'mixed\']')
    parser.add_argument('--save', default=True, 
                        type=bool, help='Save the model or not')
    parser.add_argument('--config_file_mode', default='global', 
                        type=str, help='Name of the config file')
    parser.add_argument('--extraction_config_file', default='config.json', 
                        type=str, help='Name of the config file')
    parser.add_argument('--regression_type', default='linear', 
                        type=str, help='Type of regression')
    parser.add_argument('--alpha_l', default=1e5, 
                        type=float, help='Alpha value for left hemisphere')
    parser.add_argument('--alpha_r', default=1e5, 
                        type=float, help='Alpha value for right hemisphere')
    parser.add_argument('--grid_search', default=False, 
                        type=bool, help='Whether to do grid search or not')
    parser.add_argument('--data_home_dir', default='../Datasets/Biomedical', 
                        type=str, help='Path to the data home directory')
    parser.add_argument('--data_dir', default='../Datasets/Biomedical/algonauts_2023_challenge_data', 
                        type=str, help='Path to the data directory')
    parser.add_argument('--ncsnr_dir', default='../Datasets/Biomedical/algonauts_ncsnr', 
                        type=str, help='Path to the ncsnr directory')
    parser.add_argument('--images_trials_dir', default='../Datasets/Biomedical/algonauts_train_images_trials', 
                        type=str, help='Path to the images trials directory')
    args = parser.parse_args()

    # Use the arguments
    run_mode = args.run_mode
    
    if run_mode == 'argparse':
        device = args.device
        # Check if cuda is available
        device = torch.device(device)
        cuda_torch_check()
        
        transform = str_to_func(args.transform)
        train_percentage = args.train_percentage
        batch_size = args.batch_size
        pca_component = args.pca_component
        min_pca_batch_size = args.min_pca_batch_size
        compute_pca = args.compute_pca
        #  = args.# 
        feature_model_type = args.feature_model_type
        model_layer = args.model_layer
        save = args.save
        extraction_config_file = args.extraction_config_file
        regression_type = args.regression_type
        alpha_l = args.alpha_l
        alpha_r = args.alpha_r
        grid_search = args.grid_search
    elif run_mode == 'config':
        # Select the device to run the model on
        device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
        # Check if cuda is available
        device = torch.device(device)
        cuda_torch_check()

        ### Parameters definition
        train_percentage = 90 # X% of the training data will be used for training, (100-X)% for validation
        transform = image_preprocessing.imagenet_transform_alt

        batch_size = 64
        
        compute_pca = True
        pca_component = 2048
        min_pca_batch_size = pca_component + 300 # pca_component * 2
        
        # feature_model_type = "alexnet" #@param ["alexnet", "ZFNet", "RetinaNet","resnet50", "vgg16","vgg19_bn" , "efficientnetb2", "efficientnetb2lib"]
        # model_layers = ["features.2", "features.12"] 

        # create a dictionary to store the model layers
        # ["alexnet", "ZFNet", "resnet50", "vgg16","vgg19_bn" , "efficientnetb2", "efficientnetb2lib"]
        
        # combining_mode = 'concat' #@param ["single", "concat", "mixed"]
        # test_models_layers = {'alexnet': [['features.11', 'features.12'], 'features.2'],
        #                       'ZFNet': ['features.stage3.pool3']}
        def pca_selector(model):
            no_pca = ['DINOv2']
            yes_pca = ['RetinaNet', 'ZFNet', 'resnet50', 'alexnet', 'vgg16', 'vgg19', 'efficientnetb2']
            
            if model in yes_pca:
                print('PCA will be used')
                return True
            elif model in no_pca:
                print('PCA will not be used')
                return False
            else:
                print('Cannot determine if PCA will be used or not')
        
        def preprocessing_selector(model):
            imagenet_transform_alt_list = ['RetinaNet']
            imagenet_V2_transform_list = ['ZFNet', 'resnet50', 'alexnet', 'vgg16', 'vgg19']
            dinov2_transform_list = ['DINOv2']
            efficientnetb2_transform_list = ['efficientnetb2']
            
            if model in imagenet_transform_alt_list:
                print('Using imagenet_transform_alt')
                return image_preprocessing.imagenet_transform_alt
            elif model in imagenet_V2_transform_list:
                print('Using imagenet_V2_transform')
                return image_preprocessing.imagenet_V2_transform
            elif model in dinov2_transform_list:
                print('Using dinov2_transform')
                return image_preprocessing.dinov2_transform
            elif model in efficientnetb2_transform_list:
                print('Using efficientnetb2_transform')
                return image_preprocessing.efficientnetb2_transform
            else:
                print('No preprocessing selected, using imagenet_transform_alt')
                return image_preprocessing.imagenet_transform_alt
        
        test_models_layers = {'alexnet': [['features.10', 'features.11', 'features.12'], 'features.1', 'features.4', 'features.7', 'features.9', 'features.11'],
                              'DINOv2': ['DINOv2'],
                              'ZFNet': ['features.stage1.unit1.pool1', 'features.stage3.unit3.activ', 'features.stage3.pool3', 'features.stage3.unit1.activ', 'features.stage3.unit2.activ'],
                              'resnet50': ['layer1.0.relu', 'layer1.2.relu', 'layer2.3.relu', 'layer2.3.relu', 'layer3.5.relu', 'layer4.2.relu'],
                              'RetinaNet': ['fpn'],
                              'vgg16': ['avgpool'],
                              'vgg19': ['avgpool', 'features.33', 'features.35']}
        test_the_layers = True #@param ["True", "False"] 
        # retest all the layers even if they have been tested before
        force_test = True #@param ["True", "False"]

        # Config file name (global will point to the "global" config folder)
        config_file_mode = 'global' #@param ["global", "local"]
        extraction_config_file = 'config_0.45019.json'

        # Create a submission folder and save the resulting files ?
        save = True 

        regression_type = "ridge" #@param ["linear", "ridge"]
        grid_search = True
        alpha_l = 1e5
        alpha_r = 1e5
        params_grid = {'alpha': [1, 10, 100, 1e3, 1e4, 2e4, 5e4, 1e5, 1e6, 2e6]}
        

    ### Path definition
    model_layer_full = '_'.join([
        '{}_{}'.format(model.upper(), '+'.join(transform_layers(layers)))
        for model, layers in test_models_layers.items()
    ])

    datetime_id = strftime("(%Y-%m-%d_%H-%M)")
    submission_name = f'{strftime("(%Y-%m-%d_%H-%M)")}-{model_layer_full}-PCA_{pca_component}-{regression_type.upper()}-ALPHA_{"{:.1e}".format(alpha_l)}'


    ### Path definition
    # Data folder definition
    data_home_dir = '../Datasets/Biomedical'
    data_dir = '../Datasets/Biomedical/algonauts_2023_challenge_data'
    ncsnr_dir = '../Datasets/Biomedical/algonauts_ncsnr'
    images_trials_dir = '../Datasets/Biomedical/algonauts_train_images_trials'
    
    ### Paths and folder (mainly for the test part)
    # Used to save the prediction of saved model
    config_name = '+'.join(test_models_layers.keys()) # + datetime_id
    
    ############ TEMPORANEO
    submission_name = config_name + "_" + str.upper(regression_type) + f'{strftime("(%Y-%m-%d)")}'
    parent_submission_dir = f'./files/submissions/{submission_name}'
    # folders where the config files will be saved (global and local best performing layers)
    parent_config_dir = f'./files/config/{config_name}'
    global_config_dir = './files/config/global'
    if not os.path.isdir(parent_submission_dir + '_TEST') and save and test_the_layers:
        parent_submission_dir = parent_submission_dir + '_TEST'
        os.makedirs(parent_submission_dir)
    if not os.path.isdir(parent_config_dir) and save and test_the_layers:
        os.makedirs(parent_config_dir)
    print(submission_name + "\n")
    
    # define the dictionary to store the correlation values
    noise_norm_corr_dict = {}
    noise_norm_corr_ROI_dict = {} 
    for subj in list(range(1, 9)):
        print(f'############################ {color.BOLD + color.RED} Subject: {str(subj)} {color.END + color.END} ############################ \n')
        if test_the_layers == True:
            '''
            If test_the_layers is True, we will test the layers of the models defined in test_models_layers one by one.
            At the end of the testing procedure, we will save the correlation values and visualitazions for each layer
            separately. 
            One single config file will be created at the end of the testing procedure, associating the best layer for
            each ROI. 
            '''
            print(f'######## Starting the {color.RED} LAYERS TESTING PROCEDURE {color.END} (limited to subj 1) ######## \n')
            totale_layers_to_test = sum(len(lista) for lista in test_models_layers.values())
            counter_layers_to_test = 0
            first_iteration = True
            
            # Importing the df containing ROI_wise accuracy for all the layers tested in the past
            median_roi_correlation_df_global = pd.read_csv(os.path.join(global_config_dir, 
                                                                        f"scores_subj_layer_roi_{regression_type}.csv"), index_col=0)
            
            alpha_df_global = pd.read_csv(os.path.join(global_config_dir, 
                                                                        "alpha_subj_layer.csv"), index_col=0)
            # Starting to iterate overe the models and layers to test
            for feature_model_type, model_layers in test_models_layers.items():
                for model_layer in model_layers:
                    counter_layers_to_test += 1
                    print(f'\n######## Testing Model: {color.BOLD + feature_model_type + color.END} Layer(s): {color.BOLD + str(model_layer) + color.END} {counter_layers_to_test}/{totale_layers_to_test} ######## \n')
                    # first of all i verify that the couple model+layer has never been tested before
                    # in positive case i will use the old score and do not re-test the specific layer (minimum variation in accuracy is exected)
                    # Definining paths to data and submission directories ##
                    args = argObj(subj, data_home_dir, data_dir, parent_submission_dir, ncsnr_dir, images_trials_dir, save)
                    if isinstance(model_layer, str):
                        model_layer_id = f'{feature_model_type}+{model_layer}'
                    else:
                        model_layer_id = f'{feature_model_type}+{"&".join(model_layer)}'
                    transform = preprocessing_selector(feature_model_type)
                    if model_layer_id in median_roi_correlation_df_global.index and not force_test:
                        print(f'The model-layer couple {model_layer_id} has already been tested in the past. \n')
                        if first_iteration:
                            # if first iteration, create the dataframe containing the ROI-wise scores for each model-layer couple
                            median_roi_correlation_df = median_roi_correlation_df_global.loc[[model_layer_id]]
                            if model_layer_id in alpha_df_global.index:
                                alpha_df = alpha_df_global.loc[[model_layer_id]]
                            first_iteration = False
                            continue # skip the computing
                        else:
                            # if not extract the already tested model-layer couple from the global dataframe and append it to the local one
                            median_roi_correlation_df = pd.concat([median_roi_correlation_df,
                                                                    median_roi_correlation_df_global.loc[[model_layer_id]]])
                            if model_layer_id in alpha_df_global.index:
                                alpha_df = pd.concat([alpha_df, alpha_df_global.loc[[model_layer_id]]])
                            continue 
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
                    
                    try:
                        model, feature_extractor = model_loader(feature_model_type, model_layer, device)
                    except ValueError:
                        print(f'{color.RED} Model not found! {color.END}')
                        continue 
                    
                    compute_pca = pca_selector(feature_model_type)
                    
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
                    
                    
                    # lh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_val_pred, rh_fmri_test_pred = linear_regression("linear", 
                    #                                                                                                 features_train, 
                    #                                                                                                 features_val, 
                    #                                                                                                 features_test, 
                    #                                                                                                 lh_fmri_train, 
                    #                                                                                                 rh_fmri_train, 
                    #                                                                                                 save,
                    #                                                                                                 args.subject_test_submission_dir,
                    #                                                                                                 alpha_l,
                    #                                                                                                 alpha_r,
                    #                                                                                                 grid_search)
                    
                    # now i optimize the alpha 
                    if grid_search:
                        best_alpha_l, best_alpha_r = ridge_alpha_grid_search(features_train,  
                                                                    lh_fmri_train, 
                                                                    rh_fmri_train,
                                                                    params_grid)
                        alpha_l, alpha_r = best_alpha_l, best_alpha_r
                    
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
                                                                                                                    grid_search = False)
                    
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
                    print("LH subj",subj,"| Score: ",np.nanmedian(noise_norm_corr_dict[f'lh_{subj}'])*100)
                    print("RH subj",subj,"| Score: ",np.nanmedian(noise_norm_corr_dict[f'rh_{subj}'])*100)

                    ## Saving graphs showing accuracy score indipendently for each ROI, hemisphere of the subject ##
                    if save:
                        histogram(args.data_dir, noise_norm_corr_dict[f'lh_{subj}'], 
                                    noise_norm_corr_dict[f'rh_{subj}'], 
                                    f'{submission_name}_subj{subj}', 
                                    save = args.subject_val_images_submission_dir,
                                    filename = model_layer_id)
                        box_plot(args.data_dir, noise_norm_corr_dict[f'lh_{subj}'], 
                                    noise_norm_corr_dict[f'rh_{subj}'], 
                                    f'{submission_name}_subj{subj}', 
                                    save = args.subject_val_images_submission_dir,
                                    filename = model_layer_id)
                        
                    if first_iteration:
                        # if first iteration, create the dataframe containing the ROI-wise scores for each model-layer couple
                        median_roi_correlation_df = noise_norm_corr_ROI_df(args.data_dir,
                                                                           noise_norm_corr_dict[f'lh_{subj}'],
                                                                           noise_norm_corr_dict[f'rh_{subj}'],
                                                                           model_layer_id)
                        first_iteration = False
                        if grid_search:
                            alpha_df = pd.DataFrame({
                                'alpha_l': pd.Series([best_alpha_l], index=[model_layer_id]),
                                'alpha_r': pd.Series([best_alpha_r], index=[model_layer_id])
                            })
                            
                    else:
                        median_roi_correlation_df = pd.concat([median_roi_correlation_df,
                                                                noise_norm_corr_ROI_df(args.data_dir,
                                                                                        noise_norm_corr_dict[f'lh_{subj}'],
                                                                                        noise_norm_corr_dict[f'rh_{subj}'],
                                                                                        model_layer_id)])
                        if grid_search:
                            alpha_df_temp = pd.DataFrame({
                                'alpha_l': pd.Series([best_alpha_l], index=[model_layer_id]),
                                'alpha_r': pd.Series([best_alpha_r], index=[model_layer_id])
                            })
                            alpha_df = pd.concat([alpha_df, alpha_df_temp])
                    # add to the global ROI-wise score per model-layer couple the newly computed one
                    median_roi_correlation_df_global = pd.concat([median_roi_correlation_df_global, 
                                                                    noise_norm_corr_ROI_df(args.data_dir,
                                                                                            noise_norm_corr_dict[f'lh_{subj}'],
                                                                                            noise_norm_corr_dict[f'rh_{subj}'],
                                                                                            model_layer_id)])
                    # given the score for each ROI find and the save the best perfoming one
                    # progressively save the global dataframe containing the scores for each ROI for each model-layer couple ever tested
                    median_roi_correlation_df_global.sort_index(inplace=True)
                    median_roi_correlation_df_global.to_csv(os.path.join(global_config_dir, f"scores_subj_layer_roi_{regression_type}.csv"))
                    if grid_search:
                        alpha_df_global = pd.concat([alpha_df_global, alpha_df])
                        alpha_df_global.to_csv(os.path.join(global_config_dir, "alpha_subj_layer.csv"))
                        # alpha_df.to_csv(os.path.join(global_config_dir, "alpha_subj_layer.csv"))
                    extraction_config_overall = find_best_performing_layer(median_roi_correlation_df_global, global_config_dir, save, regression_type)
            # save median_roi_correlation_df
            print(f'######## {color.RED} TESTING PROCEDURE FINAL RESULTS {color.END} (limited to subj 1) ######## \n')
            if save:
                # given the score for each ROI find and the save the best perfoming one
                median_roi_correlation_df.to_csv(os.path.join(args.subject_val_images_submission_dir, f'scores_subj_layer_roi.csv'))
                # also save the global dataframe containing the scores for each ROI for each model-layer couple ever tested
                # ordering the rows in alphabetic order
                median_roi_correlation_df_global.sort_index(inplace=True)
                median_roi_correlation_df_global.to_csv(os.path.join(global_config_dir, f"scores_subj_layer_roi_{regression_type}.csv"))
                if grid_search:
                    alpha_df.to_csv(os.path.join(global_config_dir, "alpha_subj_layer.csv"))
            extraction_config = find_best_performing_layer(median_roi_correlation_df, parent_config_dir, save, regression_type)
            extraction_config_overall = find_best_performing_layer(median_roi_correlation_df_global, global_config_dir, save, regression_type)
            
            # reset the dictionaries for the inference
            noise_norm_corr_dict = {}
            noise_norm_corr_ROI_dict = {} 
        
        print(f'######## Starting the {color.RED} ENCODING PROCEDURE {color.END} ######## \n')
        # importing the correct config file that will define all the model-layers used
        if config_file_mode == 'global' and test_the_layers:
            # if i want the global best performing config using the one i just tested
            final_extraction_config = extraction_config_overall
        elif config_file_mode == 'global' and not test_the_layers:
            # importing the global best performing config
            with open(os.path.join(global_config_dir, extraction_config_file), 'r') as json_file:
                final_extraction_config = json.load(json_file)
        elif config_file_mode == 'local' and test_the_layers:
            # if i want the local best performing config using the one i just tested
            final_extraction_config = extraction_config
        elif config_file_mode == 'local' and not test_the_layers:
            # importing the local best performing config
            with open(os.path.join(parent_config_dir, extraction_config_file), 'r') as json_file:
                final_extraction_config = json.load(json_file)
        else:
            print('Error: config_file_mode not recognized')
            sys.exit()
        
        # remove useless ROIs from the config file
        keys_to_remove = ['All vertices', 'Unknown ROI', 'Unknown Stream']
        for key in keys_to_remove:
            if key in final_extraction_config:
                del final_extraction_config[key]
                
        # generating the iterable dict model:["layer1", ['layer1', 'layer2']] dictionary from the config file
        final_models_layers = json_config_to_feature_extraction_dict(final_extraction_config)
        
        # Creating the inference folders and path
        model_layer_full = '_'.join([
            '{}_{}'.format(model.upper(), '+'.join(transform_layers(layers)))
            for model, layers in final_models_layers.items()
        ])
        datetime_id = strftime("(%Y-%m-%d_%H-%M)")
        submission_name = f'{strftime("(%Y-%m-%d_%H-%M)")}-{model_layer_full}-PCA_{pca_component}-{regression_type.upper()}-ALPHA_{"{:.1e}".format(alpha_l)}'
        parent_submission_dir = f'./files/submissions/{submission_name}'
        if not os.path.isdir(parent_submission_dir) and save:
            os.makedirs(parent_submission_dir)
        print(submission_name + "\n")
        
        # Loop parameters
        totale_layers_to_test = sum(len(lista) for lista in final_models_layers.values())
        counter_layers_to_test = 0
        first_iteration = True
        
        # load alpha df
        alpha_df_global = pd.read_csv(os.path.join(global_config_dir, 
                                                    "alpha_subj_layer.csv"), index_col=0)
        
        # Starting to iterate overe the models and layers to use
        first_iteration = True
        for feature_model_type, model_layers in final_models_layers.items():
            for model_layer in model_layers:
                counter_layers_to_test += 1
                print(f'######## Computing Model: {color.BOLD + feature_model_type + color.END} Layer(s): {color.BOLD + str(model_layer) + color.END} {counter_layers_to_test}/{totale_layers_to_test} ######## \n')
                # first of all i verify that the couple model+layer has never been tested before
                # in positive case i will use the old score and do not re-test the specific layer (minimum variation in accuracy is exected)
                transform = preprocessing_selector(feature_model_type)
                if isinstance(model_layer, str):
                    model_layer_id = f'{feature_model_type}+{model_layer}'
                else:
                    model_layer_id = f'{feature_model_type}+{"&".join(model_layer)}'
                print('############################ Model Layer: ' + model_layer + ' ############################ \n')
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
                
                try:
                    model, feature_extractor = model_loader(feature_model_type, model_layer, device)
                except ValueError:
                    print(f'{color.RED} Model not found! {color.END}')
                    continue 
                
                compute_pca = pca_selector(feature_model_type)
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
                
                # Import the best hyperparameters for the current model-layer-specific set of ROIs
                if model_layer_id in alpha_df_global.index:
                    alpha_l = alpha_df_global.loc[model_layer_id, 'alpha_l']
                    alpha_r = alpha_df_global.loc[model_layer_id, 'alpha_r']
                else:
                    print(f'{color.RED} Alpha values not found! {color.END}, using the default ones')
                
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
                                                                                                                grid_search = False)
                # generate the create the voxel mask for the current model-layer-specific set of ROIs
                lh_roi_mask, rh_roi_mask = load_roi_masks_challenge_from_list_of_ROI(args.roi_dir_enhanced, final_extraction_config, [feature_model_type, model_layer])
                
                if first_iteration:
                    # create the final prediction matrices, setting nan for voxel outside the specified ROIs
                    first_iteration = False
                    lh_fmri_val_pred_final = np.copy(lh_fmri_val_pred)
                    lh_fmri_val_pred_final[:, lh_roi_mask == 0] = np.nan
                    lh_fmri_test_pred_final = np.copy(lh_fmri_test_pred)
                    lh_fmri_test_pred_final[:, lh_roi_mask == 0] = np.nan
                    rh_fmri_val_pred_final = np.copy(rh_fmri_val_pred)
                    rh_fmri_val_pred_final[:, rh_roi_mask == 0] = np.nan
                    rh_fmri_test_pred_final = np.copy(rh_fmri_test_pred)
                    rh_fmri_test_pred_final[:, rh_roi_mask == 0] = np.nan
                    del lh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_val_pred, rh_fmri_test_pred
                    gc.collect()
                else:
                    # add the current model-layer-specific predictions to the final prediction matrices
                    lh_fmri_val_pred_final = np.where(lh_roi_mask == 1, lh_fmri_val_pred, lh_fmri_val_pred_final)
                    lh_fmri_test_pred_final = np.where(lh_roi_mask == 1, lh_fmri_test_pred, lh_fmri_test_pred_final)
                    rh_fmri_val_pred_final = np.where(rh_roi_mask == 1, rh_fmri_val_pred, rh_fmri_val_pred_final)
                    rh_fmri_test_pred_final = np.where(rh_roi_mask == 1, rh_fmri_test_pred, rh_fmri_test_pred_final)
                    del lh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_val_pred, rh_fmri_test_pred
                    gc.collect()
        
        if np.isnan(lh_fmri_val_pred_final).sum() > 0:
            print(f"predictons contain {np.isnan(lh_fmri_val_pred_final).sum()} nan")
         
        noise_norm_corr_dict[f'lh_{subj}'], noise_norm_corr_dict[f'rh_{subj}'] = median_squared_noisenorm_correlation(lh_fmri_val_pred_final, 
                                                                                                                        rh_fmri_val_pred_final,
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
        print("LH subj",subj,"| Score: ",np.nanmedian(noise_norm_corr_dict[f'lh_{subj}'])*100)
        print("RH subj",subj,"| Score: ",np.nanmedian(noise_norm_corr_dict[f'rh_{subj}'])*100)

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
        if test_the_layers:
            print("Testing rotine is completed and executed only for the first subject. Exiting...")
            sys.exit()  # Termina completamente l'esecuzione dello script
        
print("#########################")
print("##### FINAL RESULTS #####")
print("#########################")
print("#########################") 
print("Median Noise Normalized Squared Correlation Percentage (LH and RH) for each subject")    
for key, value in noise_norm_corr_dict.items():
    print("Subject ->",key,"| Score ->",np.nanmedian(value)*100)
    
concatenated_correlations = np.concatenate(list(noise_norm_corr_dict.values()))
print("#########################")
print("#########################")
print("#########################")
print("#########################")
print("Median Noise Normalized Squared Correlation Percentage on all subjects")
print("Concatenated Subjs | Score: ",np.nanmedian(concatenated_correlations)*100)

# Save the results in an output file and figures
if save:
    # Final single score on validation
    with open(os.path.join(parent_submission_dir, 'val_scores.txt'), 'a') as f:
        f.write(f'All_vertices: {np.nanmedian(concatenated_correlations)*100}')
    # Final subject/hemisphere-wise score on validation results save
    with open(os.path.join(parent_submission_dir, 'val_scores_subj_hemisphere.txt'), 'a') as f:
        for key, value in noise_norm_corr_dict.items():
            f.write(f'{key}: {np.nanmedian(value)*100}\n')
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
        
        
        
        
        
        
        
        
        
        
        
        