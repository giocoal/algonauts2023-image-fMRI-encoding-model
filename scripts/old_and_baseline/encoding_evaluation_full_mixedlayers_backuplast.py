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
import pickle

import torch
from torchvision import models
from src.cuda_checker import cuda_torch_check, memory_checker

### My modules import
from src.data_loader import argObj, data_loaders_stimuli_fmri, transform_layers, transform_layers_test, load_roi_masks_challenge_from_list_of_ROI
from src import image_preprocessing
from src.feature_extraction import model_loader, fit_pca, pca_batch_calculator, extract_and_pca_features, extract_features_no_pca
from src.encoding import linear_regression, ridge_alpha_grid_search
from src.evaluation_metrics import median_squared_noisenorm_correlation
from src.visualize import histogram, box_plot, noise_norm_corr_ROI, final_subj_corr_dataframe_boxplot_istograms, color, noise_norm_corr_ROI_df, find_best_performing_layer, json_config_to_feature_extraction_dict, median_squared_noisenorm_correlation_dataframe_results

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
        def pca_selector(model_and_layer):
            no_pca_model = ['DINOv2s','DINOv2b','DINOv2l','DINOv2g']
            no_pca = [['DINOv2', 'DINOv2'], ['efficientnetb2', 'avgpool'], ['efficientnetb2', 'flatten']]
            yes_pca = ['RetinaNet', 'ZFNet', 'resnet50', 'alexnet', 'vgg16', 'vgg19', 'efficientnetb2']
            
            if (model_and_layer in no_pca) or (model_and_layer[0] == 'DINOv2s') or ((model_and_layer[0] in no_pca_model) and (not isinstance(model_and_layer[1], list))):
                print('PCA will not be used')
                return False
            elif ((model_and_layer[0] == 'DINOv2b') or (model_and_layer[0] == 'DINOv2l')) and (isinstance(model_and_layer[1], list)): 
                if len(model_and_layer[1]) > 2:
                    print('PCA will be used')
                    return True
                else:
                    print('PCA will not be used')
                    return False
            else:
                print('PCA will be used')
                return True
            # elif model_and_layer in yes_pca:
            #     print('PCA will be used')
            #     return False
            # else:
            #     print('Cannot determine if PCA will be used or not')
        
        transform_dict = {'imagenet_transform_alt': image_preprocessing.imagenet_transform_alt,
                            'imagenet_V2_transform': image_preprocessing.imagenet_V2_transform,
                            'dinov2_transform': image_preprocessing.dinov2_transform,
                            'efficientnetb2_transform': image_preprocessing.efficientnetb2_transform,
                            'imagenet_V1_transform': image_preprocessing.imagenet_V1_transform,
                            'dinov2_transform_V2': image_preprocessing.dinov2_transform_V2,
                            'dino_resnet50_preprocess': image_preprocessing.dino_resnet50_preprocess,
                            'ViT_GPT2': image_preprocessing.ViT_GPT2_transform}
        
        def test_preprocessing_selector(model):
            imagenet_transform_alt_list = ['RetinaNet','ZFNet', 'resnet50', 'vgg19','resnet50','alexnet', 'vgg16']
            imagenet_V2_transform_list = []
            dinov2_transform_list = []
            dinov2_transform_V2_list = ['DINOv2s','DINOv2b','DINOv2l','DINOv2g']
            efficientnetb2_transform_list = ['efficientnetb2']
            dino_resnet50_preprocess_list = ['dino_res50']
            ViT_GPT2_transform_list = ['ViT_GPT2']
            
            if model in imagenet_transform_alt_list:
                print('Using imagenet_transform_alt')
                return image_preprocessing.imagenet_transform_alt, 'imagenet_transform_alt'
            elif model in imagenet_V2_transform_list:
                print('Using imagenet_V2_transform')
                return image_preprocessing.imagenet_V2_transform, 'imagenet_V2_transform'
            elif model in dinov2_transform_list:
                print('Using dinov2_transform')
                return image_preprocessing.dinov2_transform, 'dinov2_transform'
            elif model in dino_resnet50_preprocess_list:
                print('Using dino_resnet50_preprocess')
                return image_preprocessing.dino_resnet50_preprocess, 'dino_resnet50_preprocess'
            elif model in dinov2_transform_V2_list:
                print('Using dinov2_transform_V2_list')
                return image_preprocessing.dinov2_transform_V2, 'dinov2_transform_V2'
            elif model in efficientnetb2_transform_list:
                print('Using efficientnetb2_transform')
                return image_preprocessing.efficientnetb2_transform, 'efficientnetb2_transform'
            elif model in ViT_GPT2_transform_list:
                print('Using ViT_GPT2_transform')
                return image_preprocessing.ViT_GPT2_transform, 'ViT_GPT2_transform'
            else:
                print('No preprocessing selected, using imagenet_transform_alt')
                return image_preprocessing.imagenet_transform_alt, 'imagenet_transform_alt'
        
        # test_models_layers = {'DINOv2l': [['15','16','17','18','19'], ['19','20','21']],
        #                       'dino_res50': ['layer2.3.relu', 'layer4.0.relu', 'layer4.1.relu'],
        #                       'RetinaNet': ['body.layer3.5.relu_2','body.layer3.0.relu_2']}
        
        test_models_layers = {#'ViT_GPT2': [['decoder.9', 'decoder.10', 'decoder.11', 'decoder.12']],
                              'DINOv2s': [['0', '1', '2'],'2','DINOv2', '0', '1','3','4','5','6','7','8','9','10','11',['0', '1', '2'],['2', '3', '4'],['4', '5', '6'],['0', '1', '2', '3'],['0', '1', '2','3','4','5'], ['3','4','5'],['5', '6', '7','8','9'],['7', '8', '9','10','11'], ['9','10','11'], ['10','11']],
                              'DINOv2b': ['2','DINOv2', '0', '1','3','4','5','6','7','8','9','10','11',['0', '1', '2'],['2', '3', '4'],['4', '5', '6'],['0', '1', '2', '3'],['0', '1', '2','3','4','5'], ['3','4','5'],['5', '6', '7','8','9'],['7', '8', '9','10','11'], ['9','10','11'], ['10','11']],
                              'DINOv2l': ['2','DINOv2', '0', '1','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23',['0', '1', '2'],['2', '3', '4'],['4', '5', '6'],['0', '1', '2', '3'],['0', '1', '2','3','4','5'], ['3','4','5'],['5', '6', '7','8','9'],['7', '8', '9','10','11'], ['9','10','11'], ['11','12','13'], ['13','14','15'], ['11','12','13','14','15'], ['15','16','17'], ['17','18','19'], ['15','16','17','18','19'], ['19','20','21'], ['22','23'], ['19','20','21','22','23']],
                              'dino_res50': ['layer1.0.relu', 'layer1.1.relu', 'layer1.2.relu', 'layer2.1.relu', 'layer2.2.relu', 'layer2.3.relu', 'layer3.0.relu', 'layer3.1.relu', 'layer3.2.relu', 'layer3.3.relu', 'layer3.4.relu', 'layer3.5.relu', 'layer4.0.relu', 'layer4.1.relu', 'layer4.2.relu', 'avgpool'],
                              'alexnet': [['features.10', 'features.11', 'features.12'], 'features.1', 'features.4', 'features.7', 'features.9', 'features.11', 'features.12'],
                              'ZFNet': ['features.stage1.unit1.pool1', 'features.stage2.unit1.activ', 'features.stage3.unit3.activ', 'features.stage3.pool3', 'features.stage3.unit1.activ', 'features.stage3.unit2.activ'],
                              'RetinaNet': ['fpn', 'body.layer4.2.relu_2', 'body.layer4.2.relu_1', 'body.layer4.2.relu','body.layer3.5.relu_1', 'body.layer3.5.relu_2','body.layer3.0.relu_2', 'body.layer3.1.relu'],
                              'resnet50': ['layer1.0.relu', 'layer1.1.relu', 'layer1.2.relu', 'layer2.1.relu', 'layer2.2.relu', 'layer2.3.relu', 'layer3.0.relu', 'layer3.1.relu', 'layer3.2.relu', 'layer3.3.relu', 'layer3.4.relu', 'layer3.5.relu', 'layer4.0.relu', 'layer4.1.relu', 'layer4.2.relu', 'avgpool'], # 'layer2.0.relu',
                              'efficientnetb2': ['avgpool', 'flatten', 'features.8', 'features.7.0.block.3', 'features.6.4.block.3', 'features.3.0.block.2', 'features.3.2.block.3', 'features.4.1.block.3','features.5.1.block.2'],
                              'vgg16': ['avgpool'],
                              'vgg19': ['avgpool', 'features.33', 'features.35'],
                               # 'vgg16_bn': [ 'features.36','features.39','features.42','avgpool'],
                              'vgg19_bn': ['avgpool', 'features.51', 'features.45', 'features.42']}
        
        ## testing parameters
        test_the_layers = False #@param ["True", "False"] 
        # retest all the layers even if they have been tested before
        force_test = False #@param ["True", "False"]
        config_subj = 1
        extraction_config_folder = f'config_subj{config_subj}'
        
        ## Define the subject to test/inference  on
        start_subj = 6
        end_subj = 8

        ## config INFERENCE parameters 
        # Config file name (global will point to the "global" config folder)
        config_file_mode = 'global' #@param ["global", "local"]
        # which configuration to use
        config_dict = {'1': {'extraction_config_file' : 'config_0.64732.json', 'config_subj': 1},
                        '2': {'extraction_config_file' : 'config_0.64732.json', 'config_subj': 1},
                        '3': {'extraction_config_file' : 'config_0.64732.json', 'config_subj': 1},
                        '4': {'extraction_config_file' : 'config_0.64732.json', 'config_subj': 1},
                        '5': {'extraction_config_file' : 'config_0.64732.json', 'config_subj': 1},
                        '6': {'extraction_config_file' : 'config_0.55162.json', 'config_subj': 6},
                        '7': {'extraction_config_file' : 'config_0.55162.json', 'config_subj': 6},
                        '8': {'extraction_config_file' : 'config_0.55162.json', 'config_subj': 6}}
        
        # extraction_config_file = 'config_0.64732.json'

        # Create a submission folder and save the resulting files ?
        save = True 
        load_save_pca = True

        regression_type = "ridge" #@param ["linear", "ridge"]
        standardize_features = False #@param ["True", "False"]
        grid_search = True
        alpha_l = 1e5
        alpha_r = 1e5
        params_grid = {'alpha': [0.000001, 0.00001, 0.0001,0.001,0.01,0.1, 1, 10, 100, 1e3, 1e4, 2e4, 5e4, 1e5, 1e6, 2e6]}
        #params_grid = {'alpha': [1, 10, 100, 1e3, 1e4, 2e4, 5e4, 1e5, 1e6, 2e6]}

    ### Path definition
    model_layer_full = '_'.join([
        '{}_{}'.format(model.upper(), '+'.join(transform_layers_test(layers)))
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
    parent_config_dir = f'./files/{extraction_config_folder}/{config_name}'
    global_config_dir = f'./files/{extraction_config_folder}/global'
    pca_dir = './files/pca'
    if not os.path.isdir(parent_submission_dir + '_TEST') and save and test_the_layers:
        parent_submission_dir = parent_submission_dir + '_TEST'
        os.makedirs(parent_submission_dir)
    if not os.path.isdir(parent_config_dir) and save and test_the_layers:
        os.makedirs(parent_config_dir)
    print(submission_name + "\n")
    
    # define the dictionary to store the correlation values
    noise_norm_corr_dict = {}
    noise_norm_corr_ROI_dict = {} 
    datetime_id = strftime("(%Y-%m-%d_%H-%M)")
    for subj in list(range(start_subj, end_subj+1)):
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
                                                                        f"scores_subj_layer_roi.csv"), index_col=0)
            
            alpha_df_global = pd.read_csv(os.path.join(global_config_dir, 
                                                                        "alpha_subj_layer.csv"), index_col=0)
            # Starting to iterate overe the models and layers to test
            for feature_model_type, model_layers in test_models_layers.items():
                for model_layer in model_layers:
                    counter_layers_to_test += 1
                    transform, transform_string = test_preprocessing_selector(feature_model_type)
                    print(f'\n######## Testing Model: {color.BOLD + feature_model_type + color.END} Layer(s): {color.BOLD + str(model_layer) + color.END}  - Transform: {color.BOLD + str(transform_string) + color.END} - Regression Type: {color.BOLD + str(regression_type) + color.END} {counter_layers_to_test}/{totale_layers_to_test} ######## \n')
                    # first of all i verify that the couple model+layer has never been tested before
                    # in positive case i will use the old score and do not re-test the specific layer (minimum variation in accuracy is exected)
                    # Definining paths to data and submission directories ##
                    args = argObj(subj, data_home_dir, data_dir, parent_submission_dir, ncsnr_dir, images_trials_dir, save)
                    if isinstance(model_layer, str):
                        model_layer_id = f'{feature_model_type}+{model_layer}+{transform_string}+{regression_type}'
                    else:
                        model_layer_id = f'{feature_model_type}+{"&".join(model_layer)}+{transform_string}+{regression_type}'
                    
                    if model_layer_id in median_roi_correlation_df_global.index and not force_test:
                        print(color.BLUE + f'This model-layer-params: {model_layer_id} has already been tested in the past. \n' + color.END)
                        if first_iteration:
                            first_iteration = False
                            # if first iteration, create the dataframe containing the ROI-wise scores for each model-layer couple
                            median_roi_correlation_df = median_roi_correlation_df_global.loc[[model_layer_id]]
                            if model_layer_id in alpha_df_global.index:
                                alpha_df = alpha_df_global.loc[[model_layer_id]]
                            else:
                                print(color.BLUE + "Alpha has not been found" + color.END)
                            continue # skip the computing
                        else:
                            # if not extract the already tested model-layer couple from the global dataframe and append it to the local one
                            median_roi_correlation_df = pd.concat([median_roi_correlation_df,
                                                                    median_roi_correlation_df_global.loc[[model_layer_id]]])
                            if model_layer_id in alpha_df_global.index:
                                alpha_df = pd.concat([alpha_df, alpha_df_global.loc[[model_layer_id]]])
                            else:
                                print(color.BLUE + "Alpha has not been found" + color.END)
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
                    
                    compute_pca = pca_selector([feature_model_type, model_layer])
                    
                    start_time_feature_extraction = time.time()
                    
                    if compute_pca:
                        # Fit the PCA model
                        pca_batch_size, n_stacked_batches = pca_batch_calculator(len(idxs_train),
                                                                                batch_size,
                                                                                min_pca_batch_size,
                                                                                pca_component)
                        # Load pca model if it exists, otherwise fit it
                        pca_path = os.path.join(pca_dir, f'{model_layer_id}_pca_{pca_component}.pkl')
                        if os.path.exists(pca_path) and load_save_pca:
                            print(f"\n Loading PCA model from: {pca_path}\n")
                            with open(pca_path, 'rb') as pickle_file:
                                pca = pickle.load(pickle_file)
                            del pickle_file
                        else:
                            print(f"\n PCA Model not found or load_save_pca = False, fitting ...\n")
                            pca = fit_pca(feature_extractor,
                                            train_imgs_dataloader,
                                            pca_component,
                                            n_stacked_batches,
                                            pca_batch_size,
                                            device)
                            if load_save_pca:
                                with open(pca_path, 'wb') as pickle_file:
                                    pickle.dump(pca, pickle_file)
                                del pickle_file
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
                        try:
                            feature_extractor.to('cpu') # sposto sulla ram
                        except:
                            pass
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
                        try:
                            feature_extractor.to('cpu') # sposto sulla ram
                        except:
                            pass
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
                                                                    params_grid,
                                                                    UseStandardScaler = standardize_features)
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
                                                                                                                    grid_search = False,
                                                                                                                    UseStandardScaler = standardize_features)
                    
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
                        first_iteration = False
                        # if first iteration, create the dataframe containing the ROI-wise scores for each model-layer couple
                        median_roi_correlation_df = noise_norm_corr_ROI_df(args.data_dir,
                                                                           noise_norm_corr_dict[f'lh_{subj}'],
                                                                           noise_norm_corr_dict[f'rh_{subj}'],
                                                                           model_layer_id)
                        if grid_search:
                            alpha_df_temp = pd.DataFrame({
                                'alpha_l': pd.Series([best_alpha_l], index=[model_layer_id]),
                                'alpha_r': pd.Series([best_alpha_r], index=[model_layer_id])
                            })
                            alpha_df = alpha_df_temp
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
                    median_roi_correlation_df_global.to_csv(os.path.join(global_config_dir, f"scores_subj_layer_roi.csv"))
                    if grid_search:
                        alpha_df_global = pd.concat([alpha_df_global, alpha_df_temp])
                        alpha_df_global.sort_index(inplace=True)
                        alpha_df_global.to_csv(os.path.join(global_config_dir, "alpha_subj_layer.csv"))
                        # alpha_df.to_csv(os.path.join(global_config_dir, "alpha_subj_layer.csv"))
                    extraction_config_overall = find_best_performing_layer(median_roi_correlation_df_global, global_config_dir, save)
            # save median_roi_correlation_df
            print(f'######## {color.RED} TESTING PROCEDURE FINAL RESULTS {color.END} (limited to subj 1) ######## \n')
            if save:
                # given the score for each ROI find and the save the best perfoming one
                median_roi_correlation_df.to_csv(os.path.join(args.subject_val_images_submission_dir, f'scores_subj_layer_roi.csv'))
                # also save the global dataframe containing the scores for each ROI for each model-layer couple ever tested
                # ordering the rows in alphabetic order
                median_roi_correlation_df_global.sort_index(inplace=True)
                median_roi_correlation_df_global.to_csv(os.path.join(global_config_dir, f"scores_subj_layer_roi.csv"))
                if grid_search:
                    alpha_df_global.sort_index(inplace=True)
                    alpha_df_global.to_csv(os.path.join(global_config_dir, "alpha_subj_layer.csv"))
            extraction_config = find_best_performing_layer(median_roi_correlation_df, parent_config_dir, save)
            extraction_config_overall = find_best_performing_layer(median_roi_correlation_df_global, global_config_dir, save)
            
            # reset the dictionaries for the inference
            noise_norm_corr_dict = {}
            noise_norm_corr_ROI_dict = {} 
        
        print(f'######## Starting the {color.RED} ENCODING PROCEDURE {color.END} ######## \n')
        # importing the correct config file that will define all the model-layers used
        extraction_config_file = config_dict[str(subj)]['extraction_config_file']
        config_subj = config_dict[str(subj)]['config_subj']
        extraction_config_folder = f'config_subj{config_subj}'
        parent_config_dir = f'./files/{extraction_config_folder}/{config_name}'
        global_config_dir = f'./files/{extraction_config_folder}/global'
        print(f'# Using config file: {color.BOLD} subj{config_subj}_{extraction_config_file} + {color.END}\n')
        if not os.path.isdir(parent_config_dir) and save and test_the_layers:
            os.makedirs(parent_config_dir)
        print(f'# Using config file: {color.BOLD} subj{config_subj}_{extraction_config_file} + {color.END} # \n')
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
        # datetime_id = strftime("(%Y-%m-%d_%H-%M)")
        submission_name = f'{datetime_id}-{model_layer_full}-PCA_{pca_component}-{regression_type.upper()}' #-ALPHA_{"{:.1e}".format(alpha_l)}'
        parent_submission_dir = f'./files/submissions/{submission_name}'
        if not os.path.isdir(parent_submission_dir) and save:
            os.makedirs(parent_submission_dir)
        print(submission_name + "\n")
        
        # Dataframe to save the results of the validation on every subjects
        val_results_csv_file_path = os.path.join(parent_submission_dir, 'val_results.csv')
        
        # Loop parameters
        totale_layers_to_test = sum(len(lista) for lista in final_models_layers.values())
        counter_layers_to_test = 0
        first_iteration = True
        
        # load alpha df
        alpha_df_global = pd.read_csv(os.path.join(global_config_dir, 
                                                    "alpha_subj_layer.csv"), index_col=0)
        
        # Starting to iterate overe the models and layers to use
        first_iteration = True
        # print final_models_layers dict with indentation
        print("Model: Layer(s) that will be used for the inference")
        print(json.dumps(final_models_layers, indent=4))
        for feature_model_type, model_layers in final_models_layers.items():
            for model_layer in model_layers:
                counter_layers_to_test += 1
                # first of all i verify that the couple model+layer has never been tested before
                # in positive case i will use the old score and do not re-test the specific layer (minimum variation in accuracy is exected)
                # divide the model layer name from preprocessing and regression type associated
                transform_string = model_layer[1]
                transform = transform_dict[transform_string]
                regression_type = model_layer[2]
                model_layer = model_layer[0]
                print(f'######## Computing Model: {color.BOLD + feature_model_type + color.END} Layer(s): {color.BOLD + str(model_layer) + color.END} - Transform: {color.BOLD + str(transform_string) + color.END} - Regression Type: {color.BOLD + str(regression_type) + color.END} {counter_layers_to_test}/{totale_layers_to_test} ######## \n')
                # transform, transform_string = test_preprocessing_selector(feature_model_type)
                if isinstance(model_layer, str):
                    model_layer_id = f'{feature_model_type}+{model_layer}+{transform_string}+{regression_type}'
                else:
                    model_layer_id = f'{feature_model_type}+{"&".join(model_layer)}+{transform_string}+{regression_type}'
                print('############################ Model Layer: ' + str(model_layer) + ' ############################ \n')
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
                compute_pca = pca_selector([feature_model_type, model_layer])
                if compute_pca:
                    # Fit the PCA model
                    pca_batch_size, n_stacked_batches = pca_batch_calculator(len(idxs_train),
                                                                            batch_size,
                                                                            min_pca_batch_size,
                                                                            pca_component)
                    pca_path = os.path.join(pca_dir, f'{model_layer_id}_pca_{pca_component}.pkl')
                    if os.path.exists(pca_path) and load_save_pca:
                        print(f"\n Loading PCA model from: {pca_path}\n")
                        with open(pca_path, 'rb') as pickle_file:
                            pca = pickle.load(pickle_file)
                        del pickle_file
                    else:
                        print(f"\n PCA model not found or load_save_pca = False, fitting ...\n")
                        pca = fit_pca(feature_extractor,
                                        train_imgs_dataloader,
                                        pca_component,
                                        n_stacked_batches,
                                        pca_batch_size,
                                        device)
                        if load_save_pca:
                            with open(pca_path, 'wb') as pickle_file:
                                pickle.dump(pca, pickle_file)
                            del pickle_file
        
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
                    try:
                        feature_extractor.to('cpu') # sposto sulla ram
                    except:
                        pass
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
                    try:
                        feature_extractor.to('cpu') # sposto sulla ram
                    except:
                        pass
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
                    print(f'{color.RED} Alpha values not found! {color.END}, starting grid search and update config.')
                    # Calculating GRID Search
                    best_alpha_l, best_alpha_r = ridge_alpha_grid_search(features_train,  
                                                                    lh_fmri_train, 
                                                                    rh_fmri_train,
                                                                    params_grid,
                                                                    UseStandardScaler = standardize_features)
                    alpha_l, alpha_r = best_alpha_l, best_alpha_r
                    # Updating the config file
                    alpha_df_temp = pd.DataFrame({
                                'alpha_l': pd.Series([best_alpha_l], index=[model_layer_id]),
                                'alpha_r': pd.Series([best_alpha_r], index=[model_layer_id])
                            })
                    alpha_df_global = pd.concat([alpha_df_global, alpha_df_temp])
                    alpha_df_global.sort_index(inplace=True)
                    alpha_df_global.to_csv(os.path.join(global_config_dir, "alpha_subj_layer.csv"))
                
                lh_fmri_val_pred, lh_fmri_test_pred, rh_fmri_val_pred, rh_fmri_test_pred = linear_regression(regression_type, 
                                                                                                                features_train, 
                                                                                                                features_val, 
                                                                                                                features_test, 
                                                                                                                lh_fmri_train, 
                                                                                                                rh_fmri_train, 
                                                                                                                save_predictions = False,
                                                                                                                subject_submission_dir = args.subject_test_submission_dir,
                                                                                                                alpha_l = alpha_l,
                                                                                                                alpha_r = alpha_r,
                                                                                                                grid_search = False,
                                                                                                                param_grid = params_grid,
                                                                                                                UseStandardScaler = standardize_features)

                # generate the create the voxel mask for the current model-layer-specific set of ROIs
                lh_roi_mask, rh_roi_mask = load_roi_masks_challenge_from_list_of_ROI(args.roi_dir_enhanced, final_extraction_config, [feature_model_type, model_layer, transform_string, regression_type])
                
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
        print("LH/RH subj",subj,"| Score: ",np.nanmedian(np.concatenate((noise_norm_corr_dict[f'lh_{subj}']*100,noise_norm_corr_dict[f'rh_{subj}']*100))))

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
            # Save predictions and correlations
            lh_fmri_test_pred_final = lh_fmri_test_pred_final.astype(np.float32)
            rh_fmri_test_pred_final = rh_fmri_test_pred_final.astype(np.float32)
            np.save(os.path.join(args.subject_test_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred_final)
            np.save(os.path.join(args.subject_test_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred_final)
            lh_fmri_val_pred_final = lh_fmri_val_pred_final.astype(np.float32)
            rh_fmri_val_pred_final = rh_fmri_val_pred_final.astype(np.float32)
            np.save(os.path.join(args.subject_val_submission_dir, 'lh_pred_val.npy'), lh_fmri_val_pred_final)
            np.save(os.path.join(args.subject_val_submission_dir, 'rh_pred_val.npy'), rh_fmri_val_pred_final)
            # correlations
            np.save(os.path.join(args.subject_val_correlation_dir, 'lh_corr_val.npy'), noise_norm_corr_dict[f'lh_{subj}'].astype(np.float32))
            np.save(os.path.join(args.subject_val_correlation_dir, 'rh_corr_val.npy'), noise_norm_corr_dict[f'rh_{subj}'].astype(np.float32))
            # save validation subj-specific results
            median_squared_noisenorm_correlation_dataframe_results(val_results_csv_file_path, noise_norm_corr_dict, str(subj))
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
        
        
        
        
        
        
        
        
        
        
        
        