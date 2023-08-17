### Packages import
import os
import gc
import time
start_time = time.time()

# print current working directory


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import models

import sys
# add current working directory to path
sys.path.append(os.getcwd())

from src.cuda_checker import cuda_torch_check, memory_checker

### My modules import
from src.data_loader import argObj, data_loaders_stimuli_fmri
from src import image_preprocessing
from src.feature_extraction import model_loader, fit_pca, pca_batch_calculator, extract_and_pca_features
from src.encoding import linear_regression, compute_perason_numpy
from src.evaluation_metrics import median_squared_noisenorm_correlation

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from src.visualize import histogram, box_plot

from nilearn import datasets
from nilearn import plotting
from nilearn import surface

hemispheres = ["left", "right"]

### Cuda setup and check
# import torch
# Select the device to run the model on
device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
# Check if cuda is available
device = torch.device(device)
cuda_torch_check()

### Parameters definition
train_percentage = 90 # X% of the training data will be used for training, (100-X)% for validation
transform = image_preprocessing.imagenet_transform_alt

batch_size = 64
pca_component = 512
min_pca_batch_size = pca_component * 2 # pca_component * 2

feature_model_type = "alexnet" #@param ["alexnet", "vgg16", "vgg19_bn, ""efficientnetb2", "efficientnetb2lib"]
model_layer = "features.2"
regression_type = "linear" #@param ["linear", "ridge"]

save_predictions = False

alpha_l = 1e6
alpha_r = 1e6
grid_search = False

subj = 1
noise_norm_corr_dict = {}

### Path definition
if isinstance(model_layer, list):
    model_layer_full = '+'.join(model_layer)
else:
    model_layer_full = model_layer
submission_name = f'{feature_model_type}_{model_layer}-pca_{pca_component}-{regression_type}-alpha_{alpha_l}'

# Data folder definition
data_home_dir = '../Datasets/Biomedical'
data_dir = '../Datasets/Biomedical/algonauts_2023_challenge_data'
# Used to save the prediction of saved model
parent_submission_dir = f'./files/submissions/{submission_name}'
images_submission_dir = f"./files/submissions/imgs/{submission_name}"
ncsnr_dir = '../Datasets/Biomedical/algonauts_ncsnr'
images_trials_dir = '../Datasets/Biomedical/algonauts_train_images_trials'

for subj in range(1,9):
    args = argObj(subj, data_home_dir, data_dir, parent_submission_dir, ncsnr_dir, images_trials_dir, images_submission_dir) 
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

    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(args.data_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []

    rh_challenge_roi_files_name_only = [s.split(".", 1)[1].replace("_challenge_space.npy", "") for s in rh_challenge_roi_files]

    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
    ### fsaverage space maps (same code as above, but for fsaverage space)

    lh_fsaverage_roi_files = ['lh.prf-visualrois_fsaverage_space.npy',
        'lh.floc-bodies_fsaverage_space.npy', 'lh.floc-faces_fsaverage_space.npy',
        'lh.floc-places_fsaverage_space.npy', 'lh.floc-words_fsaverage_space.npy',
        'lh.streams_fsaverage_space.npy']
    rh_fsaverage_roi_files = ['rh.prf-visualrois_fsaverage_space.npy',
        'rh.floc-bodies_fsaverage_space.npy', 'rh.floc-faces_fsaverage_space.npy',
        'rh.floc-places_fsaverage_space.npy', 'rh.floc-words_fsaverage_space.npy',
        'rh.streams_fsaverage_space.npy']
    lh_fsaverage_rois = []
    rh_fsaverage_rois = []

    rh_fsaverage_roi_files_name_only = [s.split(".", 1)[1].replace("_fsaverage_space.npy", "") for s in rh_fsaverage_roi_files]

    for r in range(len(lh_fsaverage_roi_files)):
        lh_fsaverage_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            lh_fsaverage_roi_files[r])))
        rh_fsaverage_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            rh_fsaverage_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_fsaverage_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_fsaverage_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_fsaverage_rois[r1] == r2[0])[0]
                
    hemisphere = 'left' #@param ['left', 'right'] {allow-input: true}
    roi_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    fsaverage_all_vertices = {}
    for hemisphere in hemispheres:
        roi_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
        fsaverage_all_vertices[f'{hemisphere[0]}h'] = np.load(roi_dir)
        
    hemispheres = ['left', 'right']
    rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    roi_classes = ['prf-visualrois', 'floc-bodies', 'floc-faces', 'floc-places', 'floc-words', 'streams']

    # Mask of all vertices in challenge space of the selected hemisphere
    fsaverage_all_vertices = {}
    for hemisphere in hemispheres:
        roi_dir = os.path.join(args.data_dir, 'roi_masks',
        hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
        fsaverage_all_vertices[f'{hemisphere[0]}h'] = np.load(roi_dir)
        
    # ROI classes labels *_*_roi_classes
    left_fsaverage_roi_classes = {}
    left_roi_maps = {}
    right_fsaverage_roi_classes = {}
    right_roi_maps = {}

    left_challenge_roi_classes = {}
    left_challenge_roi_maps = {}
    right_challenge_roi_classes = {}
    right_challenge_roi_maps = {}

    # Loading challe
    for roi_class in roi_classes:
        
        # Mapping is the same for both hemispheres and space
        roi_map_dir = os.path.join(args.roi_dir,
            'mapping_'+roi_class+'.npy')
        
        # fsaverage array masks (whith numbers)
        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[0][0]+'h.'+roi_class+'_fsaverage_space.npy')
        left_fsaverage_roi_classes[roi_class] = np.load(roi_class_dir)
        left_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()
        
        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[1][0]+'h.'+roi_class+'_fsaverage_space.npy')
        right_fsaverage_roi_classes[roi_class] = np.load(roi_class_dir)
        right_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()
        
        # challenge array masks
        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[0][0]+'h.'+roi_class+'_challenge_space.npy')
        left_challenge_roi_classes[roi_class] = np.load(roi_class_dir)
        left_challenge_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()
        
        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[1][0]+'h.'+roi_class+'_challenge_space.npy')
        right_challenge_roi_classes[roi_class] = np.load(roi_class_dir)
        right_challenge_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()
        
    left_fsaverage_roi_classes_masks = {}
    right_fsaverage_roi_classes_masks = {}
    left_challenge_roi_classes_masks = {}
    right_challenge_roi_classes_masks = {}

    for roi_class in roi_classes:
        left_fsaverage_roi_classes_masks[roi_class] = np.where(left_fsaverage_roi_classes[roi_class] != 0, 1, 0)
        right_fsaverage_roi_classes_masks[roi_class] = np.where(right_fsaverage_roi_classes[roi_class] != 0, 1, 0)
        left_challenge_roi_classes_masks[roi_class] = np.where(left_challenge_roi_classes[roi_class] != 0, 1, 0)
        right_challenge_roi_classes_masks[roi_class] = np.where(right_challenge_roi_classes[roi_class] != 0, 1, 0) 

    for roi_class in roi_classes:
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_classes_masks', hemispheres[0][0]+'h.'+roi_class+'_challenge_space.npy'), left_challenge_roi_classes_masks[roi_class])
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_classes_masks', hemispheres[1][0]+'h.'+roi_class+'_challenge_space.npy'), right_challenge_roi_classes_masks[roi_class])
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_classes_masks', hemispheres[0][0]+'h.'+roi_class+'_fsaverage_space.npy'), left_fsaverage_roi_classes[roi_class])
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_classes_masks', hemispheres[1][0]+'h.'+roi_class+'_fsaverage_space.npy'), right_fsaverage_roi_classes[roi_class])

    left_fsaverage_roi_masks = {}
    right_fsaverage_roi_masks = {}
    left_challenge_roi_masks = {}
    right_challenge_roi_masks = {}

    # # Load the ROI brain surface maps
    # roi_class_dir = os.path.join(args.roi_dir,
    #     hemisphere[0]+'h.'+roi_class+'_fsaverage_space.npy')
    # roi_map_dir = os.path.join(args.roi_dir,
    #     'mapping_'+roi_class+'.npy')
    # fsaverage_roi_class = np.load(roi_class_dir)
    # roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # # Select the vertices corresponding ONLY to the ROI of interest
    # roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
    # fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)


    for roi in rois:
        # Define the ROI class based on the selected ROI (for loading the ROI mask)
        if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = 'prf-visualrois'
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = 'floc-bodies'
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = 'floc-faces'
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_class = 'floc-places'
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = 'floc-words'
        elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
            roi_class = 'streams'
            
        # Mapping is the same for both hemispheres and space
        roi_map_dir = os.path.join(args.roi_dir,
            'mapping_'+roi_class+'.npy')

        #### fsaverage array masks (whith numbers)
        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[0][0]+'h.'+roi_class+'_fsaverage_space.npy')
        left_fsaverage_roi_classes[roi_class] = np.load(roi_class_dir)
        left_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()
        
        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[1][0]+'h.'+roi_class+'_fsaverage_space.npy')
        right_fsaverage_roi_classes[roi_class] = np.load(roi_class_dir)
        right_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()

        # challenge array masks
        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[0][0]+'h.'+roi_class+'_challenge_space.npy')
        left_challenge_roi_classes[roi_class] = np.load(roi_class_dir)
        left_challenge_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()

        roi_class_dir = os.path.join(args.roi_dir,
            hemispheres[1][0]+'h.'+roi_class+'_challenge_space.npy')
        right_challenge_roi_classes[roi_class] = np.load(roi_class_dir)
        right_challenge_roi_maps[roi_class] = np.load(roi_map_dir, allow_pickle=True).item()

        

        # And not the mask corresponding ONLY to the ROI of interest
        roi_mapping_temp = list(left_roi_maps[roi_class].keys())[list(left_roi_maps[roi_class].values()).index(roi)]
        left_fsaverage_roi_masks[roi] = np.asarray(left_fsaverage_roi_classes[roi_class] == roi_mapping_temp, dtype=int)
        
        roi_mapping_temp = list(right_roi_maps[roi_class].keys())[list(right_roi_maps[roi_class].values()).index(roi)]
        right_fsaverage_roi_masks[roi] = np.asarray(right_fsaverage_roi_classes[roi_class] == roi_mapping_temp, dtype=int)
        
        roi_mapping_temp = list(left_challenge_roi_maps[roi_class].keys())[list(left_challenge_roi_maps[roi_class].values()).index(roi)]
        left_challenge_roi_masks[roi] = np.asarray(left_challenge_roi_classes[roi_class] == roi_mapping_temp, dtype=int)
        
        roi_mapping_temp = list(right_challenge_roi_maps[roi_class].keys())[list(right_challenge_roi_maps[roi_class].values()).index(roi)]
        right_challenge_roi_masks[roi] = np.asarray(right_challenge_roi_classes[roi_class] == roi_mapping_temp, dtype=int)

    for roi in rois:
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_masks', hemispheres[0][0]+'h.'+roi+'_challenge_space.npy'), left_challenge_roi_masks[roi])
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_masks', hemispheres[1][0]+'h.'+roi+'_challenge_space.npy'), right_challenge_roi_masks[roi])
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_masks', hemispheres[0][0]+'h.'+roi+'_fsaverage_space.npy'), left_fsaverage_roi_masks[roi])
        np.save(os.path.join(args.roi_dir_enhanced, 'roi_masks', hemispheres[1][0]+'h.'+roi+'_fsaverage_space.npy'), right_fsaverage_roi_masks[roi])
    
    def unknown_mask_creator(dict):
        matrix = np.array(list(dict.values()))
        save_dict = {}
        # Crea un vettore con valori 1 quando tutti i valori della colonna (esclusi gli ultimi 6) sono uguali a 0
        save_dict['unknown_functional_ROI_masks'] = np.all(matrix[:-6] == 0, axis=0).astype(int)
        save_dict['unknown_streams_ROI_masks'] = np.all(matrix[-6:] == 0, axis=0).astype(int)
        save_dict['unknown_ROI_masks'] = np.all(matrix == 0, axis=0).astype(int)
        return save_dict
    
    left_fsaverage_unknown_ROI_masks = unknown_mask_creator(left_fsaverage_roi_masks)
    right_fsaverage_unknown_ROI_masks = unknown_mask_creator(right_fsaverage_roi_masks)
    left_challenge_unknown_ROI_masks = unknown_mask_creator(left_challenge_roi_masks)
    right_challenge_unknown_ROI_masks = unknown_mask_creator(right_challenge_roi_masks)
    
    # iterate over the dictionaries and save the masks
    for key in left_fsaverage_unknown_ROI_masks.keys():
        np.save(os.path.join(args.roi_dir_enhanced, 'unknown_masks', hemispheres[0][0]+'h.'+key+'_challenge_space.npy'), left_challenge_unknown_ROI_masks[key])
        np.save(os.path.join(args.roi_dir_enhanced, 'unknown_masks', hemispheres[1][0]+'h.'+key+'_challenge_space.npy'), right_challenge_unknown_ROI_masks[key])
        np.save(os.path.join(args.roi_dir_enhanced, 'unknown_masks', hemispheres[0][0]+'h.'+key+'_fsaverage_space.npy'), left_fsaverage_unknown_ROI_masks[key])
        np.save(os.path.join(args.roi_dir_enhanced, 'unknown_masks', hemispheres[1][0]+'h.'+key+'_fsaverage_space.npy'), right_fsaverage_unknown_ROI_masks[key])