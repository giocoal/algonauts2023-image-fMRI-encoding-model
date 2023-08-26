import os
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch as torch
from torch.utils.data import DataLoader, Dataset

from src.visualize import json_config_to_feature_extraction_dict

import json

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
            img = self.transform(img)
        return img

class ImagefrmiDataset(Dataset):
    def __init__(self, imgs_paths, fmri_path, idxs, transform):
        # Acquisisco solo le immagini dello split specifico (train, val, test)
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        # Acquisisco solo i fmri dello split specifico (train, val, test)
        self.fmri = np.load(fmri_path)[idxs]
    def __len__(self):
        return len(self.imgs_paths)
    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img)
        # Load the fmri vector
        fmri = torch.Tensor(self.fmri[idx])
        return img, fmri

class argObj:
    """
    This class is used to define the paths to the data and the submission directories
    """
    def __init__(self, subj, data_home_dir, data_dir, parent_submission_dir, 
                 parent_ncsnr_dir, images_trials_parent_dir, save):
        # Define the dir where data is stored
        self.data_home_dir = data_home_dir
        
        # 1 became 01
        self.subj = format(subj, '02') # '0numberofchars'
        self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
        
        # fmri training data dir
        self.fmri_dir = os.path.join(self.data_dir, 'training_split', 'training_fmri')
        self.lh_fmri = os.path.join(self.fmri_dir, 'lh_training_fmri.npy')
        self.rh_fmri = os.path.join(self.fmri_dir, 'rh_training_fmri.npy')

        # stimuli data dir
        self.train_img_dir = os.path.join(self.data_dir, 'training_split', 'training_images')
        self.test_img_dir = os.path.join(self.data_dir, 'test_split', 'test_images')
        
        # NCSNR
        self.parent_ncsnr_dir = parent_ncsnr_dir
        self.ncsnr_dir = os.path.join(self.parent_ncsnr_dir, 'subj'+self.subj)
        
        # TEST PREDICTION SUBMISSION DIR
        self.parent_submission_dir = parent_submission_dir
        # Create the parent submission directory if not existing
        if not os.path.isdir(self.parent_submission_dir) and save:
            os.makedirs(self.parent_submission_dir)
        self.test_submission_dir = os.path.join(self.parent_submission_dir, "test_predictions")
        self.subject_test_submission_dir = os.path.join(self.test_submission_dir,
            'subj'+self.subj)
        # Create the submission directory if not existing
        if not os.path.isdir(self.subject_test_submission_dir) and save:
            os.makedirs(self.subject_test_submission_dir)
            
        # VAL PREDICTION SUBMISSION DIR
        self.val_submission_dir = os.path.join(self.parent_submission_dir, "val_predictions")
        self.subject_val_submission_dir = os.path.join(self.val_submission_dir,
            'subj'+self.subj)
        # Create the submission directory if not existing
        if not os.path.isdir(self.subject_val_submission_dir) and save:
            os.makedirs(self.subject_val_submission_dir)
            
        # VAL correlations SUBMISSION DIR
        self.val_correlation_dir = os.path.join(self.parent_submission_dir, "val_correlations")
        self.subject_val_correlation_dir = os.path.join(self.val_correlation_dir,
            'subj'+self.subj)
        # Create the correlation directory if not existing
        if not os.path.isdir(self.subject_val_correlation_dir) and save:
            os.makedirs(self.subject_val_correlation_dir)
          
        # VAL imgs submission dir
        self.val_images_submission_dir = os.path.join(self.parent_submission_dir, "val_imgs")
        # Create the parent submission directory if not existing
        if not os.path.isdir(self.val_images_submission_dir) and save:
            os.makedirs(self.val_images_submission_dir)
        self.subject_val_images_submission_dir = os.path.join(self.val_images_submission_dir,
            'subj'+self.subj)
        # Create the submission directory if not existing
        if not os.path.isdir(self.subject_val_images_submission_dir) and save:
            os.makedirs(self.subject_val_images_submission_dir)
            
        
            
        # ROY DIR
        self.roi_dir = os.path.join(self.data_dir, 'roi_masks')

        # Train Images Trials 
        self.images_trials_parent_dir = images_trials_parent_dir
        self.images_trials_dir = os.path.join(self.images_trials_parent_dir, 'subj'+self.subj)
        
        # fsaverage surfaces dir 
        self.fsaverage_surface_dir = os.path.join(self.data_home_dir, 'fsaverage_surface')
        
        # ROI advanced masks DIR
        
        self.roi_dir_enhanced = os.path.join(self.data_dir, 'roi_masks_enhanced')
        if not os.path.isdir(self.roi_dir_enhanced) and save:
            os.makedirs(self.roi_dir_enhanced)
            os.makedirs(os.path.join(self.roi_dir_enhanced, 'roi_classes_masks'))
            os.makedirs(os.path.join(self.roi_dir_enhanced, 'roi_masks'))
            os.makedirs(os.path.join(self.roi_dir_enhanced, 'unknown_masks'))
            
        self.roi_dir_enhanced_df = os.path.join(self.roi_dir_enhanced, 'roi_df')
        
    def images_idx_splitter(self, train_percentage):
        """
        This function splits the training and test images into training, validation and test partitions. 
        Given a percentage of the training images, it will assign that percentage to the training partition.
        """
        # Create lists will all training and test image file names, sorted
        train_img_list = os.listdir(self.train_img_dir)
        train_img_list.sort()
        test_img_list = os.listdir(self.test_img_dir)
        test_img_list.sort()
                
        ## Create the training, validation and test partitions indices ##
        # i set the random seed to 5 to have the same split for all the models
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
        
        print('## Stimulus Images Loading: Info')
        print('Total train images: ' + str(len(train_img_list)))
        print('Training stimulus images: ' + format(len(idxs_train)))
        print('Validation stimulus images: ' + format(len(idxs_val)))
        print('Test stimulus images: ' + format(len(idxs_test)))
        print("\n")
        
        # Get the paths of all image files
        train_imgs_paths = sorted(list(Path(self.train_img_dir).iterdir()))
        test_imgs_paths = sorted(list(Path(self.test_img_dir).iterdir()))
        
        return idxs_train, idxs_val, idxs_test, train_imgs_paths, test_imgs_paths
    

class data_loaders_stimuli_fmri:
  def __init__(self, 
               idxs_train, 
               idxs_val, 
               idxs_test, 
               train_imgs_paths, 
               test_imgs_paths,
               lh_fmri_path,
               rh_fmri_path):
    
    self.idxs_train = idxs_train
    self.idxs_val = idxs_val
    self.idxs_test = idxs_test
    self.train_imgs_paths = train_imgs_paths
    self.test_imgs_paths = test_imgs_paths
    self.lh_fmri = lh_fmri_path
    self.rh_fmri = rh_fmri_path
      
  def images_dataloader(self, batch_size, transform):
    """
    This function creates the dataloaders for the training, validation and test images.
    """
    train_imgs_dataloader = DataLoader(
        ImageDataset(self.train_imgs_paths, self.idxs_train, transform), 
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(self.train_imgs_paths, self.idxs_val, transform), 
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(self.test_imgs_paths, self.idxs_test, transform), 
        batch_size=batch_size
    )
    return train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader
  
  def fmri_splitter(self):
    ## Split the fmri data into training and validation partitions ##
    lh_fmri = np.load(self.lh_fmri)
    rh_fmri = np.load(self.rh_fmri)
    lh_fmri_train = lh_fmri[self.idxs_train]
    lh_fmri_val = lh_fmri[self.idxs_val]
    rh_fmri_train = rh_fmri[self.idxs_train]
    rh_fmri_val = rh_fmri[self.idxs_val]
    # del lh_fmri, rh_fmri
    return lh_fmri_train, lh_fmri_val, rh_fmri_train, rh_fmri_val

  def images_fmri_dataloader(self, batch_size, transform):
    train_dataloader_lh = DataLoader(
        ImagefrmiDataset(self.train_imgs_paths, self.lh_fmri, self.idxs_train, transform), 
        batch_size=batch_size
    )
    val_dataloader_lh = DataLoader(
        ImagefrmiDataset(self.train_imgs_paths, self.lh_fmri, self.idxs_val, transform), 
        batch_size=batch_size
    )
    test_dataloader_lh = DataLoader(
        ImagefrmiDataset(self.test_imgs_paths, self.lh_fmri, self.idxs_test, transform), 
        batch_size=batch_size
    )
    
    train_dataloader_rh = DataLoader(
        ImagefrmiDataset(self.train_imgs_paths, self.rh_fmri, self.idxs_train, transform), 
        batch_size=batch_size
    )
    val_dataloader_rh = DataLoader(
        ImagefrmiDataset(self.train_imgs_paths, self.rh_fmri, self.idxs_val, transform), 
        batch_size=batch_size
    )
    test_dataloader_rh = DataLoader(
        ImagefrmiDataset(self.test_imgs_paths, self.rh_fmri, self.idxs_test, transform), 
        batch_size=batch_size
    )
    
    return train_dataloader_lh, val_dataloader_lh, test_dataloader_lh, train_dataloader_rh, val_dataloader_rh, test_dataloader_rh

def transform_layers_test(layers):
    """
    Function used when creating the submission name
    """
    transformed_layers = []
    for layer in layers:
        # layer[0] to select only the layer and not the other params
        if isinstance(layer, list):
            transformed_layers.append('&'.join(layer))
        else:
            transformed_layers.append(layer)
    return transformed_layers

def transform_layers(layers):
    """
    Function used when creating the submission name
    """
    transformed_layers = []
    for layer in layers:
        # layer[0] to select only the layer and not the other params
        if isinstance(layer[0], list):
            transformed_layers.append('&'.join(layer[0]))
        else:
            transformed_layers.append(layer[0])
    return transformed_layers

# class masks_loader:
#     def __init__(self, 
#                 roi_masks_enhanced_path):

#         self.roi_masks_enhanced_path = roi_masks_enhanced_path
        
#     def load_roi_masks_challenge_from_list_of_ROI(self, final_extraction_config, model_layer_id, verbose=True):
#         """
#         This function loads the masks
#         """
#         roi_masks_enhanced_path_df =  os.path.join(self.roi_masks_enhanced_path, 'roi_df')
#         lh_roi_challenge_onehot = pd.read_csv(os.path.join(roi_masks_enhanced_path_df, 'lh_challenge_onehot.csv'))
#         rh_roi_challenge_onehot = pd.read_csv(os.path.join(roi_masks_enhanced_path_df, 'rh_challenge_onehot.csv'))
#         # i verify in config dict which rois are associated with the current layer
#         keys_with_target_value = [key for key, value in final_extraction_config.items() if value == model_layer_id]
#         print(f"ROIs which best-perform with the current layer: {keys_with_target_value}")
#         ### create the voxel mask for the current model-layer-specific set of ROIs
#         # subset of ROIs from the one-hot encoded dataframe
#         lh_roi_challenge_onehot_subset = lh_roi_challenge_onehot[keys_with_target_value]
#         rh_roi_challenge_onehot_subset = rh_roi_challenge_onehot[keys_with_target_value]
#         lh_mask = []
#         rh_mask = []
#         for index, row in lh_roi_challenge_onehot_subset.iterrows():
#             if row.sum() == 1:
#                 lh_mask.append(1)
#             elif row.sum() == 0:
#                 lh_mask.append(0)
#             else:
#                 raise ValueError(f"Errore: la riga {index} ha più di un 1.")
#         for index, row in rh_roi_challenge_onehot_subset.iterrows():
#             if row.sum() == 1:
#                 rh_mask.append(1)
#             elif row.sum() == 0:
#                 rh_mask.append(0)
#             else:
#                 raise ValueError(f"Errore: la riga {index} ha più di un 1.")
#         return np.array(lh_mask), np.array(rh_mask)
    
def load_roi_masks_challenge_from_list_of_ROI(roi_masks_enhanced_path, final_extraction_config, model_layer_id, verbose=True):
        """
        This function loads the masks
        """
        roi_masks_enhanced_path_df =  os.path.join(roi_masks_enhanced_path, 'roi_df')
        lh_roi_challenge_onehot = pd.read_csv(os.path.join(roi_masks_enhanced_path_df, 'lh_challenge_onehot.csv'))
        rh_roi_challenge_onehot = pd.read_csv(os.path.join(roi_masks_enhanced_path_df, 'rh_challenge_onehot.csv'))
        # i verify in config dict which rois are associated with the current layer
        # mantain only the model-layer params
        #final_extraction_config_reduced = {key: value[:2] if isinstance(value, list) else value for key, value in final_extraction_config.items()}
        keys_with_target_value = [key for key, value in final_extraction_config.items() if value == model_layer_id]
        print(f"ROIs which best-perform with the current layer: {keys_with_target_value}")
        ### create the voxel mask for the current model-layer-specific set of ROIs
        # subset of ROIs from the one-hot encoded dataframe
        lh_roi_challenge_onehot_subset = lh_roi_challenge_onehot[keys_with_target_value]
        rh_roi_challenge_onehot_subset = rh_roi_challenge_onehot[keys_with_target_value]
        lh_mask = []
        rh_mask = []
        for index, row in lh_roi_challenge_onehot_subset.iterrows():
            if row.sum() == 1:
                lh_mask.append(1)
            elif row.sum() == 0:
                lh_mask.append(0)
            else:
                raise ValueError(f"Errore: la riga {index} ha più di un 1.")
        for index, row in rh_roi_challenge_onehot_subset.iterrows():
            if row.sum() == 1:
                rh_mask.append(1)
            elif row.sum() == 0:
                rh_mask.append(0)
            else:
                raise ValueError(f"Errore: la riga {index} ha più di un 1.")
        return np.array(lh_mask), np.array(rh_mask)

def from_config_dict_to_submission_name(config_dict, datetime_id):
    """
    This function creates the submission name from the config dict
    """
    
    # Initialize the result string
    result_string = ""

    # Iterate through the dictionary and construct the desired string
    for key, value in config_dict.items():
        config_subj = value['config_subj']
        extraction_config_file = value['extraction_config_file']
        result_string += f"{key}_{config_subj}_{extraction_config_file}_"

    # Remove the trailing underscore
    result_string = result_string.rstrip('_')
    return datetime_id + result_string


def save_json_file(parent_submission_dir, config_dict):
    if os.path.exists(os.path.join(parent_submission_dir, 'config_used.json')):
            with open(os.path.join(parent_submission_dir, 'config_used.json'), 'w') as file:
                json.dump(config_dict, file, indent=4)
                
                
class FileNameGenerator:
    # write the basic structure of a class
    def __init__(self, feature_model_type, model_layer, transform_string, regression_type, pca_component, compute_pca, pca_mode):
        self.feature_model_type = feature_model_type
        self.model_layer = model_layer
        self.transform_string = transform_string
        self.regression_type = regression_type
        self.pca_component = pca_component
        self.compute_pca = compute_pca
        self.pca_mode = pca_mode
    def get_model_layer_id(self):
        if isinstance(self.model_layer, str):
            model_layer_id = f'{self.feature_model_type}+{self.model_layer}+{self.transform_string}+{self.regression_type}+{self.pca_component if self.compute_pca else 9999999}+{self.pca_mode}'
        else:
            model_layer_id = f'{self.feature_model_type}+{"&".join(self.model_layer)}+{self.transform_string}+{self.regression_type}+{self.pca_component if self.compute_pca else 9999999}+{self.pca_mode}'
        return model_layer_id
    def get_features_file_name(self, subj):
        if isinstance(self.model_layer, str):
            features_file_name = f'{subj}+{self.feature_model_type}+{self.model_layer}+{self.transform_string}+{self.pca_component if self.compute_pca else 9999999}'
        else:
            features_file_name = f'{subj}+{self.feature_model_type}+{"&".join(self.model_layer)}+{self.transform_string}+{self.pca_component if self.compute_pca else 9999999}'
        return features_file_name + "_train", features_file_name + "_val", features_file_name + "_test"
    def get_pca_file_name(self):
        if isinstance(self.model_layer, str):
            pca_file_name = f'{self.feature_model_type}+{self.model_layer}+{self.transform_string}+{self.pca_component if self.compute_pca else 9999999}'
        else:
            pca_file_name = f'{self.feature_model_type}+{"&".join(self.model_layer)}+{self.transform_string}+{self.pca_component if self.compute_pca else 9999999}'
        return pca_file_name