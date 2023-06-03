import os
from pathlib import Path

import numpy as np

from PIL import Image

from torch.utils.data import DataLoader, Dataset

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
  
  