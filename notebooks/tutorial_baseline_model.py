platform = 'jupyter_notebook'

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr


if platform == 'jupyter_notebook':
    data_dir = '../Datasets/Biomedical/algonauts_2023_challenge_data'
    parent_submission_dir = './algonauts_2023_challenge_submission_colab'
 


if __name__ == "__main__":
    device = 'cuda'
    device = torch.device(device)
    
    for subj in list(range(1, 9)): 
        class argObj:
            def __init__(self, data_dir, parent_submission_dir, subj):
                
                self.subj = format(subj, '02')
                self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
                self.parent_submission_dir = parent_submission_dir
                self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                    'subj'+self.subj)

                # Create the submission directory if not existing
                if not os.path.isdir(self.subject_submission_dir):
                    os.makedirs(self.subject_submission_dir)

        args = argObj(data_dir, parent_submission_dir, subj)
            
        fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

        print('LH training fMRI data shape:')
        print(lh_fmri.shape)
        print('(Training stimulus images × LH vertices)')

        print('\nRH training fMRI data shape:')
        print(rh_fmri.shape)
        print('(Training stimulus images × RH vertices)')   
        
        train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
        test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

        # Create lists will all training and test image file names, sorted
        train_img_list = os.listdir(train_img_dir)
        train_img_list.sort()
        test_img_list = os.listdir(test_img_dir)
        test_img_list.sort()
        print('Training images: ' + str(len(train_img_list)))
        print('Test images: ' + str(len(test_img_list)))    
        
        rand_seed = 5 #@param
        np.random.seed(rand_seed)

        # Calculate how many stimulus images correspond to 90% of the training data
        num_train = int(np.round(len(train_img_list) / 100 * 90))
        # Shuffle all training stimulus images
        idxs = np.arange(len(train_img_list))
        np.random.shuffle(idxs)
        # Assign 90% of the shuffled stimulus images to the training partition,
        # and 10% to the test partition
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
        # No need to shuffle or split the test stimulus images
        idxs_test = np.arange(len(test_img_list))

        print('Training stimulus images: ' + format(len(idxs_train)))
        print('\nValidation stimulus images: ' + format(len(idxs_val)))
        print('\nTest stimulus images: ' + format(len(idxs_test)))
        
        transform = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x24 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])
        
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
        if subj == 3 or subj == 6:
            batch_size = 310
        else: 
            batch_size = 300 # 300 #@param (310 for 3 and 6)            
        # batch_size = 300 #@param
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
        
        lh_fmri_train = lh_fmri[idxs_train]
        lh_fmri_val = lh_fmri[idxs_val]
        rh_fmri_train = rh_fmri[idxs_train]
        rh_fmri_val = rh_fmri[idxs_val]

        del lh_fmri, rh_fmri
        
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
        model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
        model.eval() # set the model to evaluation mode, since you are not training it
        
        train_nodes, _ = get_graph_node_names(model)
        print(train_nodes)

        model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
        feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

        def fit_pca(feature_extractor, dataloader):

            # Define PCA parameters
            pca = IncrementalPCA(n_components=100, batch_size=batch_size)

            # Fit PCA to batch
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                # Fit PCA to batch
                pca.partial_fit(ft.detach().cpu().numpy())
            return pca

        pca = fit_pca(feature_extractor, train_imgs_dataloader)

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

        features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
        features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
        features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)

        print('\nTraining images features:')
        print(features_train.shape)
        print('(Training stimulus images × PCA features)')

        print('\nValidation images features:')
        print(features_val.shape)
        print('(Validation stimulus images × PCA features)')

        print('\nTest images features:')
        print(features_val.shape)
        print('(Test stimulus images × PCA features)')

        del model, pca

        # Fit linear regressions on the training data
        reg_lh = LinearRegression().fit(features_train, lh_fmri_train)
        reg_rh = LinearRegression().fit(features_train, rh_fmri_train)
        # Use fitted linear regressions to predict the validation and test fMRI data
        lh_fmri_val_pred = reg_lh.predict(features_val)
        lh_fmri_test_pred = reg_lh.predict(features_test)
        rh_fmri_val_pred = reg_rh.predict(features_val)
        rh_fmri_test_pred = reg_rh.predict(features_test)

        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh_fmri_val_pred.shape[1])):
            lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh_fmri_val_pred.shape[1])):
            rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]
            
        lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
        rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)

        np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
        np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)