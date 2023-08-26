import numpy as np
import math
import os 
import json
import pandas as pd

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

def process_csv_file(csv_path):
    data = pd.read_csv(csv_path, index_col=0)
    for old_label in data.index:
        if not pd.isna(old_label):
            network = old_label.split('+')[0]
            layers = old_label.split('+')[1].split('&')
            if len(layers) == 1:
                layers = layers[0]
            if pca_selector([network, layers]):
                print([network, layers])
                new_label = old_label + "+2048"
                data = data.rename(index={old_label: new_label})
            else:
                print([network, layers])
                new_label = old_label + "+9999999"
                data = data.rename(index={old_label: new_label})
        
    data.to_csv(csv_path)
    
def process_csv_file_concat(csv_path):
    data = pd.read_csv(csv_path, index_col=0)
    for old_label in data.index:
        if not pd.isna(old_label):
            new_label = old_label + "+concat_pca"
            data = data.rename(index={old_label: new_label})
        
    data.to_csv(csv_path)

# file_path = f"D:\\Projects\\Thesis\\files\\config_test\\global\\alpha_subj_layer.csv"
# process_csv_file_concat(file_path)
# file_path = f"D:\\Projects\\Thesis\\files\\config_test\\global\\scores_subj_layer_roi.csv"
# process_csv_file_concat(file_path)

for subj in range(1, 9):
    file_path = f"D:\\Projects\\Thesis\\files\\config_subj{subj}\\global\\alpha_subj_layer.csv"
    process_csv_file_concat(file_path)
    file_path = f"D:\\Projects\\Thesis\\files\\config_subj{subj}\\global\\scores_subj_layer_roi.csv"
    process_csv_file_concat(file_path)