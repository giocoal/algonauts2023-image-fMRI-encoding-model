import numpy as np
import math
import os 
import json

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

def process_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if not isinstance(value, list) or (isinstance(value, float) and math.isnan(value)):
                continue
            elif pca_selector(value[:2]):
                print(value[:2])
                value.append("2048")
            else:
                print(value[:2])
                value.append("9999999")
        
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
for subj in range(1, 9):
    folder_path = f"D:\\Projects\\Thesis\\files\\config_subj{subj}\\global"
    #folder_path = f"D:\\Projects\\Thesis\\files\\config_test\\global"
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            process_json_file(file_path)