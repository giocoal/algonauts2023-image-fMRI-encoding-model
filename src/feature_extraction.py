import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models import AlexNet_Weights, VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights, EfficientNet_B2_Weights, EfficientNet_B5_Weights, ResNet50_Weights
from efficientnet_pytorch import EfficientNet
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from pytorchcv.model_provider import get_model as ptcv_get_model

from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import numpy as np

def feature_extractor_gen(model, model_layer, device):
    """Creates a feature extractor from a model.
    Args:
        model: The model from which to extract features.
        return_nodes: The names of the nodes from which to extract features.
    Returns:
        A feature extractor that takes an input tensor and returns a list of
        tensors corresponding to the outputs of the nodes in `return_nodes`.
    """
    model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
    model.eval() # set the model to evaluation mode, since you are not training it
    for param in model.parameters():
        param.requires_grad = False
    # If i hstack multiple layer model_layer will be a list
    if isinstance(model_layer, list):
        feature_extractor = create_feature_extractor(model, return_nodes=model_layer)
    elif isinstance(model_layer, str):
        feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
    else:
        raise ValueError('model_layer must be a list or a string')
    return model, feature_extractor 

def model_loader(feature_model_type, model_layer, device):
    print('## Loading feature extraction model...')
    if feature_model_type == 'alexnet':
        ## AlexNet ##
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                            'alexnet', 
                            weights=AlexNet_Weights.IMAGENET1K_V1)
        #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type == 'ZFNet':
        model = ptcv_get_model("ZFNet", pretrained=True)
        # ['x', 'features.stage1.unit1.conv', 'features.stage1.unit1.activ', 'features.stage1.unit1.local_response_norm', 'features.stage1.pool1', 'features.stage2.unit1.conv', 'features.stage2.unit1.activ', 'features.stage2.unit1.local_response_norm', 'features.stage2.pool2', 'features.stage3.unit1.conv', 'features.stage3.unit1.activ', 'features.stage3.unit2.conv', 'features.stage3.unit2.activ', 'features.stage3.unit3.conv', 'features.stage3.unit3.activ', 'features.stage3.pool3', 'size', 'view', 'output.fc1.fc', 'output.fc1.activ', 'output.fc1.dropout', 'output.fc2.fc', 'output.fc2.activ', 'output.fc2.dropout', 'output.fc3']
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type == 'vgg16':
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                            'vgg16_bn', 
                            weights=VGG16_BN_Weights.IMAGENET1K_V1)
        # ['x', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15', 'features.16', 'features.17', 'features.18', 'features.19', 'features.20', 'features.21', 'features.22', 'features.23', 'features.24', 'features.25', 'features.26', 'features.27', 'features.28', 'features.29', 'features.30', 'features.31', 'features.32', 'features.33', 'features.34', 'features.35', 'features.36', 'features.37', 'features.38', 'features.39', 'features.40', 'features.41', 'features.42', 'features.43', 'avgpool', 'flatten', 'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4', 'classifier.5', 'classifier.6']
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type == 'efficientnetb2':
        model = models.efficientnet_b2(weights = EfficientNet_B2_Weights.IMAGENET1K_V1)
        #@param ['x', 'features.0', 'features.1.0.block.0', 'features.1.0.block.1', 'features.1.0.block.2', 'features.1.1.block.0', 'features.1.1.block.1', 'features.1.1.block.2', 'features.1.1.stochastic_depth', 'features.1.1.add', 'features.2.0.block.0', 'features.2.0.block.1', 'features.2.0.block.2', 'features.2.0.block.3', 'features.2.1.block.0', 'features.2.1.block.1', 'features.2.1.block.2', 'features.2.1.block.3', 'features.2.1.stochastic_depth', 'features.2.1.add', 'features.2.2.block.0', 'features.2.2.block.1', 'features.2.2.block.2', 'features.2.2.block.3', 'features.2.2.stochastic_depth', 'features.2.2.add', 'features.3.0.block.0', 'features.3.0.block.1', 'features.3.0.block.2', 'features.3.0.block.3', 'features.3.1.block.0', 'features.3.1.block.1', 'features.3.1.block.2', 'features.3.1.block.3', 'features.3.1.stochastic_depth', 'features.3.1.add', 'features.3.2.block.0', 'features.3.2.block.1', 'features.3.2.block.2', 'features.3.2.block.3', 'features.3.2.stochastic_depth', 'features.3.2.add', 'features.4.0.block.0', 'features.4.0.block.1', 'features.4.0.block.2', 'features.4.0.block.3', 'features.4.1.block.0', 'features.4.1.block.1', 'features.4.1.block.2', 'features.4.1.block.3', 'features.4.1.stochastic_depth', 'features.4.1.add', 'features.4.2.block.0', 'features.4.2.block.1', 'features.4.2.block.2', 'features.4.2.block.3', 'features.4.2.stochastic_depth', 'features.4.2.add', 'features.4.3.block.0', 'features.4.3.block.1', 'features.4.3.block.2', 'features.4.3.block.3', 'features.4.3.stochastic_depth', 'features.4.3.add', 'features.5.0.block.0', 'features.5.0.block.1', 'features.5.0.block.2', 'features.5.0.block.3', 'features.5.1.block.0', 'features.5.1.block.1', 'features.5.1.block.2', 'features.5.1.block.3', 'features.5.1.stochastic_depth', 'features.5.1.add', 'features.5.2.block.0', 'features.5.2.block.1', 'features.5.2.block.2', 'features.5.2.block.3', 'features.5.2.stochastic_depth', 'features.5.2.add', 'features.5.3.block.0', 'features.5.3.block.1', 'features.5.3.block.2', 'features.5.3.block.3', 'features.5.3.stochastic_depth', 'features.5.3.add', 'features.6.0.block.0', 'features.6.0.block.1', 'features.6.0.block.2', 'features.6.0.block.3', 'features.6.1.block.0', 'features.6.1.block.1', 'features.6.1.block.2', 'features.6.1.block.3', 'features.6.1.stochastic_depth', 'features.6.1.add', 'features.6.2.block.0', 'features.6.2.block.1', 'features.6.2.block.2', 'features.6.2.block.3', 'features.6.2.stochastic_depth', 'features.6.2.add', 'features.6.3.block.0', 'features.6.3.block.1', 'features.6.3.block.2', 'features.6.3.block.3', 'features.6.3.stochastic_depth', 'features.6.3.add', 'features.6.4.block.0', 'features.6.4.block.1', 'features.6.4.block.2', 'features.6.4.block.3', 'features.6.4.stochastic_depth', 'features.6.4.add', 'features.7.0.block.0', 'features.7.0.block.1', 'features.7.0.block.2', 'features.7.0.block.3', 'features.7.1.block.0', 'features.7.1.block.1', 'features.7.1.block.2', 'features.7.1.block.3', 'features.7.1.stochastic_depth', 'features.7.1.add', 'features.8', 'avgpool', 'flatten', 'classifier.0', 'classifier.1']
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type == 'efficientnet_b5':
        model = models.efficientnet_b5(weights = EfficientNet_B5_Weights.IMAGENET1K_V1)
        # ['x', 'features.0', 'features.1.0.block.0', 'features.1.0.block.1', 'features.1.0.block.2', 'features.1.1.block.0', 'features.1.1.block.1', 'features.1.1.block.2', 'features.1.1.stochastic_depth', 'features.1.1.add', 'features.1.2.block.0', 'features.1.2.block.1', 'features.1.2.block.2', 'features.1.2.stochastic_depth', 'features.1.2.add', 'features.2.0.block.0', 'features.2.0.block.1', 'features.2.0.block.2', 'features.2.0.block.3', 'features.2.1.block.0', 'features.2.1.block.1', 'features.2.1.block.2', 'features.2.1.block.3', 'features.2.1.stochastic_depth', 'features.2.1.add', 'features.2.2.block.0', 'features.2.2.block.1', 'features.2.2.block.2', 'features.2.2.block.3', 'features.2.2.stochastic_depth', 'features.2.2.add', 'features.2.3.block.0', 'features.2.3.block.1', 'features.2.3.block.2', 'features.2.3.block.3', 'features.2.3.stochastic_depth', 'features.2.3.add', 'features.2.4.block.0', 'features.2.4.block.1', 'features.2.4.block.2', 'features.2.4.block.3', 'features.2.4.stochastic_depth', 'features.2.4.add', 'features.3.0.block.0', 'features.3.0.block.1', 'features.3.0.block.2', 'features.3.0.block.3', 'features.3.1.block.0', 'features.3.1.block.1', 'features.3.1.block.2', 'features.3.1.block.3', 'features.3.1.stochastic_depth', 'features.3.1.add', 'features.3.2.block.0', 'features.3.2.block.1', 'features.3.2.block.2', 'features.3.2.block.3', 'features.3.2.stochastic_depth', 'features.3.2.add', 'features.3.3.block.0', 'features.3.3.block.1', 'features.3.3.block.2', 'features.3.3.block.3', 'features.3.3.stochastic_depth', 'features.3.3.add', 'features.3.4.block.0', 'features.3.4.block.1', 'features.3.4.block.2', 'features.3.4.block.3', 'features.3.4.stochastic_depth', 'features.3.4.add', 'features.4.0.block.0', 'features.4.0.block.1', 'features.4.0.block.2', 'features.4.0.block.3', 'features.4.1.block.0', 'features.4.1.block.1', 'features.4.1.block.2', 'features.4.1.block.3', 'features.4.1.stochastic_depth', 'features.4.1.add', 'features.4.2.block.0', 'features.4.2.block.1', 'features.4.2.block.2', 'features.4.2.block.3', 'features.4.2.stochastic_depth', 'features.4.2.add', 'features.4.3.block.0', 'features.4.3.block.1', 'features.4.3.block.2', 'features.4.3.block.3', 'features.4.3.stochastic_depth', 'features.4.3.add', 'features.4.4.block.0', 'features.4.4.block.1', 'features.4.4.block.2', 'features.4.4.block.3', 'features.4.4.stochastic_depth', 'features.4.4.add', 'features.4.5.block.0', 'features.4.5.block.1', 'features.4.5.block.2', 'features.4.5.block.3', 'features.4.5.stochastic_depth', 'features.4.5.add', 'features.4.6.block.0', 'features.4.6.block.1', 'features.4.6.block.2', 'features.4.6.block.3', 'features.4.6.stochastic_depth', 'features.4.6.add', 'features.5.0.block.0', 'features.5.0.block.1', 'features.5.0.block.2', 'features.5.0.block.3', 'features.5.1.block.0', 'features.5.1.block.1', 'features.5.1.block.2', 'features.5.1.block.3', 'features.5.1.stochastic_depth', 'features.5.1.add', 'features.5.2.block.0', 'features.5.2.block.1', 'features.5.2.block.2', 'features.5.2.block.3', 'features.5.2.stochastic_depth', 'features.5.2.add', 'features.5.3.block.0', 'features.5.3.block.1', 'features.5.3.block.2', 'features.5.3.block.3', 'features.5.3.stochastic_depth', 'features.5.3.add', 'features.5.4.block.0', 'features.5.4.block.1', 'features.5.4.block.2', 'features.5.4.block.3', 'features.5.4.stochastic_depth', 'features.5.4.add', 'features.5.5.block.0', 'features.5.5.block.1', 'features.5.5.block.2', 'features.5.5.block.3', 'features.5.5.stochastic_depth', 'features.5.5.add', 'features.5.6.block.0', 'features.5.6.block.1', 'features.5.6.block.2', 'features.5.6.block.3', 'features.5.6.stochastic_depth', 'features.5.6.add', 'features.6.0.block.0', 'features.6.0.block.1', 'features.6.0.block.2', 'features.6.0.block.3', 'features.6.1.block.0', 'features.6.1.block.1', 'features.6.1.block.2', 'features.6.1.block.3', 'features.6.1.stochastic_depth', 'features.6.1.add', 'features.6.2.block.0', 'features.6.2.block.1', 'features.6.2.block.2', 'features.6.2.block.3', 'features.6.2.stochastic_depth', 'features.6.2.add', 'features.6.3.block.0', 'features.6.3.block.1', 'features.6.3.block.2', 'features.6.3.block.3', 'features.6.3.stochastic_depth', 'features.6.3.add', 'features.6.4.block.0', 'features.6.4.block.1', 'features.6.4.block.2', 'features.6.4.block.3', 'features.6.4.stochastic_depth', 'features.6.4.add', 'features.6.5.block.0', 'features.6.5.block.1', 'features.6.5.block.2', 'features.6.5.block.3', 'features.6.5.stochastic_depth', 'features.6.5.add', 'features.6.6.block.0', 'features.6.6.block.1', 'features.6.6.block.2', 'features.6.6.block.3', 'features.6.6.stochastic_depth', 'features.6.6.add', 'features.6.7.block.0', 'features.6.7.block.1', 'features.6.7.block.2', 'features.6.7.block.3', 'features.6.7.stochastic_depth', 'features.6.7.add', 'features.6.8.block.0', 'features.6.8.block.1', 'features.6.8.block.2', 'features.6.8.block.3', 'features.6.8.stochastic_depth', 'features.6.8.add', 'features.7.0.block.0', 'features.7.0.block.1', 'features.7.0.block.2', 'features.7.0.block.3', 'features.7.1.block.0', 'features.7.1.block.1', 'features.7.1.block.2', 'features.7.1.block.3', 'features.7.1.stochastic_depth', 'features.7.1.add', 'features.7.2.block.0', 'features.7.2.block.1', 'features.7.2.block.2', 'features.7.2.block.3', 'features.7.2.stochastic_depth', 'features.7.2.add', 'features.8', 'avgpool', 'flatten', 'classifier.0', 'classifier.1']
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type == 'vgg19':
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                        'vgg19', 
                        weights=VGG19_Weights.IMAGENET1K_V1)
        # ['x', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15', 'features.16', 'features.17', 'features.18', 'features.19', 'features.20', 'features.21', 'features.22', 'features.23', 'features.24', 'features.25', 'features.26', 'features.27', 'features.28', 'features.29', 'features.30', 'features.31', 'features.32', 'features.33', 'features.34', 'features.35', 'features.36', 'features.37', 'features.38', 'features.39', 'features.40', 'features.41', 'features.42', 'features.43', 'features.44', 'features.45', 'features.46', 'features.47', 'features.48', 'features.49', 'features.50', 'features.51', 'features.52', 'avgpool', 'flatten', 'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4', 'classifier.5', 'classifier.6']
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type == 'vgg19_bn':
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                        'vgg19_bn', 
                        weights=VGG19_BN_Weights.IMAGENET1K_V1)
        # ['x', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15', 'features.16', 'features.17', 'features.18', 'features.19', 'features.20', 'features.21', 'features.22', 'features.23', 'features.24', 'features.25', 'features.26', 'features.27', 'features.28', 'features.29', 'features.30', 'features.31', 'features.32', 'features.33', 'features.34', 'features.35', 'features.36', 'features.37', 'features.38', 'features.39', 'features.40', 'features.41', 'features.42', 'features.43', 'features.44', 'features.45', 'features.46', 'features.47', 'features.48', 'features.49', 'features.50', 'features.51', 'features.52', 'avgpool', 'flatten', 'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4', 'classifier.5', 'classifier.6']
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type.rstrip("0123456789") == 'resnet':
        # resnet18, resnet34, resnet50, resnet101, resnet152
        if feature_model_type == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                        feature_model_type, 
                        weights=weights)
        # ['x', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu_1', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.downsample.0', 'layer1.0.downsample.1', 'layer1.0.add', 'layer1.0.relu_2', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu_1', 'layer1.1.conv3', 'layer1.1.bn3', 'layer1.1.add', 'layer1.1.relu_2', 'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.relu', 'layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.relu_1', 'layer1.2.conv3', 'layer1.2.bn3', 'layer1.2.add', 'layer1.2.relu_2', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.relu_1', 'layer2.0.conv3', 'layer2.0.bn3', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.0.add', 'layer2.0.relu_2', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.relu_1', 'layer2.1.conv3', 'layer2.1.bn3', 'layer2.1.add', 'layer2.1.relu_2', 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.relu', 'layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.relu_1', 'layer2.2.conv3', 'layer2.2.bn3', 'layer2.2.add', 'layer2.2.relu_2', 'layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.relu', 'layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.relu_1', 'layer2.3.conv3', 'layer2.3.bn3', 'layer2.3.add', 'layer2.3.relu_2', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.relu_1', 'layer3.0.conv3', 'layer3.0.bn3', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.0.add', 'layer3.0.relu_2', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.relu_1', 'layer3.1.conv3', 'layer3.1.bn3', 'layer3.1.add', 'layer3.1.relu_2', 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.relu', 'layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.relu_1', 'layer3.2.conv3', 'layer3.2.bn3', 'layer3.2.add', 'layer3.2.relu_2', 'layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.relu', 'layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.relu_1', 'layer3.3.conv3', 'layer3.3.bn3', 'layer3.3.add', 'layer3.3.relu_2', 'layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.relu', 'layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.relu_1', 'layer3.4.conv3', 'layer3.4.bn3', 'layer3.4.add', 'layer3.4.relu_2', 'layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.relu', 'layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.relu_1', 'layer3.5.conv3', 'layer3.5.bn3', 'layer3.5.add', 'layer3.5.relu_2', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.relu_1', 'layer4.0.conv3', 'layer4.0.bn3', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.0.add', 'layer4.0.relu_2', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.relu_1', 'layer4.1.conv3', 'layer4.1.bn3', 'layer4.1.add', 'layer4.1.relu_2', 'layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.relu', 'layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.relu_1', 'layer4.2.conv3', 'layer4.2.bn3', 'layer4.2.add', 'layer4.2.relu_2', 'avgpool', 'flatten', 'fc']
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    elif feature_model_type == 'resnet-50-maxpool':
        # resnet18, resnet34, resnet50, resnet101, resnet152
        if feature_model_type == 'resnet-50-maxpool':
            weights = ResNet50_Weights.IMAGENET1K_V2
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                        'resnet50', 
                        weights=weights)
        # ['x', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu_1', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.downsample.0', 'layer1.0.downsample.1', 'layer1.0.add', 'layer1.0.relu_2', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu_1', 'layer1.1.conv3', 'layer1.1.bn3', 'layer1.1.add', 'layer1.1.relu_2', 'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.relu', 'layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.relu_1', 'layer1.2.conv3', 'layer1.2.bn3', 'layer1.2.add', 'layer1.2.relu_2', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.relu_1', 'layer2.0.conv3', 'layer2.0.bn3', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.0.add', 'layer2.0.relu_2', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.relu_1', 'layer2.1.conv3', 'layer2.1.bn3', 'layer2.1.add', 'layer2.1.relu_2', 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.relu', 'layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.relu_1', 'layer2.2.conv3', 'layer2.2.bn3', 'layer2.2.add', 'layer2.2.relu_2', 'layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.relu', 'layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.relu_1', 'layer2.3.conv3', 'layer2.3.bn3', 'layer2.3.add', 'layer2.3.relu_2', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.relu_1', 'layer3.0.conv3', 'layer3.0.bn3', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.0.add', 'layer3.0.relu_2', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.relu_1', 'layer3.1.conv3', 'layer3.1.bn3', 'layer3.1.add', 'layer3.1.relu_2', 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.relu', 'layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.relu_1', 'layer3.2.conv3', 'layer3.2.bn3', 'layer3.2.add', 'layer3.2.relu_2', 'layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.relu', 'layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.relu_1', 'layer3.3.conv3', 'layer3.3.bn3', 'layer3.3.add', 'layer3.3.relu_2', 'layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.relu', 'layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.relu_1', 'layer3.4.conv3', 'layer3.4.bn3', 'layer3.4.add', 'layer3.4.relu_2', 'layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.relu', 'layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.relu_1', 'layer3.5.conv3', 'layer3.5.bn3', 'layer3.5.add', 'layer3.5.relu_2', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.relu_1', 'layer4.0.conv3', 'layer4.0.bn3', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.0.add', 'layer4.0.relu_2', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.relu_1', 'layer4.1.conv3', 'layer4.1.bn3', 'layer4.1.add', 'layer4.1.relu_2', 'layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.relu', 'layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.relu_1', 'layer4.2.conv3', 'layer4.2.bn3', 'layer4.2.add', 'layer4.2.relu_2', 'avgpool', 'flatten', 'fc']
        model, feature_extractor = feature_extractor_gen(model, "avgpool", device)
        feature_extractor.avgpool = nn.AdaptiveMaxPool2d(3)
    elif feature_model_type == 'DINOv2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
        model.eval() # set the model to evaluation mode, since you are not training it
        for param in model.parameters():
            param.requires_grad = False
        feature_extractor = model
    elif feature_model_type == 'RetinaNet':
        model = models.detection.retinanet_resnet50_fpn(weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1).backbone
        model, feature_extractor = feature_extractor_gen(model, model_layer, device)
    else: 
        print("Warning: Model not found.")
        model = None 
        feature_extractor = None
    print("\n")
    print(f'Feature extractor: {feature_model_type}, layer: {str(model_layer)}')
    return model, feature_extractor     

def pca_batch_calculator(total_instances, batch_size, min_pca_batch_size, pca_component):
    print("\n")
    print(f'## Calculating PCA batch size...')
    n_stacked_batches = 1
    while True:
        pca_batch_size = batch_size * n_stacked_batches
        if total_instances % pca_batch_size >= min_pca_batch_size:
            break
        n_stacked_batches += 1
    pca_batch_size = batch_size * n_stacked_batches

    print(f'Batches size: {batch_size}')
    print(f'Total train instances: {total_instances}')
    print(f'PCA components: {pca_component}')
    print(f'Minimum pca batch size: {min_pca_batch_size}')
    print(f'Number of stacked batches for pca: {n_stacked_batches}')
    print(f'PCA batch size (batch_size * n_stacked_batches): {pca_batch_size}')
    print(f'Last pca batch size: {total_instances % pca_batch_size}')
    return pca_batch_size, n_stacked_batches
    
def fit_pca(feature_extractor, dataloader, pca_component, n_stacked_batches, pca_batch_size, device):
    print(f'## Fitting Incremental PCA ({pca_component} components) to training data...')
    # Initialize empty ft_stacked matrix which will be used to stack the features 
    # of n_stacked_batches batches
    ft_stacked = None
    # Initialize counter for stacked batches
    count_stacked_batches = 0
    # Define PCA parameters
    pca = IncrementalPCA(n_components=pca_component, batch_size=pca_batch_size)

    with torch.no_grad(): 
        # Fit PCA to batch
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Bove batch to gpu and extract features
            ft = feature_extractor(d.to(device))
            # Flatten the features and detach them from the graph
            if isinstance(ft, dict):
                # In the case of FPN (Retinanet), extract only the features from the last layer
                if list(ft.keys())[0] == 'fpn':
                    ft = ft['fpn']
                # if feature_extractor output is a dictionary, flatten each layer and stack them horizontally
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).detach().cpu().numpy()
            else:
                # else flatten the output and detach it from the graph
                ft = torch.flatten(ft, start_dim=1).detach().cpu().numpy()
            # Fit PCA to batch
            # print(ft.shape)
            # Stack vertically the ft matrix
            if ft_stacked is None:
                # If the feature batch is the first one, initialize ft_stacked
                ft_stacked = ft #.astype(np.float32)
            else:
                # if the feature batch is not the first one, stack vertically the ft matrix
                ft_stacked = np.vstack((ft_stacked, ft)) #.astype(np.float32)
            # convert to float32 to save memory
            # ft_stacked = ft_stacked.astype(np.float32)
            # Check if n_stacked_batches is reached
            count_stacked_batches += 1
            if count_stacked_batches == n_stacked_batches:
                # print(ft_stacked.shape)
                # If n_stacked_batches is reached, fit PCA to ft_stacked
                pca.partial_fit(ft_stacked) #.astype(np.float32)
                # After fitting PCA, reset ft_stacked and counter
                ft_stacked = None
                ft = None
                count_stacked_batches = 0
            # Free VRAM memory by deleting model graph
            # torch.cuda.empty_cache()
        # Check if there are remaining stacked batches to process
        # that will be the case if the total number of batches is not a multiple of n_stacked_batches
        # the last pca batch will be smaller than pca_batch_size
        if count_stacked_batches > 0 and ft_stacked is not None:
            # Fit PCA to ft_stacked
            pca.partial_fit(ft_stacked)
            ft_stacked = None
            ft = None
            # torch.cuda.empty_cache()
            
        # Free VRAM memory by deleting model graph
        # d.to('cpu')
        del d, feature_extractor, ft, ft_stacked
        torch.cuda.empty_cache()
        return pca

def extract_and_pca_features(feature_extractor, dataloader, pca, n_stacked_batches, device):
    # Initialize empty ft_stacked matrix which will be used to stack the features 
    # of n_stacked_batches batches
    ft_stacked = None
    # Initialize counter for stacked batches
    count_stacked_batches = 0
    # Define list to store downsampled features
    features = []

    # Set model to evaluation mode (e.g., disable dropout)
    with torch.no_grad(): 
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Bove batch to gpu and extract features
            ft = feature_extractor(d.to(device))
            # Flatten the features and detach them from the graph
            if isinstance(ft, dict):
                if list(ft.keys())[0] == 'fpn':
                    ft = ft['fpn']
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).detach().cpu().numpy()
            else:
                ft = torch.flatten(ft, start_dim=1).detach().cpu().numpy()
            # Fit PCA to batch
            # print(ft.shape)
            # Stack vertically the ft matrix
            if ft_stacked is None:
                # If the feature batch is the first one, initialize ft_stacked
                ft_stacked = ft
            else:
                # if the feature batch is not the first one, stack vertically the ft matrix
                ft_stacked = np.vstack((ft_stacked, ft))
            # Check if n_stacked_batches is reached
            count_stacked_batches += 1
            if count_stacked_batches == n_stacked_batches:
                # print(ft_stacked.shape)
                # If n_stacked_batches is reached, transform ft_stacked with PCA
                ft_stacked = pca.transform(ft_stacked)
                # Append downsampled features to features list
                features.append(ft_stacked)
                # After transforming with PCA, reset ft_stacked and counter
                ft_stacked = None
                count_stacked_batches = 0
            # Free VRAM memory by deleting model graph
            # torch.cuda.empty_cache()
        # Check if there are remaining stacked batches to process
        # that will be the case if the total number of batches is not a multiple of n_stacked_batches
        # the last pca batch will be smaller than pca_batch_size
        if count_stacked_batches > 0 and ft_stacked is not None:
            # Fit PCA to ft_stacked
            ft_stacked = pca.transform(ft_stacked)
            features.append(ft_stacked)
            ft_stacked = None
            # torch.cuda.empty_cache()
        print(f"Inital features number: {ft.shape[1]}, final features number: {features[0].shape[1]}")
        del d, feature_extractor, ft, ft_stacked
        torch.cuda.empty_cache()
        return np.vstack(features)
    
def extract_features_no_pca(feature_extractor, dataloader, device):
    # Initialize empty ft_stacked matrix which will be used to stack the features 
    # of n_stacked_batches batches
    ft_stacked = None
    # Set model to evaluation mode (e.g., disable dropout)
    with torch.no_grad(): 
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Bove batch to gpu and extract features
            ft = feature_extractor(d.to(device))
            # Flatten the features and detach them from the graph
            if isinstance(ft, dict):
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()]).detach().cpu().numpy()
            else:
                ft = torch.flatten(ft, start_dim=1).detach().cpu().numpy()
            # Fit PCA to batch
            # print(ft.shape)
            # Stack vertically the ft matrix
            if ft_stacked is None:
                # If the feature batch is the first one, initialize ft_stacked
                ft_stacked = ft
            else:
                # if the feature batch is not the first one, stack vertically the ft matrix
                ft_stacked = np.vstack((ft_stacked, ft))
        
        print(f"Features number: {ft_stacked.shape[1]}")
        del d, feature_extractor, ft
        torch.cuda.empty_cache()
        return ft_stacked