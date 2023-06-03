from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models import AlexNet_Weights

imagenet_V1_transform = transforms.Compose([
    transforms.Resize((256,256)), # resize the images to 224x224 pixels (256x256)
    transforms.CenterCrop(224), # IN THE OLDER SCRIPT: no center crop, only resize to 224x224
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    # normalize the images color channels
    # mean: [0.485, 0.456, 0.406] for the three channels
    # std: [0.229, 0.224, 0.225] for the three channels
])

imagenet_V2_transform = transforms.Compose([
    transforms.Resize((224,224)), # resize the images to 224x224 pixels (256x256)
    # transforms.CenterCrop(224), # IN THE OLDER SCRIPT: no center crop, only resize to 224x224
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    # normalize the images color channels
    # mean: [0.485, 0.456, 0.406] for the three channels
    # std: [0.229, 0.224, 0.225] for the three channels
])

imagenet_transform_alt = transforms.Compose([
    transforms.Resize((224,224)), # resize the images to 224x224 pixels (256x256)
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    # normalize the images color channels
    # mean: [0.485, 0.456, 0.406] for the three channels
    # std: [0.229, 0.224, 0.225] for the three channels
])

efficientnetb2_transform = transforms.Compose([
    transforms.Resize((288,288), interpolation=InterpolationMode.BICUBIC), # resize the images to 224x224 pixels (256x256)
    transforms.CenterCrop(288),
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    # normalize the images color channels
    # mean: [0.485, 0.456, 0.406] for the three channels
    # std: [0.229, 0.224, 0.225] for the three channels
])

dinov2_transform = transforms.Compose([
    transforms.Resize((350,350), interpolation=InterpolationMode.BICUBIC), # resize the images to 224x224 pixels (256x256)
    transforms.CenterCrop(350),
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    # normalize the images color channels
    # mean: [0.485, 0.456, 0.406] for the three channels
    # std: [0.229, 0.224, 0.225] for the three channels
])

retinanet_transform = RetinaNet_ResNet50_FPN_Weights.COCO_V1.transforms