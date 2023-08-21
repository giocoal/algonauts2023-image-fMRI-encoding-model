from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models import AlexNet_Weights
from torchvision import transforms as pth_transforms
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

ViT_GPT2_pretrained_weights = "nlpconnect/vit-gpt2-image-captioning"
ViT_GPT2_feature_extractor = ViTImageProcessor.from_pretrained(ViT_GPT2_pretrained_weights)

def ViT_GPT2_transform(img):
    return ViT_GPT2_feature_extractor(images=img, return_tensors="pt").pixel_values[0]

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

dinov2_transform_V2 =  transforms.Compose([
                    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])

dino_resnet50_preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

# dino_resnet50_preprocess = pth_transforms.Compose([
#             pth_transforms.Resize(256, interpolation=3),
#             pth_transforms.CenterCrop(224),
#             pth_transforms.ToTensor(),
#             pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])

retinanet_transform = RetinaNet_ResNet50_FPN_Weights.COCO_V1.transforms