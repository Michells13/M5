from net import ResNet_Triplet_COCO
from cocoTripletDataset import TripletCOCO
import os
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
import torch
from torchvision import transforms
import numpy as np

weights = "trained_Mask_backbone_5_epoch_1e-5_margin5.pth"

transformsPretrained = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
jsonPath = "./COCO/mcv_image_retrieval_annotations.json"
trainImagesPath = "./COCO/train2014/"
trainImages = os.listdir(trainImagesPath)
size = (240,320)
device = "cuda"

# Transform
transforms = torch.nn.Sequential(
    transformsPretrained,
    transforms.Resize(size),
)


train_dataset = TripletCOCO(trainImagesPath, trainImages, jsonPath, transforms)

model = ResNet_Triplet_COCO()
model = model.to(device)
model.load_state_dict(torch.load(weights, map_location=device))

with torch.no_grad():
    model.eval()
    for element in train_dataset:
        images = [e.unsqueeze(0).to(device) for e in element[0]]
        
        anchor = model(images[0]).cpu().numpy()
        positive = model(images[1]).cpu().numpy()
        negative = model(images[2]).cpu().numpy()
        
        posError = np.sum(np.square(anchor - positive))
        negError = np.sum(np.square(anchor - negative))
        

