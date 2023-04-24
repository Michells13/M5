from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from cocoTripletDataset import TripletCOCO_Img2Text
import os
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2
from net import FasterRCNN_Triplet_COCO
from torchvision import transforms



def cache_outputs(loader, pre_model, img_model, cache_filename, device, max_rows):
    """Create a database for the retrieval.
    
    Write model's outputs for the loader's data into a file located at
    cache_filename.
    Args:
        loader: PyTorc DataLoader representing your dataset
        model: callable, used to produce feature vectors,
            takes (n_samples, in_features), 
            outputs (n_samles, out_features)
        cache_filename: path to a file, into which feature vectors
            will be written
    Returns:
        None
    """
    # pre_model.eval()
    img_model.eval()
    with torch.no_grad():
        f = open(cache_filename, "wb")
        rows = 0
        for data, _, _ in tqdm(loader):
            print("\n")
            print(rows)
            if rows >= max_rows:
                break
            data = data.to(device)
            output = pre_model(data)
            output = img_model(output.squeeze())
            np.savetxt(f, output.cpu().numpy())
            rows += len(output)
        f.close()


cache_filename = r"img_final.txt"

val_jsonPath = r"D:\coco\annotations_trainval2014\annotations\person_keypoints_val2014.json"
val_jsonPathCap= r"D:\coco\annotations_trainval2014\annotations\captions_val2014.json"
val_trainImagesPath = r"D:\coco\val2014\val2014"
IMG_MODEL_PATH = r"D:\img2text\last_weights_img.pth"

device = "cuda"
batch_size = 32
n_epochs = 4
log_interval = 5
size = (240,320)

# Transform
transformsPretrained = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
transformsAp = torch.nn.Sequential(
    transformsPretrained,
    transforms.Resize(size),
)

val_dataset = TripletCOCO_Img2Text(val_trainImagesPath, os.listdir(val_trainImagesPath), val_jsonPath, val_jsonPathCap, transformsAp)

device = "cuda"
img_pre_model = FasterRCNN_Triplet_COCO().to(device)

img_model = nn.Sequential(
    nn.Linear(1024, 256),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(256, 128)
).to(device).eval()
img_model.load_state_dict(torch.load(IMG_MODEL_PATH))


loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

cache_outputs(loader, img_pre_model, img_model, cache_filename, device, 1000)