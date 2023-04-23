import torch
from pycocotools.coco import COCO
import fasttext
from cocoTripletDataset import TripletCOCO_Img2Text, TripletCOCO_Text2Img
import os
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2
from torch.utils.data import DataLoader
from net import FasterRCNN_Triplet_COCO
from torchvision import transforms
import re
import numpy as np
import wandb


train_jsonPath = r"/ghome/group04/mcv/datasets/COCO/instances_train2014.json"
train_jsonPathCap= r"/ghome/group04/mcv/datasets/COCO/captions_train2014.json"
train_trainImagesPath = r"/ghome/group04/mcv/datasets/COCO/train2014"

val_jsonPath = r"/ghome/group04/mcv/datasets/COCO/instances_val2014.json"
val_jsonPathCap= r"/ghome/group04/mcv/datasets/COCO/captions_val2014.json"
val_trainImagesPath = r"/ghome/group04/mcv/datasets/COCO/val2014"

fasttext_model_path = r"/home/mcv/m5/fasttext_wiki.en.bin"

out_text_model_path = r"best_text_taskb.pth"
out_img_model_path = r"best_img_taskb.pth"

device = "cuda"
batch_size = 32
n_epochs = 4
log_interval = 5
size = (240,320)


text_pre_model = fasttext.load_model(path=fasttext_model_path)
img_pre_model = FasterRCNN_Triplet_COCO().to(device)

 


# Transform
transformsPretrained = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
transformsAp = torch.nn.Sequential(
    transformsPretrained,
    transforms.Resize(size),
)

img_model = nn.Sequential(
    nn.Linear(1024, 256),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(256, 128)
).to(device).eval()

text_model = nn.Sequential(
    nn.Linear(300, 256),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(256, 128)
).to(device).eval()

train_dataset = TripletCOCO_Text2Img(train_trainImagesPath, os.listdir(train_trainImagesPath), train_jsonPath, train_jsonPathCap, transformsAp)
val_dataset = TripletCOCO_Text2Img(val_trainImagesPath, os.listdir(val_trainImagesPath), val_jsonPath, val_jsonPathCap, transformsAp)


def sentence2vector(sentence, model):
    sentence = re.sub(r'[^a-zA-Z]', '', sentence).lower()
    words = sentence.split()
    vectors = [model.get_word_vector(word) for word in words]
    summary_vector = np.mean(vectors, axis=0)
    
    return summary_vector


criterion = nn.TripletMarginLoss()
optimizer = torch.optim.Adam(
    list(img_model.parameters()) + list(text_model.parameters()),
    lr=3e-4
)



def prep_sentence(sentence, text_pre_model):
    sentence = text_pre_model.get_sentence_vector(
        re.sub(r'[^a-zA-Z ]', '', sentence.lower())
    )
    return sentence



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

wandb.init(project="text2img")
min_val_loss = 100
for epoch in range(n_epochs):
    img_model.train()
    text_model.train()
    
    for img, pos, neg in train_loader:
        
        img = img.to(device)
        pos = [prep_sentence(p, text_pre_model) for p in pos]
        neg = neg.to(device)
        pos = torch.tensor(pos, device=device)
        with torch.no_grad():
            img_feature_vector = img_pre_model(img)
            neg_feature_vector = img_pre_model(neg)
            
        img_common = img_model(img_feature_vector.reshape((-1, img_feature_vector.shape[1])))
        neg_img = img_model(neg_feature_vector.reshape((-1, neg_feature_vector.shape[1])))
        pos_common  = text_model(pos)

        loss = criterion(pos_common,img_common, neg_img)
        img_model.zero_grad()
        text_model.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if epoch % log_interval == 0:
        print(f"Train loss: {loss.item()}")
        wandb.log({"train_loss": loss})
    
    img_model.eval()
    text_model.eval()
    losses = []
    for img, pos, neg in val_loader:
        img = img.to(device)
        pos = [prep_sentence(p, text_pre_model) for p in pos]
        neg = neg.to(device)
        pos = torch.tensor(pos, device=device)
        with torch.no_grad():
            img_feature_vector = img_pre_model(img)
            neg_feature_vector = img_pre_model(neg)
            
        img_common = img_model(img_feature_vector.reshape((-1, img_feature_vector.shape[1])))
        neg_img = img_model(neg_feature_vector.reshape((-1, neg_feature_vector.shape[1])))
        pos_common  = text_model(pos)

        loss = criterion(pos_common,img_common, neg_img)
        losses.append(loss.item())
    val_loss = torch.mean(torch.tensor(losses))
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(img_model.state_dict(), out_img_model_path)
        torch.save(text_model.state_dict(), out_text_model_path)
    print(f"Val loss: {val_loss}")
    wandb.log({"val_loss": val_loss})
