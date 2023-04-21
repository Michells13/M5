import os
from pytorch_metric_learning import losses, miners
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision
from cocoTripletDataset import TripletCOCO, TripletCOCO_Text2Img
import torch
import torch.optim as optim
from torchvision import transforms
import wandb
import numpy as np
from net import ResNet_Triplet_COCO, FasterRCNN_Triplet_COCO,ImgModel, TextModel
import fasttext

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        loss_vals = torch.relu(distance_positive - distance_negative + self.margin)
        return loss_vals.mean()

def trainText2Img(imgFeatureExt,textFeatureExt,imgModel,textModel, train_loader, optimizer, num_epochs, loss_func, device ):
    # Train the number of epochs given
    for epoch in range(1, num_epochs + 1):
        
        # Train with batches
        for batch_idx, data in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            # Store batches in train device
            data = data[0]
            captions, pos_img, neg_img = data
            pos_img, neg_img = pos_img.to(device), neg_img.to(device)
            captions = [caption.lower() for caption in captions]
            
            # Get text and image features
            captionFeatures = [np.expand_dims(textFeatureExt.get_sentence_vector(caption), 0) for caption in captions]  
            captionFeatures = np.vstack(captionFeatures)
            captionFeatures = torch.from_numpy(captionFeatures) 
            captionFeatures = captionFeatures.to(device)
            pos_img = imgFeatureExt(pos_img)
            neg_img = imgFeatureExt(neg_img)
            
            # Forward pass
            anchors = textModel(captionFeatures)
            positives = imgModel(pos_img)
            negatives = imgModel(neg_img)
            
            # Triplet loss
            loss = loss_func(anchors, positives, negatives)
            
            # Backward pass and update
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    "Epoch {} Iteration {}: Loss = {}, Number of triplets = {}".format(
                        epoch, batch_idx, loss, anchors.shape[0]
                    )
                )
                wandb.log({"iteration": batch_idx + (epoch-1)*len(train_loader), "loss": loss.item()})
    


def train(model, train_loader, optimizer, num_epochs, loss_func, device = "cuda"):
    """
    This function trains the model using the triplet loss

    Parameters
    ----------
    model : model
        Model to get the features of an image.
    train_loader : dataset loader
        Dataset loader that returns triplets.
    optimizer : optimizer
        Optimizer to train the model.
    num_epochs : int
        Number of epochs to train.
    loss_func : function
        Loss function.
    device : str, optional
        Device to train. The default is "cuda".

    Returns
    -------
    None.

    """

    # Model in train mode
    model.train()
    
    # Train the number of epochs given
    for epoch in range(1, num_epochs + 1):
        
        # Train with batches
        for batch_idx, (data, labels) in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            # Store batches in train device
            data = [img.to(device) for img in data]
            
            # Forward pass
            anchors = model(data[0])
            positives = model(data[1])
            negatives = model(data[2])
            
            # Triplet loss
            loss = loss_func(anchors, positives, negatives)
            
            # Backward pass and update
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    "Epoch {} Iteration {}: Loss = {}, Number of triplets = {}".format(
                        epoch, batch_idx, loss, anchors.shape[0]
                    )
                )
                wandb.log({"iteration": batch_idx + (epoch-1)*len(train_loader), "loss": loss.item()})
    

if __name__ == "__main__":
    
    run = wandb.init(project='M5_WEEK5', job_type='train')
    wandb.run.name = "COCO_img2txt"

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Define img and text models to train
    image_model = ImgModel()
    text_model = TextModel()
    image_model.to(device)
    text_model.to(device)
    # init weights
    image_model.init_weights()
    text_model.init_weights()

        
    # Paths and some train params
    transformsPretrained = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
    jsonPathCap= "../../WEEK4/COCO/captions_train2014.json"
    trainImages= "../../WEEK4/COCO/train2014/"
    batch_size = 32
    epochs = 5
    size = (240,320)
    marginVal = 0.1  
    weighted = False
    
    # Load img feature Extractor
    imgFeatureExt = FasterRCNN_Triplet_COCO(weighted = weighted)
    imgFeatureExt = imgFeatureExt.to(device)
    
    #Load text extractor
    textFeatureExt = fasttext.load_model("ag_news.bin")
    
    
    
    # Set trainable all parameters
    for param in image_model.parameters():
        param.requires_grad = True
    for param in text_model.parameters():
        param.requires_grad = True
    
    # Freeze feature extractors
    for param in imgFeatureExt.parameters():
        param.requires_grad = False
    # for param in textFeatureExt.parameters():
    #     param.requires_grad = False
    
    # Transform
    transformsAp = torch.nn.Sequential(
        transformsPretrained,
        transforms.Resize(size),
    )

    # Init dataset
    train_dataset = TripletCOCO_Text2Img(trainImages, os.listdir(trainImages), jsonPathCap, transformsAp)
    #train_dataset = TripletCOCOproba(trainImagesPath, trainImages, jsonPath, transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, shuffle=True)#,
                                               #num_workers = 2)
    
    # Init optimizer and join both models
    lr = 1e-5
   
    params = list(image_model.parameters())
    params += list(text_model.parameters())
    optimizer = optim.AdamW(params, lr=lr)
    
    # Init loss func
    loss_fc = TripletLoss(margin = marginVal)
    
    # Train model

    trainText2Img(imgFeatureExt,textFeatureExt,image_model,text_model, train_loader, optimizer, epochs, loss_fc, device )
    # Save both models
    state_dict = [image_model.state_dict(), text_model.state_dict()]
    torch.save(state_dict,"bothModels.pth")
    

    
    