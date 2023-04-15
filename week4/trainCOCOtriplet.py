import os
from pytorch_metric_learning import losses, miners
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision
from cocoTripletDataset import TripletCOCO, TripletCOCOproba
import torch
import torch.optim as optim
from torchvision import transforms
import wandb
import numpy as np
from net import ResNet_Triplet_COCO, FasterRCNN_Triplet_COCO

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

def trainNew(model, train_loader, optimizer, num_epochs, loss_func, device = "cuda"):
    miner = miners.BatchEasyHardMiner()
    
    model.train()
    for epoch in range(1, num_epochs + 1):
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            embeddings = model(data)
            miner_output = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, miner_output)
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                print(
                    "Epoch {} Iteration {}: Loss = {}, Number of triplets = {}".format(
                        epoch, batch_idx, loss,  miner_output[0].shape[0]#len(indices_tuple[0])
                    )
                )
                wandb.log({"iteration": batch_idx, "loss": loss.item()})


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
    
    marginValues = [0.1, 5, 100]
    weightedValues = [False, True]
    for weighted in weightedValues:
        for marginVal in marginValues:
        
            # Init wandb
            run = wandb.init(project='M5_WEEK4', job_type='train')
            wandb.run.name = "COCO_triplet_fasterrcnn_weighted_" + str(weighted) + "_margin_" + str(marginVal)
            
            # Train params
            transformsPretrained = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
            jsonPath = "./COCO/mcv_image_retrieval_annotations.json"
            trainImagesPath = "./COCO/train2014/"
            trainImages = os.listdir(trainImagesPath)
            batch_size = 32
            n_epochs = 5
            device = "cuda"
            size = (240,320)#(480,640)
            
            # Load Resnet50 pretrained in COCO
            #model = ResNet_Triplet_COCO()
            model = FasterRCNN_Triplet_COCO(weighted = weighted)
            model = model.to(device)
            
            # Set trainable all parameters
            for param in model.parameters():
                param.requires_grad = True
            
            # Transform
            transformsAp = torch.nn.Sequential(
                transformsPretrained,
                transforms.Resize(size),
            )
        
            # Init dataset
            train_dataset = TripletCOCO(trainImagesPath, trainImages, jsonPath, transformsAp)
            #train_dataset = TripletCOCOproba(trainImagesPath, trainImages, jsonPath, transforms)
        
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                       batch_size=batch_size, shuffle=True)#,
                                                       #num_workers = 2)
            
            # Init optimizer
            lr = 1e-5
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            
            # Init loss func
            loss_fc = TripletLoss(margin = marginVal)
            
            # Train model
            train(model, train_loader, optimizer, n_epochs, loss_fc, device)
            
            # Save model
            torch.save(model.state_dict(), "trained_faster_objdet_weighted_" + str(weighted) + "_margin_" + str(marginVal) + ".pth")
            
            wandb.finish()
    
    