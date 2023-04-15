import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    
from torchvision import models
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import copy
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_metric_learning import distances, losses, miners, reducers, testers


# Creating some helper functions
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def obtainResnet18featureExtractor():
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    # remove the classifier head, leave only feature extractor
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    # add flatten layer to the feature extractor,
    # so shape is (n_samples, n_features) 
    feature_extractor = nn.Sequential(feature_extractor, nn.Flatten())
    
    return feature_extractor

if __name__ == "__main__":
    
    # Set device
    device = "cuda"
    batch_size = 8
    num_epochs = 100
    lr = 1e-3
    
    
    # Resize the images and transform to tensors
    transformation_images = ResNet18_Weights.IMAGENET1K_V1.transforms()
    transformation_targets = transforms.Compose([
                                                 transforms.ToTensor()
                                                 ])
                                                
    # Load the training dataset
    train_dataset = datasets.ImageFolder(root="./MIT_split/train/", transform = transformation_images)
    
    # Create a simple dataloader just for simple visualization
    train_dataloader = DataLoader(train_dataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=batch_size)
    
    
    # Create model
    net = obtainResnet18featureExtractor()
    net = net.to(device)
    
    criterion = losses.ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    
    
    counter = []
    loss_history = [] 
    iteration_number= 0
    
    # Iterate throught the epochs
    for epoch in range(num_epochs):
    
        # Iterate over batches
        for i, (imgs, labels) in enumerate(train_dataloader, 0):
    
            # Send the images and labels to CUDA
            imgs, labels = imgs.to(device), labels.to(device)
    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Pass in the two images into the network and obtain two outputs
            output1 = net(imgs)
    
            # Mining can be used here
            
            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, labels)
    
            # Calculate the backpropagation
            loss_contrastive.backward()
    
            # Optimize
            optimizer.step()
    
            # Every 10 batches print out the loss
            if i % 10 == 0 :
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10
    
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    
    
