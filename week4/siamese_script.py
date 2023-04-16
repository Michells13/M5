import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    
import optuna
from similarRetrieval import FAISSretrieval, KNNretrieval
from metrics import mPrecisionK, mRecallK, MAP, precisionRecall
import wandb
# Creating some helper functions


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        if mining_func is None:
            loss = loss_func(embeddings, labels)
        else:
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )

            )
            wandb.log({"iteration": batch_idx + (epoch-1)*len(train_loader), "loss": loss.item()})


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    train_embeddings = train_embeddings.cpu().numpy()
    test_embeddings = test_embeddings.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    
    retrieval = KNNretrieval(train_embeddings, "l2", train_embeddings.shape[0])
    (dis, neighbors) = retrieval.getMostSimilar(test_embeddings, train_embeddings.shape[0])
    results = []
    for i, label in enumerate(test_labels):
        results.append((train_labels[neighbors[i]] == label).astype(np.int32))
    results = np.array(results)
    
    # Compute metrics
    print("P@1: ", mPrecisionK(results, 1))
    print("P@5: ", mPrecisionK(results, 5))
    print("MAP: ", MAP(results))

if __name__ == "__main__":

    # Device
    device = torch.device("cuda")
    
    # Default transforms
    transformation_images = ResNet18_Weights.IMAGENET1K_V1.transforms()
    
    # Init datasets
    dataset_train = datasets.ImageFolder(root="./MIT_split/train/",transform=transformation_images)
    dataset_test = datasets.ImageFolder(root="./MIT_split/test/",transform=transformation_images)
    
    
    
    # Define hyperparameters
    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes =   [16, 32, 64, 128]
    minerOpts = ["hard", "semihard", "no"]
    num_epochs = 1#100
    batch_size = 32
    lr = 1e-5
    minerOpt = "no"
    
    for lr in lrs:
        # Init model
        model1 = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor = torch.nn.Sequential(*(list(model1.children())[:-1]))
        model = nn.Sequential(feature_extractor, nn.Flatten()).to(device)
        
        run = wandb.init(project='M5_WEEK4', job_type='train')
        wandb.run.name = "Resnet_siamese_lr_" + str(lr) + "_batchSize_" + str(batch_size) + "_miner_" + minerOpt

        print("Learning rate: ", lr)
        print("Batch size: ", batch_size)
        print("Miner: ", minerOpt)

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        ### pytorch-metric-learning stuff ###
        distance = distances.LpDistance(power=2)
        loss_func = losses.ContrastiveLoss(distance = distance)
        
        if minerOpt == "no":
            mining_func = None
        else:
            mining_func = miners.BatchEasyHardMiner(
                neg_strategy=minerOpt
            )
        
        ### pytorch-metric-learning stuff ###
        for epoch in range(1, num_epochs + 1):
            train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        
        test(dataset_train, dataset_test, model)
        torch.save(model.state_dict(), "resnet_siamese_lr_" + str(lr) + "_batchSize_" + str(batch_size) + "_miner_" + minerOpt + ".pth")

        wandb.finish()
        