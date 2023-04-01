from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from retrieve_utils import cache_outputs
import torch
from torch import nn


dataset_dir = r"/home/user/MIT_train_val_test/val"
cache_filename = r"output_cache.txt"

transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = ImageFolder(dataset_dir, transform=transform)
loader = DataLoader(dataset, batch_size=4)

model = resnet18(ResNet18_Weights)
# remove the classifier head, leave only feature extractor
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
# add flatten layer to the feature extractor,
# so shape is (n_samples, n_features) 
feature_extractor = nn.Sequential(feature_extractor, nn.Flatten())

cache_outputs(loader, feature_extractor, cache_filename)
