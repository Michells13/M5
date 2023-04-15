
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    

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




class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights using Glorot Normalization
        nn.init.xavier_normal_(self.depthwise.weight)
        nn.init.xavier_normal_(self.pointwise.weight)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class VGG_SP_NOFC(nn.Module):
    def __init__(self):
        super(VGG_SP_NOFC, self).__init__()
        # Separable conv 1
        self.sep1 = SeparableConv(3, 16)
        # Batch norm 1
        self.batch1 = nn.BatchNorm2d(16)
        
        # Separable conv 2
        self.sep2 = SeparableConv(16, 32)
        # Batch norm 2
        self.batch2 = nn.BatchNorm2d(32)
        
        # Separable conv 3
        self.sep3 = SeparableConv(32, 64)
        # Batch norm 3
        self.batch3 = nn.BatchNorm2d(64)
        
        # Separable conv 4
        self.sep4 = SeparableConv(64, 128)
        # Batch norm 4
        self.batch4 = nn.BatchNorm2d(128)
        
        # Separable conv 5
        self.sep5 = SeparableConv(128, 256)
        # Batch norm 5
        self.batch5 = nn.BatchNorm2d(256)
        
        # FC
        self.fc = nn.Linear(256, 2)
        nn.init.xavier_normal_(self.fc.weight)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        
        # Global average pooling
        self.gAvPool = nn.AvgPool2d(kernel_size=1, stride=1)
        
        # Activations
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        
        

    def forward(self, input1, input2):
        # Sep 1
        input1 = self.pool(self.batch1(self.relu(self.sep1(input1))))
        # Sep 2
        input1 = self.pool(self.batch2(self.relu(self.sep2(input1))))
        # Sep 3
        input1 = self.pool(self.batch3(self.relu(self.sep3(input1))))
        # Sep 4
        input1 = self.pool(self.batch4(self.relu(self.sep4(input1))))
        # Sep 5
        input1 = self.batch5(self.relu(self.sep5(input1)))
        # Global pooling
        input1 = self.gAvPool(input1)
        input1 = torch.squeeze(input1)
        # FC
        input1 = self.fc(input1)
        
        # Sep 1
        input2 = self.pool(self.batch1(self.relu(self.sep1(input2))))
        # Sep 2
        input2 = self.pool(self.batch2(self.relu(self.sep2(input2))))
        # Sep 3
        input2 = self.pool(self.batch3(self.relu(self.sep3(input2))))
        # Sep 4
        input2 = self.pool(self.batch4(self.relu(self.sep4(input2))))
        # Sep 5
        input2 = self.batch5(self.relu(self.sep5(input2)))
        # Global pooling
        input2 = self.gAvPool(input2)
        input2 = torch.squeeze(input2)
        # FC
        input2 = self.fc(input2)
        

        
        return input1,input2
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

class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        #img0 = img0.convert("L")
        #img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)



# Load the training dataset
folder_dataset = datasets.ImageFolder(root="/media/michell/DSet/MIT_small_train_1/train/")

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()
                                    ])

# Initialize the network
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transformation)

# Create a simple dataloader just for simple visualization
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=2,
                        batch_size=8)

# Extract one batch
example_batch = next(iter(vis_dataloader))

# Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
# If the label is 1, it means that it is not the same person, label is 0, same person in both images
concatenated = torch.cat((example_batch[0], example_batch[1]),0)

imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy().reshape(-1))


#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

# Load the training dataset
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=64)

net = SiameseNetwork().cuda()
#net =VGG_SP_NOFC().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0005 )


counter = []
loss_history = [] 
iteration_number= 0

# Iterate throught the epochs
for epoch in range(100):

    # Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):

        # Send the images and labels to CUDA
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        output1, output2 = net(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(output1, output2, label)

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

show_plot(counter, loss_history)

# Locate the test dataset and load it into the SiameseNetworkDataset
folder_dataset_test = datasets.ImageFolder(root="/media/michell/DSet/MIT_small_train_1/test/")
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transformation)
test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)

# Grab one image that we are going to test
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    # Iterate over 10 images and test them with the first image (x0)
    _, x1, label2 = next(dataiter)

    # Concatenate the two images together
    concatenated = torch.cat((x0, x1), 0)
    
    output1, output2 = net(x0.cuda(), x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')

