from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from retrieve_utils import cache_outputs
import torch
torch.manual_seed(0)
from torch import nn


train_dataset_dir = r"D:\MCV-M5-Team04\MIT_train_val_test\MIT_train_val_test\train"
val_dataset_dir = r"D:\MCV-M5-Team04\MIT_train_val_test\MIT_train_val_test\val"

# Resize the images and transform to tensors
transformation_images = ResNet18_Weights.IMAGENET1K_V1.transforms()   
         
train_dataset = ImageFolder(train_dataset_dir, transform=transformation_images)
val_dataset = ImageFolder(val_dataset_dir, transform=transformation_images)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
device = "cuda"
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model.to(device)



# create training uitilities: a loss function (criterion), an optimizer and a scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

epoch_val_losses = []  # create list to save mean validation batch loss for every epoch
                        # to save the best model (with the least validation loss)

# train the model with validation
num_epochs = 100
for epoch in range(num_epochs):
    
    # training stage
    model.train()
    
    train_batch_losses = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # move mini-batches to proper device
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        
        loss = criterion(output, target)
        
        train_batch_losses.append(loss.item())
        
        # cast models gradients to zero
        model.zero_grad()  # if all models parameters are passed to the optimizer,
                            # it is the same as optimizer.zero_grad()
        
        # caclulate gradients
        loss.backward()
        
        # optimize parameters
        optimizer.step()
        
        # print log info
        if batch_idx % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}\nLoss: {loss.item():4f}\n')
    
    # validation stage
    model.eval()  # de-activate Dropout layers, make normalisation layers use running statistics
    
    val_batch_losses = []
    
    for batch_idx, (data, target) in enumerate(train_loader):

        # move mini-batches to proper device
        data = data.to(device)
        target = target.to(device)
                    
        with torch.no_grad():  # disable the gradient computation
            output = model(data)
            loss = criterion(output, target)    
        val_batch_losses.append(loss.item())
    
    mean_val_loss = sum(val_batch_losses) / len(val_batch_losses)
    epoch_val_losses.append(mean_val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}\nValidation loss: {mean_val_loss:4f}\n\n')
    
    # decrease the learning rate WRT validation loss
    scheduler.step(mean_val_loss)
    
    # save model
    if mean_val_loss == min(epoch_val_losses):
        torch.save(model.state_dict(), 'least_loss.pt')
    torch.save(model.state_dict(), 'latest.pt')

