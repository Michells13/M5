import os
from pytorch_metric_learning import losses
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from cocoTripletDataset import TripletCOCO
import torch
import torch.optim as optim
from torchvision import transforms
import wandb



def train(model, train_loader, optimizer, num_epochs, loss_func, device = "cuda"):
    
    model.train()
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = torch.vstack(data)
            data = data.to(device)
            optimizer.zero_grad()
            embeddings = model(data)
            embeddings = torch.flatten(embeddings["pool"], 1)
            indices_tuple = (tuple(range(0, embeddings.shape[0], 3)),
                             tuple(range(1, embeddings.shape[0], 3)),
                             tuple(range(2, embeddings.shape[0], 3)))
            loss = loss_func(embeddings, indices_tuple = indices_tuple)
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                print(
                    "Epoch {} Iteration {}: Loss = {}, Number of triplets = {}".format(
                        epoch, batch_idx, loss, len(indices_tuple[0])
                    )
                )
                wandb.log({"iteration": batch_idx, "loss": loss.item()})
    
    

if __name__ == "__main__":
    run = wandb.init(project='M5_WEEK4', job_type='train')
    wandb.run.name = "COCO_triplet"
    
    # Train params
    transformsPretrained = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
    jsonPath = "./COCO/mcv_image_retrieval_annotations.json"
    trainImagesPath = "./COCO/train2014/"
    trainImages = os.listdir(trainImagesPath)
    batch_size = 8
    n_epochs = 1
    device = "cuda"
    size = (240,320)#(480,640)
    
    # Load MASK RCNN pretrained in COCO
    model = maskrcnn_resnet50_fpn_v2(MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    # Get backbone
    model = model.backbone.body
    model = model.to(device)
    
    # Set trainable all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Transform
    transforms = torch.nn.Sequential(
        transformsPretrained,
        transforms.Resize(size),
    )

    # Init dataset
    train_dataset = TripletCOCO(trainImagesPath, trainImages, jsonPath, transforms)
    #train_dataset.trainImages = train_dataset.trainImages[:1000]### REMOVE THIS
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, shuffle=True)#, collate_fn=collate_fn)
    
    # Init optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Init loss func
    loss_fc = losses.TripletMarginLoss()
    
    # Train model
    train(model, train_loader, optimizer, n_epochs, loss_fc, device)
    
    # Save model
    torch.save(model.state_dict(), "trained_Mask_backbone.pth")
    
    wandb.finish()
    
    