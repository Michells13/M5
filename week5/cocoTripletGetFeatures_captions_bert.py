from cocoTripletDataset import TripletCOCOdatabase_Img2Text
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2
from net import FasterRCNN_Triplet_COCO
import fasttext
from torchvision import transforms
import torch
import os
from tqdm import tqdm
import time
import numpy as np
from torch import nn
import re
from transformers import AutoTokenizer, AutoModel

def cache_outputs_coco_captions(loader, text_pre_model_tok, text_pre_model_model, modelText,  img_pre_model, modelImage, cache_filename, device):
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
        device: device to perform inference

    Returns:
        None
    """

    modelImage.eval()
    modelText.eval()
    timesText = []
    timesImages = []
    with torch.no_grad():
        fText = open(cache_filename[:-4] + "_captions.txt", "wb")
        fImages = open(cache_filename[:-4] + "_images.txt", "wb")
        for data, _ in tqdm(loader):
            data, captions = data
            start = time.time()
            captions = [re.sub(r'[^a-zA-Z ]', '', sentence.lower()) for sentence in captions]
            captions = text_pre_model_model(input_ids = captions["input_ids"].to(device), attention_mask = captions["attention_mask"].to(device))
            captionFeatures = modelText(captions)
            stop = time.time()
            timesText.append(stop-start)
            np.savetxt(fText, captionFeatures.cpu().numpy())
            
            data = data.to(device)
            start = time.time()
            output = img_pre_model(data)
            output = output.reshape((-1, output.shape[1]))
            output = modelImage(output)
            stop = time.time()
            timesImages.append(stop-start)
            np.savetxt(fImages, output.cpu().numpy())
        fText.close()
        fImages.close()
    
    print("Median text inference time: ", np.median(np.array(timesText)))
    print("Median image inference time: ", np.median(np.array(timesImages)))


if __name__ == "__main__":
    
    device = "cuda"
    
    modelImages = nn.Sequential(
        nn.Linear(1024, 256),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).to(device).eval()

    modelText = nn.Sequential(
        nn.Linear(768, 256),
        nn.Dropout(),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).to(device).eval()
    
    
    batch_size = 1
    size = (240,320)#(480, 640)#(240,320)
    allCaptions = "../WEEK4/COCO/captions_val2014.json"
    fasttext_model_path = "fasttext_wiki.en.bin"
    # Transform
    transformsPretrained = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    transformsAp = torch.nn.Sequential(
        transformsPretrained,
        transforms.Resize(size),
    )

    text_pre_model_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
    text_pre_model_model = AutoModel.from_pretrained('bert-base-uncased').to(device).eval()


    img_pre_model = FasterRCNN_Triplet_COCO().to(device)
    
    # Get backbone
    #model = model.backbone
    modelImages = modelImages.to(device)
    # Load trained weights
    weightsText = "best_text_taskb.pth"
    weightsImages = "best_img_taskb.pth"
    modelText.load_state_dict(torch.load(weightsText, map_location=device))
    modelImages.load_state_dict(torch.load(weightsImages, map_location=device))
    
    
    # Transform
    transforms = torch.nn.Sequential(
        transformsPretrained,
        transforms.Resize(size),
    )
    
    # Init  database
    img_num = 4000
    databaseImagesPath = "../WEEK4/COCO/val2014/"#"./COCO/train2014/"
    databaseImages = os.listdir(databaseImagesPath)
    database_dataset = TripletCOCOdatabase_Img2Text(databaseImagesPath, databaseImages,
                                           allCaptions, transforms, img_num)
    database_loader = torch.utils.data.DataLoader(database_dataset, 
                                               batch_size=batch_size, shuffle=False)#, collate_fn=collate_fn)
    
    cache_filename = "features_" + str(img_num) + "_.txt"

    cache_outputs_coco_captions(database_loader, text_pre_model_tok, text_pre_model_model, modelText, img_pre_model, modelImages, cache_filename, device)
    