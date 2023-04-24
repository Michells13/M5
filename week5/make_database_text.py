from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from cocoTripletDataset import TripletCOCO_Img2Text
import os
import fasttext
import re

def prep_sentence(sentence, text_pre_model):
    sentence = text_pre_model.get_sentence_vector(
        re.sub(r'[^a-zA-Z ]', '', sentence).lower()
    )
    return sentence

def cache_outputs(dataset, pre_model, text_model, cache_filename, device, max_rows):
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
    Returns:
        None
    """
    text_model.eval()
    with torch.no_grad():
        f = open(cache_filename, "wb")
        for idx, (_, pos, _) in enumerate(dataset):
            print(idx)
            if idx == max_rows:
                break
            pos = prep_sentence(pos, text_pre_model)
            # pos = list(pos)
            # if len(pos) > 5:
            #     pos = pos[:5]
            # if len(pos) < 5:
            #     for _ in range(5 - len(pos)):
            #         pos.append("placeholder")
            # pos = [prep_sentence(p, text_pre_model) for p in pos]
            pos = torch.tensor(pos, device=device).unsqueeze(0)
            output = text_model(pos)
            np.savetxt(f, output.cpu().numpy())
        f.close()


cache_filename = r"text_final2.txt"

val_jsonPath = r"D:\coco\annotations_trainval2014\annotations\person_keypoints_val2014.json"
val_jsonPathCap= r"D:\coco\annotations_trainval2014\annotations\captions_val2014.json"
val_trainImagesPath = r"D:\coco\val2014\val2014"

TEXT_MODEL_PATH = r"D:\img2text\last_weight_text.pth"

fasttext_model_path = r"D:\fasttext_wiki.en.bin"
text_pre_model = fasttext.load_model(path=fasttext_model_path)


device = "cuda"
text_model = nn.Sequential(
    nn.Linear(300, 256),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(256, 128)
).to(device).eval()
text_model.load_state_dict(torch.load(TEXT_MODEL_PATH))



val_dataset = TripletCOCO_Img2Text(val_trainImagesPath, os.listdir(val_trainImagesPath), val_jsonPath, val_jsonPathCap, None, random_positive=False, return_all_captions=False)
# loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

cache_outputs(val_dataset, text_pre_model, text_model, cache_filename, device, 4000)