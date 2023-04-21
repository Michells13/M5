from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import os
import cv2
from pycocotools.coco import COCO
import random




class TripletCOCO_Img2Text(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative caption
    """

    def __init__(self, trainImagesFolder, trainImages, captionLabels, transform):
        # Opening JSON file

        fcap = open(captionLabels)
        cocoCapJson= json.load(fcap)
        fcap.close()
        self.transform = transform
        self.trainImagesFolder = trainImagesFolder
        self.trainImages = trainImages
        
        self.annotations_by_image = {}
        
        for ann in cocoCapJson['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
            self.objs = self.annotations_by_image
  
        
        
        # Remove images that don't have captions
        i1 = 0
        while i1 < len(self.trainImages):
            image1 = self.trainImages[i1]
            image1Num = int(image1[:-4].split("_")[2])
            
            if not(image1Num in self.annotations_by_image.keys()):
                del self.trainImages[i1]
            else:
                i1 += 1

    
    def __getitem__(self, index):
        # Get anchor image
        img1name = self.trainImages[index]
        img1 = cv2.imread(self.trainImagesFolder + img1name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Get random positive caption
        cap_pos = int(img1name[:-4].split("_")[2])
        image_captionsP = [ann['caption'] for ann in self.objs[cap_pos]]
        random_index = random.randrange(0, len(image_captionsP))
        positiveCaption = image_captionsP[random_index]
        
        
        # Get random negative caption
        # Get random image
        img3name = np.random.choice(self.trainImages)
        cap_neg = int(img3name[:-4].split("_")[2])
        image_captionsN = [ann['caption'] for ann in self.objs[cap_neg]]
        random_index = random.randrange(0, len(image_captionsN))
        negativeCaption = image_captionsN[random_index]
            

    
        
        # Transform
        img1 = Image.fromarray(img1)

        if self.transform is not None:
            img1 = self.transform(img1)

        return (img1, positiveCaption, negativeCaption), []

    def __len__(self):
        return len(self.trainImages)




class TripletCOCO_Text2Img(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative caption
    """

    def __init__(self, trainImagesFolder, trainImages, captionLabels, transform):
        # Opening JSON file

        fcap = open(captionLabels)
        cocoCapJson= json.load(fcap)
        fcap.close()
        self.transform = transform
        self.trainImagesFolder = trainImagesFolder
        self.trainImages = trainImages
        
        self.annotations_by_image = {}
        
        for ann in cocoCapJson['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
            self.objs = self.annotations_by_image
  
        
        # Remove images that don't have captions
        i1 = 0
        while i1 < len(self.trainImages):
            image1 = self.trainImages[i1]
            print(int(image1[:-4].split("_")[2]))
            image1Num = int(image1[:-4].split("_")[2])
            
            if not(image1Num in self.objs.keys()):
                del self.trainImages[i1]
            else:
                i1 += 1

    def __getitem__(self, index):
        
        # Get anchor image
        img1name = self.trainImages[index]
        img1 = cv2.imread(self.trainImagesFolder + img1name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Get random positive caption
        cap_pos = int(img1name[:-4].split("_")[2])
        image_captionsP = [ann['caption'] for ann in self.objs[cap_pos]]
        random_index = random.randrange(0, len(image_captionsP))
        positiveCaption = image_captionsP[random_index]
        
        # Get random negative image

        img3name = np.random.choice(self.trainImages)
        img3value = int(img3name[:-4].split("_")[2])
            
        negativeImg = cv2.imread(self.trainImagesFolder + img3name)
        negativeImg = cv2.cvtColor(negativeImg, cv2.COLOR_BGR2RGB)
        
        # Transform
        img1 = Image.fromarray(img1)
        negativeImg = Image.fromarray(negativeImg)
        if self.transform is not None:
            img1 = self.transform(img1)
            negativeImg = self.transform(negativeImg)
        return (positiveCaption, img1, negativeImg), []

    def __len__(self):
        return len(self.trainImages)        
        
class TripletCOCO(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, trainImagesFolder, trainImages, trainImageLabels, transform):
        # Opening JSON file
        f = open(trainImageLabels)
        labelJson = json.load(f)
        f.close()
        self.labelTrain = labelJson["train"]
        self.transform = transform
        self.trainImagesFolder = trainImagesFolder
        self.trainImages = trainImages
        
        # Obtain labels
        self.objs = {}
        
        # Get objects per image  // Get caps per image
        for obj in self.labelTrain.keys():
            for image in self.labelTrain[obj]:
                if image in self.objs.keys():
                    self.objs[image].append(obj)
                else:
                    self.objs[image] = [obj]
        
        # Remove images that don't have captions
        i1 = 0
        while i1 < len(self.trainImages):
            image1 = self.trainImages[i1]
            image1Num = int(image1[:-4].split("_")[2])
            
            if not(image1Num in self.objs.keys()):
                del self.trainImages[i1]
            else:
                i1 += 1

    
    def __getitem__(self, index):
        # Get anchor image
        img1name = self.trainImages[index]
        img1 = cv2.imread(self.trainImagesFolder + img1name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Get random positive image
        img1value = int(img1name[:-4].split("_")[2])
        img1objs = self.objs[img1value]
        
        positiveImgValue = img1value
        while positiveImgValue == img1value:
            # Get random obj
            sharingObj = np.random.choice(img1objs)
            # Get random image 
            positiveImgValue = np.random.choice(self.labelTrain[sharingObj])
        img2name = "COCO_train2014_{:012d}.jpg".format(positiveImgValue)
        img2 = cv2.imread(self.trainImagesFolder + img2name)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Get random negative image
        while True:
            # Get random image
            img3name = np.random.choice(self.trainImages)
            img3value = int(img3name[:-4].split("_")[2])
            img3objs = self.objs[img3value] 
            
            #Not needed for text stuuf
            # if not firstStrategy(img3objs, img1objs):
            #     break
        
        img3 = cv2.imread(self.trainImagesFolder + img3name)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        
        # Transform
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.trainImages)

class TripletCOCOdatabase(Dataset):
    """
    Dataset for retrieval
    """

    def __init__(self, databaseImagesFolder, databaseImages, databaseImageLabels, 
                 transform, section, allLabels  = None):
        
        # Opening JSON file
        f = open(databaseImageLabels)
        labelJson = json.load(f)
        f.close()
        self.labelDatabase = labelJson[section]
        self.transform = transform
        self.databaseImagesFolder = databaseImagesFolder
        self.databaseImages = databaseImages
        
        # Obtain labels
        self.objs = {}
        
        # Get objects per image
        for obj in self.labelDatabase.keys():
            for image in self.labelDatabase[obj]:
                if image in self.objs.keys():
                    self.objs[image].append(obj)
                else:
                    self.objs[image] = [obj]
        
        # Rem images without images
        i1 = 0
        while i1 < len(self.databaseImages):
            image1 = self.databaseImages[i1]
            image1Num = int(image1[:-4].split("_")[2])
            
            if not(image1Num in self.objs.keys()):
                del self.databaseImages[i1]
            else:
                i1 += 1
        
        if not(allLabels is None):
            # Get every object in the image
            coco=COCO(allLabels)
            
            # Obtain labels
            self.objs = {}
            for image in self.databaseImages:
                imageId = int(image[:-4].split('_')[-1])
                ann_ids = coco.getAnnIds(imgIds=[imageId])
                anns = coco.loadAnns(ann_ids)
                annId = []
                for ann in anns:
                    annId.append(str(ann["category_id"]))
                if len(annId)>0:
                    self.objs[imageId]=annId

    def __getitem__(self, index):
        # Get image
        img1name = self.databaseImages[index]
        img1 = cv2.imread(self.databaseImagesFolder + img1name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        
        # Transform
        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1, []

    def getObjs(self, index):
        # Get image name
        img1name = self.databaseImages[index]

        # Get objs
        img1value = int(img1name[:-4].split("_")[2])
        
        img1objs = self.objs[img1value]
        
        return img1objs
        
    def __len__(self):
        return len(self.databaseImages)


if __name__ == "__main__":
    
    jsonPath = "/media/michell/DSet/annotations_trainval2014/annotations/person_keypoints_train2014.json"
    jsonPathCap= "/media/michell/DSet/annotations_trainval2014/annotations/captions_train2014.json"
    trainImagesPath = "/media/michell/DSet/train2014"
    trainImages = os.listdir(trainImagesPath)

    dataset = TripletCOCO_Text2Img(trainImagesPath, trainImages, jsonPathCap,None)