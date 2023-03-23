from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import cv2
from noise import addBlack, addGaussianNoise, addRandomNoise, addSPNoise
from matplotlib import pyplot as plt
import os

def bboxToMask(image, bbox):
    """
    This function creates a mask of a bbox.

    Parameters
    ----------
    image : numpy array
        Original image.
    bbox : numpy array
        BBox [xmin, ymin, w, h].

    Returns
    -------
    mask : numpy array
        Mask with 1 values in the random bbox pixels.

    """
    mask = np.zeros(image.shape[:2], np.uint8)
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2] - 1), int(bbox[1] + bbox[3] - 1)
    mask = cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color = 1, thickness = -1)

    return mask

def getRandomPartOutsideBox(image, bbox, maskBox):
    """
    This functions returns a mask with a random area outside the object bbox in the image.

    Parameters
    ----------
    image : numpy array
        Original image.
    bbox : numpy array
        BBox [xmin, ymin, w, h].
    maskBox : numpy array
        Mask with True in pixels that are out of the object bbox.

    Returns
    -------
    maskNoise : numpy array
        Mask with 1 values in the random bbox pixels.

    """
    
    valid = False
    # Set minimum number of pixels
    minPixels = image.shape[0]*image.shape[1]/(8*8)
    # Set width and height
    box_width = image.shape[1]/4
    box_height = image.shape[0]/4
    
    # Find a random valid bbox
    while not valid:
        x = np.random.randint(0, image.shape[1] - box_width)
        y = np.random.randint(0, image.shape[0] - box_height)
        
        bbox_noise = np.array([x, y, box_width, box_height])
        maskNoise = bboxToMask(image, bbox_noise)
        
        # Outside the object bbox
        maskNoise[maskBox == False] = 0
        pixelCount = np.sum(maskNoise == 1)
        valid = pixelCount > minPixels
        
    return maskNoise

def getRandomPartInsideBox(image, bbox, maskSeg):
    """
    This functions returns a mask with a random part inside the object bbox in the image.

    Parameters
    ----------
    image : numpy array
        Original image.
    bbox : numpy array
        BBox [xmin, ymin, w, h].
    maskSeg : numpy array
        Mask with True in pixels that are out of the object region.

    Returns
    -------
    maskNoise : numpy array
        Mask with 1 values in the random area pixels.

    """
    
    valid = False
    # Set minimum number of pixels
    minPixels = bbox[2]*bbox[3]/(8*8)
    # Set width and height
    box_width = bbox[2]/4
    box_height = bbox[3]/4
    
    # Find a random valid bbox
    while not valid:
        x = np.random.randint(bbox[0], bbox[0] + bbox[2] - box_width)
        y = np.random.randint(bbox[1], bbox[1] + bbox[3] - box_height)
        
        bbox_noise = np.array([x, y, box_width, box_height])
        maskNoise = bboxToMask(image, bbox_noise)
        
        # Outside the object bbox
        maskNoise[maskSeg == False] = 0
        pixelCount = np.sum(maskNoise == 1)
        valid = pixelCount > minPixels
        
    return maskNoise
    

if __name__ == '__main__':

    # Save funcs and names
    differentNoiseFuncs = [addBlack, addGaussianNoise, addRandomNoise, addSPNoise]
    differentNoiseNames = ["black", "gaussian", "random", "sp"]
    
    # Paths
    pathAnnotations = "./annotations/instances_val2017.json"
    pathImages = "./val2017/"
    pathNewImages = "./noisyImages/"
    
    # Create folder of new images
    if not os.path.exists(pathNewImages):
        os.makedirs(pathNewImages)
    
    # Load COCO annots
    coco = COCO(pathAnnotations)
    
    # Get random image and its annots
    imageId = int(np.random.choice(coco.getImgIds()))
    imageInfo = coco.loadImgs(imageId)[0]
    image = np.array(Image.open(pathImages + imageInfo["file_name"]))
    annotsId = coco.getAnnIds(imageId)
    annots = coco.loadAnns(annotsId)
    
    # Save original
    cv2.imwrite(pathNewImages + imageInfo["file_name"], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # For each object create masks
    for i, annot in enumerate(annots):
        # Segmentation mask
        maskSeg = coco.annToMask(annot) != 1
        # BBox mask
        bbox = annot["bbox"]
        maskBBox = bboxToMask(image, bbox) != 1
        # Get smaller random mask outside and inside the bbox
        maskPartOut = getRandomPartOutsideBox(image, bbox, maskBBox)
        maskPartOut = maskPartOut == 1
        maskPartIn = getRandomPartInsideBox(image, bbox, maskSeg)
        maskPartIn[maskBBox == True] = 1
        maskPartIn = maskPartIn == 1
        
        # Add noise outside bbox
        for j in range(len(differentNoiseFuncs)):
            # Add noise outside of BBox and outside of bbox
            img = image.copy()
            img = differentNoiseFuncs[j](img, maskBBox)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(pathNewImages + imageInfo["file_name"][:-4] + "_bbox_" + differentNoiseNames[j] + "_" + str(i) + ".png", img)
        
        # Add noise outside segmentation
        for j in range(len(differentNoiseFuncs)):
            # Add noise outside of BBox and outside of bbox
            img = image.copy()
            img = differentNoiseFuncs[j](img, maskSeg)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(pathNewImages + imageInfo["file_name"][:-4] + "_seg_" + differentNoiseNames[j] + "_" + str(i) + ".png", img)
    
        # Add noise in random area of outside bbox
        for j in range(len(differentNoiseFuncs)):
            # Add noise outside of BBox and outside of bbox
            img = image.copy()
            img = differentNoiseFuncs[j](img, maskPartOut)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(pathNewImages + imageInfo["file_name"][:-4] + "_bboxAreaOut_" + differentNoiseNames[j] + "_" + str(i) + ".png", img)
        
        # Add noise in random area of inside bbox
        for j in range(len(differentNoiseFuncs)):
            # Add noise outside of BBox and outside of bbox
            img = image.copy()
            img = differentNoiseFuncs[j](img, maskPartIn)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(pathNewImages + imageInfo["file_name"][:-4] + "_bboxAreaIn_" + differentNoiseNames[j] + "_" + str(i) + ".png", img)
        
    print("Done!")