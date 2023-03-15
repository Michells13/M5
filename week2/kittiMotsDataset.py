import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
import os
import cv2
import PIL.Image as Image
import numpy as np
import random
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt

def read_png_KITTI_MOTS(png_dir):
    """
    This functions reads KITTI MOTS annotation png file and returns the
    list of objects in COCO format

    Parameters
    ----------
    png_dir : str
        Path of the png image.

    Returns
    -------
    objs : list
        List of objects in the image in COCO format.

    """
    
    # Create list
    objs = []
    # Read
    img = np.array(Image.open(png_dir))
    # Get ids
    obj_ids = np.unique(img)
    # Remove background and ignore labels
    obj_ids = np.setdiff1d(obj_ids, [0, 10000])
    
    # Read each object
    for obj in obj_ids:
        
        img_obj = img.copy()
        # Create object image
        img_obj[img == obj] = 1
        img_obj[img != obj] = 0
        img_obj = img_obj.astype(np.uint8)
        
        coords = np.argwhere(img_obj == 1)
        
        # Get BBox
        xmin = np.min(coords[:,1])
        ymin = np.min(coords[:,0])
        xmax = np.max(coords[:,1])
        ymax = np.max(coords[:,0])
        
        mode = detectron2.structures.BoxMode.XYXY_ABS
        
        # Class id
        class_id = obj // 1000 
        # To be 0, 1 (COCO format)
        class_id = class_id - 1
        
        # Polygon
        contours, _ = cv2.findContours(img_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [list(np.squeeze(contour).flatten().astype(float)) for contour in contours]
        
        # Insert in dict
        obj_dict = {}
        obj_dict["bbox"] = [xmin, ymin, xmax, ymax]
        obj_dict["bbox_mode"] = mode
        obj_dict["category_id"] = class_id
        obj_dict["segmentation"] = contours
        
        objs.append(obj_dict)
    
    return objs


def get_KITTI_MOTS_dataset(img_dir, annot_dir):
    """
    This functions generates the dataset in COCO format given the path of the
    KITTI-MOTS dataset images and annotations

    Parameters
    ----------
    img_dir : TYPE
        DESCRIPTION.
    annot_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    dataset_dicts : TYPE
        DESCRIPTION.

    """
    
    dataset_dicts = []
    

    # Get sequences
    seqs = os.listdir(img_dir)
    for seq in seqs:
        # Get images
        images = os.listdir(os.path.join(img_dir, seq))
        for image in images:
            record = {}
            
            filename = os.path.join(img_dir, seq, image)
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = seq + "_" + image[:-4]
            record["height"] = height
            record["width"] = width
            
            # Read annotation PNG
            annotFile = os.path.join(annot_dir, seq, image)
            objs = read_png_KITTI_MOTS(annotFile)
            
            record["annotations"] = objs
                
            dataset_dicts.append(record)
            
    return dataset_dicts


# datasetTraining = "KITTI-MOTS/training/image_02/"
# datasetValidation = "KITTI-MOTS/testing/image_02/"
# datasetAnnot = "KITTI-MOTS/instances/"
# for d in ["training", "testing"]:
#     DatasetCatalog.register("KITTI_MOTS_" + d, lambda d=d: get_KITTI_MOTS_dataset("KITTI-MOTS/" + d + "/image_02/", datasetAnnot))
#     MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=["car", "pedestrian"])
    
# kitti_mots_metadata = MetadataCatalog.get(datasetTraining)
# dataset_dicts = get_KITTI_MOTS_dataset(datasetTraining, datasetAnnot)
# for d in random.sample(dataset_dicts, 3):
#     #d = dataset_dicts[0]
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_mots_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     plt.imshow(out.get_image())
#     plt.show()
