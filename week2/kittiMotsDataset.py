from detectron2.structures import BoxMode
import os
import cv2
import PIL.Image as Image
import numpy as np

def read_png_KITTI_MOTS(png_dir, pretrained):
    """
    This functions reads KITTI MOTS annotation png file and returns the
    list of objects in COCO format

    Parameters
    ----------
    png_dir : str
        Path of the png image.

    pretrained : bool
        True if the pretrained model with coco dataset will be used (will not be trained)
        
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
        
        mode = BoxMode.XYXY_ABS
        
        # Class id
        class_id = obj // 1000 
        if pretrained:
            # Get COCO labels id
            # Car
            if class_id == 1:
                class_id = 3
            # Person
            elif class_id == 2:
                class_id = 1
        else:
            # To be 0, 1 (COCO format)
            class_id = class_id - 1
        
        # Polygon
        contours, _ = cv2.findContours(img_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [list(np.squeeze(contour).flatten().astype(float)) for contour in contours if len(contour)>3]
        
        if len(contours) != 0:
        
            # Insert in dict
            obj_dict = {}
            obj_dict["bbox"] = [xmin, ymin, xmax, ymax]
            obj_dict["bbox_mode"] = mode
            obj_dict["category_id"] = class_id
            obj_dict["segmentation"] = contours
            
            objs.append(obj_dict)
    
    return objs


def get_KITTI_MOTS_dataset(img_dir, annot_dir, pretrained = True):
    """
    This functions generates the dataset in COCO format given the path of the
    KITTI-MOTS dataset images and annotations

    Parameters
    ----------
    img_dir : str
        Images directory.
    annot_dir : str
        Annotations (pngs) directory.
    pretrained : bool
        True if the pretrained model with coco dataset will be used (will not be trained)

    Returns
    -------
    dataset_dicts : list(dict)
        Annotations in coco format.

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
            objs = read_png_KITTI_MOTS(annotFile, pretrained)
            
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
