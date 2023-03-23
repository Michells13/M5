# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2

# import some common detectron2 utilities
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer


# Config Faster RCNN
cfgFaster = get_cfg() 
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfgFaster.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfgFaster.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfgFaster.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

# Config Mask RCNN
cfgMask = get_cfg() 
cfgMask.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfgMask.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfgMask.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

# Init eval
predictorFaster = DefaultPredictor(cfgFaster)
predictorMask = DefaultPredictor(cfgMask)


# Dataset path
pathDataset = "./noisyImages/"
pathPreds = "./noisyImagesPred/"

# Create folder of `preds
if not os.path.exists(pathPreds):
    os.makedirs(pathPreds)


# Get predictions of images
for image in os.listdir(pathDataset):
    # Load image
    img = cv2.imread(pathDataset + image)
    
    # Inference models
    outputFaster = predictorFaster(img)
    outputMask = predictorMask(img)
    
    # Visualize results
    vFaster = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfgFaster.DATASETS.TEST[0]), scale=1.2)
    vMask = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfgFaster.DATASETS.TEST[0]), scale=1.2)
    
    # Get instances
    instancesFaster = outputFaster["instances"].to("cpu")
    instancesMask = outputMask["instances"].to("cpu")

    # Plot
    outFaster = vFaster.draw_instance_predictions(instancesFaster)
    outMask = vMask.draw_instance_predictions(instancesMask)
    
    # Save detections
    cv2.imwrite(pathPreds + image[:-4] + "_faster.png", outFaster.get_image()[:, :, ::-1])
    cv2.imwrite(pathPreds + image[:-4] + "_mask.png", outMask.get_image()[:, :, ::-1])

print("Done!")

