# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from pathlib import Path
from kittiMotsDataset import get_KITTI_MOTS_dataset
from copy import deepcopy

# dataset = r"/ghome/group04/new_split_dataset/"
dataset = Path(r"/home/user/data_tracking_image_2/")
datasetTraining = dataset / r"train/"
datasetValidation = dataset / r"val/"
datasetAnnot = dataset / r"instances/"

out_dir = Path(r"/home/user/MCV-M5-Team04/week2/finetune_out")


# Register dataset
for d in ["train", "val"]:
    DatasetCatalog.register("KITTI_MOTS_" + d, lambda d=d: get_KITTI_MOTS_dataset(dataset + d, datasetAnnot, False))
    MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=["car", "pedestrian"])
    
kitti_mots_metadata = MetadataCatalog.get(str(datasetTraining))
#dataset_dicts = get_KITTI_MOTS_dataset(datasetTraining, datasetAnnot)

# Config model
cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("KITTI_MOTS_train",)
cfg.DATASETS.TEST = ("KITTI_MOTS_val", )
cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0025 # pick a good LR
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 2    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = 200       # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 256)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 2 classes: car and pedestrian
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


cfg.MODEL.WEIGHTS = r"/home/user/MCV-M5-Team04/week2/model_final.pth"
predictor = DefaultPredictor(cfg)

from time import time

max_imgs = 1000
imgs = 0
# for img_path in datasetValidation.rglob(r"*.png"):
for subdir in datasetValidation.iterdir():
    for img_path in subdir.iterdir():
        img = cv2.imread(str(img_path))
        start = time()
        for i in range(30):
            print(i)
            outputs = predictor(img)
        print(f"Time elapsed: {time() - start}")
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out = out.get_image()[:, :, ::-1]
        out_path = str(out_dir / f"{imgs:04}.png")
        
        cv2.imshow("sfa", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # exit()
        
        
        # print(imgs)
        # cv2.imwrite(out_path, out)
        # imgs +=1
        # if imgs >= max_imgs:
        #     exit()


