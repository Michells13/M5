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
from kittiMotsDataset import get_KITTI_MOTS_dataset
from detectron2.engine import DefaultTrainer
import wandb

datasetTraining = "KITTI-MOTS/training/image_02/"
datasetValidation = "KITTI-MOTS/testing/image_02/"
datasetAnnot = "KITTI-MOTS/instances/"

# Register dataset
for d in ["training", "testing"]:
    DatasetCatalog.register("KITTI_MOTS_" + d, lambda d=d: get_KITTI_MOTS_dataset("KITTI-MOTS/" + d + "/image_02/", datasetAnnot))
    MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=["car", "pedestrian"])
    
kitti_mots_metadata = MetadataCatalog.get(datasetTraining)
#dataset_dicts = get_KITTI_MOTS_dataset(datasetTraining, datasetAnnot)

# Init wandb
wandb.init(sync_tensorboard=True,
           settings=wandb.Settings(start_method="thread", console="off"))

# Config model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("KITTI_MOTS_training",)
cfg.DATASETS.TEST = ("KITTI_MOTS_testing", )
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 2 classes: car and pedestrian
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

# Init
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# for d in random.sample(dataset_dicts, 3):
#     #d = dataset_dicts[0]
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_mots_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     plt.imshow(out.get_image())
#     plt.show()
