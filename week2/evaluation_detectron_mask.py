import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from kittiMotsDataset import get_KITTI_MOTS_dataset
import wandb

# Init wandb
run = wandb.init(sync_tensorboard=True,
               settings=wandb.Settings(start_method="thread", console="off"),
               project = "mask_rcnn")
wandb.run.name = "mask_rcnn"

dataset = "/media/michell/DSet/split_dataset/"
datasetTraining = dataset + "train/"    
datasetValidation = dataset + "val/"
datasetTest = dataset + "test/"
datasetAnnot = dataset + "instances/"
# Get COCO classes
classes =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush']
# Register dataset
for d in ["train", "val"]:
    try:
        DatasetCatalog.get("KITTI_MOTS_" + d)
    except KeyError:
        DatasetCatalog.register("KITTI_MOTS_" + d, lambda d=d: get_KITTI_MOTS_dataset(dataset + d, datasetAnnot))
        MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=classes)
# Load configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("KITTI_MOTS_train",)
cfg.DATASETS.TEST = ("KITTI_MOTS_val",)
cfg.DATALOADER.NUM_WORKERS = len(classes)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# Create trainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

# Create evaluator
evaluator = COCOEvaluator("KITTI_MOTS_val", cfg, False, output_dir="./output/")

# Evaluate model
val_loader = trainer.build_test_loader(cfg, "KITTI_MOTS_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

# Print evaluation results
metrics = evaluator.evaluate()
print(metrics)
wandb.finish()
