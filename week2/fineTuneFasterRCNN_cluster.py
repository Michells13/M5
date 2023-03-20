from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import model_zoo
import os
from kittiMotsDataset import get_KITTI_MOTS_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
import wandb
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset

dataset = r"/ghome/group04/new_split_dataset/"
datasetTraining = dataset + r"train/"
datasetValidation = dataset + r"val/"
datasetAnnot = dataset + r"instances/"

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        
        evaluator_list = [coco_evaluator]
        
        return DatasetEvaluators(evaluator_list)


# Register dataset
for d in ["train", "val"]:
    DatasetCatalog.register("KITTI_MOTS_" + d, lambda d=d: get_KITTI_MOTS_dataset(dataset + d, datasetAnnot, False))
    MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=["car", "pedestrian"])
    
kitti_mots_metadata = MetadataCatalog.get(datasetTraining)
#dataset_dicts = get_KITTI_MOTS_dataset(datasetTraining, datasetAnnot)

# Init wandb
run = wandb.init(sync_tensorboard=True,
               settings=wandb.Settings(start_method="thread", console="off"), 
               project = "detectron2Eval")
wandb.run.name = "Fine_tune_FasterRCNN"

# Config model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("KITTI_MOTS_train",)
cfg.DATASETS.TEST = ("KITTI_MOTS_val", )
cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0025 # pick a good LR
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = 200       # do not decay learning rate
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 256)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 2 classes: car and pedestrian
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


# Init training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# Eval val set
print("Final final validation results: ")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# Create predictor
predictor = DefaultPredictor(cfg)
#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("KITTI_MOTS_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_val")
# Final evaluation
print(inference_on_dataset(trainer.model, val_loader, evaluator))

wandb.finish()
