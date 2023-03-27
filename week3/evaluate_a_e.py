from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import model_zoo
import os, sys, cv2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.utils.visualizer import Visualizer


ooc_dataset = r"/ghome/mcv/datasets/out_of_context"
original_nontextured_dataset = r"/ghome/group04/week3/datasets/dataset"
hugging_textured_dataset = r"/ghome/group04/week3/datasets/results-hugging"
vgg_textured_dataset = r"/ghome/group04/week3/datasets/results-vgg"


faster_config = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
mask_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

def evaluate(config, dataset, save_folder):
    # Config model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    counter = 0
    for filename in os.listdir(dataset):
        image = cv2.imread(os.path.join(dataset, filename))
        if image is not None:
            outputs = predictor(image)
            
            print(counter)
            print(outputs["instances"].pred_classes)
            print(outputs["instances"].pred_boxes)
            
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # Save the image to a file
            cv2.imwrite(save_folder + r"/output" + str(counter) + ".jpg", out.get_image()[:, :, ::-1])
            counter = counter + 1


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
create_folder("output_faster_ooc")
create_folder("output_mask_ooc")
create_folder("output_dataset_faster")
create_folder("output_hugging_faster")
create_folder("output_vgg_faster")
create_folder("output_dataset_mask")
create_folder("output_hugging_mask")
create_folder("output_vgg_mask")

# Out of context (Task a)
evaluate(faster_config, ooc_dataset, "output_faster_ooc")
evaluate(mask_config, ooc_dataset, "output_mask_ooc")

# Texture/Style transfer (Task e)
evaluate(faster_config, original_nontextured_dataset, "output_dataset_faster")
evaluate(faster_config, hugging_textured_dataset, "output_hugging_faster")
evaluate(faster_config, vgg_textured_dataset, "output_vgg_faster")

evaluate(mask_config, original_nontextured_dataset, "output_dataset_mask")
evaluate(mask_config, hugging_textured_dataset, "output_hugging_mask")
evaluate(mask_config, vgg_textured_dataset, "output_vgg_mask")
