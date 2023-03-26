

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2
import numpy as np


img_dir = r"/home/user/coco2017val/val2017"
annot_path = r"/home/user/coco2017val/annotations/instances_val2017.json"
out_dir = r"/home/user/semantic_transplant/out"


def get_img(ann, img_dir, coco):
    img = coco.loadImgs(ann["image_id"])[0]
    img_file_name = img_dir + "/" + img["file_name"]
    img = cv2.imread(img_file_name)
    
    return img
    
def get_binary_mask(ann, coco):
    ann_segm = ann['segmentation']
    img = coco.loadImgs(ann["image_id"])[0]

    img_height = img["height"]
    img_width = img["width"]
    
    
    # TODO: find out what are the cases
    # case 1
    if type(ann_segm) is list:
        rles = maskUtils.frPyObjects(ann_segm , img_height , img_width)
        rle = maskUtils.merge(rles)
    # case 2
    elif type(ann_segm['counts']) is list:
        rle = maskUtils.frPyObjects(ann_segm , img_height , img_width)
    # case 3
    else:
        rle = ann_segm['segmentation']

    # again, use pycocotools to get the binary mask
    ann_mask = maskUtils.decode(rle)
    
    return ann_mask


def cut_out(img, mask):
    foreground = cv2.bitwise_and(img, img, mask=mask)
    # crop used part
    used = np.where(mask)
    used_us = used[0]
    used_vs = used[1]
    cropped = foreground[
        np.min(used_us):np.max(used_us),
        np.min(used_vs):np.max(used_vs),
    ]
    return cropped




def transplant(src_ann, dst_ann, img_dir, coco, offset=(0, 0)):
    # TODO: add random movement params
    """Transplant instances from source to destination
    Inputs:
        ...
    Returns:
        An image of dst_img's shape, that has an instance from the
        src_img pasted in
    
    """
    
    mask = get_binary_mask(src_ann, coco)
    src_img = get_img(src_ann, img_dir, coco)
    dst_img = get_img(dst_ann, img_dir, coco)
    
    # TODO: align mask and img with dst_img
    
    used = np.where(mask)
    u_min = np.min(used[0])
    v_min = np.min(used[1])
    for u, v in zip(*used):
        u_dst = u - u_min + offset[0]
        v_dst = v - v_min + offset[1]
        dst_img[u_dst, v_dst] = src_img[u, v]
    
    return dst_img

    # used_us = used[0]
    # used_vs = used[1]
    
    # instance = cut_out(img, mask)  # TODO: cut_out is probably redundant
    # TODO: check if instance smaller (can fit) than dst_img
    
    
    pass


coco = COCO(annot_path)
# img_id = 397133  # just for test
# anns_ids = coco_info.getAnnIds(img_id, iscrowd=False)
# anns = coco_info.loadAnns(anns_ids)
# src_ann = anns[2]



# img_id = 397133  # just for test
# anns_ids = coco_info.getAnnIds(img_id, iscrowd=False)
# anns = coco_info.loadAnns(anns_ids)
# dst_ann = anns[3]
 
offset = (40, 300)
annId1 = 30516
annId2 = 30516
src_ann = coco.loadAnns(annId1)[0]
dst_ann = coco.loadAnns(annId2)[0]
combined_img = transplant(src_ann, dst_ann, img_dir, coco, offset)

src_img = get_img(src_ann, img_dir, coco)
dst_img = get_img(dst_ann, img_dir, coco)





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

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


def draw_predictions(img):
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = out.get_image()[:, :, ::-1]
    return img


src_img = draw_predictions(src_img)
dst_img = draw_predictions(dst_img)
combined_img = draw_predictions(combined_img)

import os
out_subdir = out_dir + f"/{annId1}_{annId2}"
if not os.path.exists(out_subdir):
    os.makedirs(out_subdir)

cv2.imwrite(out_subdir + f"/src_{annId1}.png", src_img)
cv2.imwrite(out_subdir + f"/dst_{annId2}.png", dst_img)
cv2.imwrite(out_subdir + f"/combined_{annId1}_{annId2}.png", combined_img)


    