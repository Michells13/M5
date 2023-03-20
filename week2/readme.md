# Week 2
The objective of this week's project was to understand how Detectron2 works and use Faster RCNN and Mask RCNN for object detection and object segmentation. 
We used pretrained models from the COCO dataset and evaluated them using the KITTI-MOTS dataset. Finally, we fine-tuned the models using the same dataset, 
evaluated and visualized the results.
## Here are some instructions to run the scripts:
### Install the following dependencies:
* Pytorch
* Matplotlib
* Detectron2
* Wandb
* Numpy
* Sklearn
### Scripts:
* kittiMotsDataset.py	: Get the KITTI-MOTS dataset and create using the COCO format the custom dataset for detectron2.
* fineTuneMaskRCNN.py	: Train the Mask RCNN model using detectron2 and logged by WnB.
* pretrained_evaluation.py	: Evaluate the pretrained models using KITTI-MOTS dataset.
* pretrained_inference.py	: Visualize the pretrained model predictions.
* test_speed.py	: Measure the time of inference of different models.
* utils.py	: Obtain the best anchor box size and aspect ratios using the training bounding boxes.
* compare_imdirs.py    : Simultaneously, in two show contents of image folders. Used to find interesting distinctions in model behaviour.
* fineTuneFasterRCNN_cluster.py    : Train the Faster RCNN model using detectron2 and logged by WnB
* vis_finetuned.py    : save N prediction visualizations of model. Used with compare_imdirs.py to find meaningful differences in model behaviour.
* vis_orig.py    : same as above, but for an untrained network.
### Dataset:
* KITTI-MOTS
