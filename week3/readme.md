# Week 3

The objective of this week's project was to understand the limitations of Faster R-CNN and Mask R-CNN for object detection and segmentation.

We utilized pre-trained models from the COCO dataset and evaluated their performance using the Out-of-Context dataset. In addition, we investigated the impact of transplanting objects into images with less co-occurrence, as well as cloning objects to see how the model's predictions were affected. Furthermore, we explored examples to understand the impact of feature interference on object detection in the COCO dataset. Finally, we investigated how changing the texture of an object affected the model's results.

## Here are some instructions to run the scripts:
### Install the following dependencies:
* Pytorch
* Pycocotools
* OpenCV
* Matplotlib
* Detectron2
* Numpy
* Sklearn
* Keras
### Scripts:
#### Task a and e:
-   eval_a_e.py : a script that evaluates fastrcnn and maskrcnn on the Out Of Context dataset, a smaller hanpicked dataset (with isolated objects) and 2 other datasets that compromise of the handpicked dataset, but the images have had a style/texture transfer applied
-   style_transfer.py : a script that applies style/texture transfer on a handpicked dataset (with isolated objects) based on this implementation https://github.com/rgeirhos/texture-vs-shape

#### Tasks b and c:
-   make_cooc_matrix.py : a script for the co-occurrence matrix computation. It uses vectorized numpy functions and is much more efficient than the old version (few milliseconds for the val COCO).
-   make_txt.py : creates a list representation of a co-occurrence matrix
-   select_anns.py : a script that gets two category ids (A and B) and returns the images that have A category objects but not B ones
-   transplant.py : by specifying the object to cut out and the image to insert it in with annotation ID, this script returns the combined image and runs an object-detector inference. Finally, it saves the results.

#### Task d:
-   addNoise.py : this script contains functions to add Gaussian, Random, and Salt & Pepper noise to given images in the areas indicated in the masks.
-   createNoisyImage.py : this script takes an image name from the COCO validation set and stores a set of images. For each object in the image, it removes the background (out of the BBox or ROI) and also adds different types of noise.
-   inferenceVisualization.py : this script inferences the pre-trained Faster R-CNN and Mask R-CNN networks on COCO with the indicated images. It also stores the images with the detections made.


### Dataset:
* COCO validation
* Out-of-Context dataset
