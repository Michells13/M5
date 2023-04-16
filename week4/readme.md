# Week 4

The objective of this week's project was to use deep learning metrics for image retrieval.

We utilized pre-trained models from ResNet to observe the feature vectors that they create for the MIT dataset images. In addition, we investigated the improvement when the network was fine-tuned in a siamese or triplet way. After that, we visualized the representations of MIT split images. Finally, we attempted to perform image retrieval using the objects in the COCO dataset images and the Faster RCNN model.

## Here are some instructions to run the scripts:
### Install the following dependencies:
* Pytorch
* Torchvision
* Pytorch_metric_learning
* Faiss
* Pycocotools
* OpenCV
* Matplotlib
* Numpy
* Sklearn

### Scripts:
#### Task a:
* --

#### Task b:
* `siamese_example.py``: A script used to 

#### Task c:
* ``triplets.py``: A script used to train and evaluate a resnet18 model in a triplet architecture plus hyperparameters search with optuna.  

#### Task d:
* ``vizualizeResults.py``: A script to vizualize models with TSNE and UMAP

#### Task e:
* ``trainCOCOtriplet.py``: a script used to train the networks (method 1 and 2 explained in the slides) using the triplet margin loss.
* ``cocoTripletGetFeatures.py``: a script used to extract and store the feature vectors of the images using the fine-tuned models.
* ``cocoRetrieve.py``: a script used to evaluate (P@1, P@5, MAP results) and visualize the closest vector images.
### Dataset:
* COCO custom split
* MIT dataset

