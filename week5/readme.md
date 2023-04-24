# Week 5


The objective of this week's project was to use deep learning techniques for cross-modal retrieval tasks.

We used Faster R-CNN as the image feature extractor, as explained in the week 4 slides, and computed the average of the proposal features. For text feature extraction, we utilized FastText and BERT models. In addition, to achieve a cross-related feature space, we defined two small fully connected models to obtain a shared feature space of size 128 from both the images and text captions.

## Here are some instructions to run the scripts:
### Install the following dependencies:
* Pytorch
* Torchvision
* Faiss
* Pycocotools
* OpenCV
* Matplotlib
* Numpy
* Sklearn
* Transformers
* FastText

### Scripts:

* `cocoTripletDataset.py`: script where the custom dataset classes are defined. In the `TripletCOCO_Img2Text` class, the element is a triplet with an anchor image, a positive caption, and a negative caption. In the `TripletCOCO_Text2Img` class, the element is a triplet composed of an anchor caption and positive and negative images.
#### Task a:
* `img2text.py`: a script to train img2text networks with fastText as a text model.
* `make_database_img.py`: a scirpt that caches image representation in a common feature space.
* `make_database_text.py`: a scirpt that caches text representation in a common feature space.
* `main_retrieve.py`: a script to qualitatively evaluate img2text retriaval network(s): show query image and k closest captions.
* `cocoRetrieveImg2Text_v2.py`: a script to calculate metrics for learned common img/text representations.

#### Task b:
*   `text2img.py`: a script used to train the networks (only the fully connected ones) using the triplet margin loss in the COCO official train set, with FastText model as the text feature extractor for text-to-image retrieval.
* `cocoTripletGetFeatures_captions.py`: a script used to extract and store the feature vectors of the images and the captions using the fine-tuned models with FastText text feature extractor.
* `cocoRetrieveText2Image.py`: a script used to evaluate (P@1, P@5, MAP results, precision-recall curve) and visualize the closest vector images per caption.
#### Task c:
*   `img2textBert.py`: a script used to train the networks (only the fully connected ones) using the triplet margin loss in the COCO official train set, with BERT model as the text feature extractor for image-to-text retrieval.
*   `text2imgBert.py`: a script used to train the networks with BERT model as the text feature extractor for text-to-image retrieval.
* `cocoTripletGetFeatures_captions_bert.py`: a script used to extract and store the feature vectors of the images and the captions using the fine-tuned models with BERT as the text feature extractor.
* The results were computed using the scripts of Task a and b.

### Dataset:
* COCO official splits
