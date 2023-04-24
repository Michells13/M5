import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import time
from cocoTripletDataset import TripletCOCO_Img2Text
import os


def retrieve(database_filename, query_database_filename, query_idx, k, metric_function, sorted=False, max_rows=None):
    """ Perfrom retrieval task.
    
    Return k indices that access database samples most similar to
    a samle accessible by query_idx. Similarity is estimated using
    metric_fucntion.
    
    Args:
        database_filename: .txt file where each row reperesents
            a database sample via some feature vector
        query_idx: the index of an element for which to look
            similar samples
        k: number of most similar elements' indices to return
        metric_function: function used to measure similarity between
            feature vectors: takes query (shape is (n_features,)) and
            database (shape is (n_samples, n_features)), outputs
            distances (shape is (n_samples))
        sorted: defines the NumPy function to get most similari indices.
            If sorted=False (default), then np.argpartition is used. It
            returns unsorted top-k indices. If sorted=True, argsort is
            used. It returns sorted indices, but slower.
    """
    database = np.loadtxt(database_filename, max_rows=max_rows)
    query_database = np.loadtxt(query_database_filename, max_rows=max_rows)
    query = query_database[query_idx]
    distances = metric_function(query, database)
    k += 1  # compensate for the query_idx itself
    if sorted:
        top_k = np.argsort(distances)[:k]
    else:  # faster
        top_k = np.argpartition(distances, k)[:k]
    top_k = np.delete(top_k,
                      np.where(top_k == query_idx)[0])  # delete query_idx
    
    return top_k





database_cache_filename = r"D:\img2text\text_final2.txt"
query_cache_filename = r"D:\img2text\img_final.txt"

# create a metric function
def l2(a, b):
    diff = a - b
    square = np.power(diff, 2)
    sq_sum = np.sum(square, axis=-1)
    return sq_sum

# create a metric function
def l2_norm(a, b):
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b, axis=1).reshape((-1, 1))) / np.std(b, axis=1).reshape((-1, 1))
    diff = a - b
    square = np.power(diff, 2)
    sq_sum = np.sum(square, axis=-1)
    return sq_sum









# see actual retrieves
val_jsonPath = r"D:\coco\annotations_trainval2014\annotations\person_keypoints_val2014.json"
val_jsonPathCap= r"D:\coco\annotations_trainval2014\annotations\captions_val2014.json"
val_trainImagesPath = r"D:\coco\val2014\val2014"

val_dataset = TripletCOCO_Img2Text(val_trainImagesPath, os.listdir(val_trainImagesPath), val_jsonPath, val_jsonPathCap, None, random_positive=False)
pass

#NOTE: add all captions
for query_idx in range(300):
    top_k = retrieve(database_cache_filename, query_cache_filename, query_idx, 5, l2, sorted=True, max_rows=5000)
    # print retrieved texts
    for idx in top_k:
        print(val_dataset[idx][1])
    print("\n\n\n\n\n")
    # show query image
    plt.imshow(val_dataset[query_idx][0])
    plt.show()