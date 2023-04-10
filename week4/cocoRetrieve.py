import numpy as np
import os
from cocoTripletDataset import TripletCOCOdatabase, firstStrategy
from metrics import mPrecisionK, mRecallK, MAP
from tqdm import tqdm
from matplotlib import pyplot as plt
from similarRetrieval import FAISSretrieval, KNNretrieval

classes =  ["", 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush']

def computePositives(neighbors, databaseDataset, queryDataset):
    """
    This function generates the matrix with 1 values when the query and prediction
    are positive and 0 otherwise.

    Parameters
    ----------
    neighbors : numpy array
        Matrix with results of retrieval.
    databaseDataset : dataset
        Database dataset.
    queryDataset : dataset
        Query dataset.

    Returns
    -------
    numpy array
        The results with 1 or 0 values.

    """
    
    resultsQueries = []
    
    for i_query in tqdm(range(neighbors.shape[0])):
        resultQuery = []
        
        queryObjs = queryDataset.getObjs(i_query)
        
        for i_db in range(neighbors.shape[1]):
            
            dbIndex = neighbors[i_query, i_db]
            
            dbObjs = databaseDataset.getObjs(dbIndex)
            
            if firstStrategy(dbObjs, queryObjs): 
                resultQuery.append(1)
            else:
                resultQuery.append(0)
        
        resultsQueries.append(resultQuery)
    
    return np.array(resultsQueries)
if __name__ == "__main__":
    
    # Use FAISS
    useFaiss = False
    # Use all image objects
    useAllObjects = False
    metric = "l1"
    
    
    jsonPath = "./COCO/mcv_image_retrieval_annotations.json"
    allLabelsTrain = "./COCO/instances_train2014.json"
    allLabelsTest = "./COCO/instances_val2014.json"
    
    # Init query dataset
    querySection = "test"
    queryImagesPath = "./COCO/val2014/"
    queryImages = os.listdir(queryImagesPath)
    
    if not useAllObjects:
        allLabelsTest = None
    query_dataset = TripletCOCOdatabase(queryImagesPath, queryImages, jsonPath,
                                           None, querySection, allLabelsTest)
    
    queryFeaturesPath = "trained_Mask_backbone_5_epoch_1e-5_margin5_test.txt"
    queryFeatures = np.loadtxt(queryFeaturesPath)
    queryFeatures = queryFeatures.astype(np.float32)
    
    # Init database dataset
    databaseSection = "database"
    databaseImagesPath = "./COCO/train2014/"
    databaseImages = os.listdir(databaseImagesPath)
    if not useAllObjects:
        allLabelsTrain = None
    database_dataset = TripletCOCOdatabase(databaseImagesPath, databaseImages, jsonPath,
                                           None, databaseSection, allLabelsTrain)
    
    databaseFeaturesPath = "trained_Mask_backbone_5_epoch_1e-5_margin5_database.txt"
    databaseFeatures = np.loadtxt(databaseFeaturesPath)
    databaseFeatures = databaseFeatures.astype(np.float32)

    print(databaseFeaturesPath)
    print(queryFeaturesPath)
    
    # Init similarity method
    if useFaiss:
        retrieval = FAISSretrieval(databaseFeatures, databaseFeatures.shape[1])
    else:
        retrieval = KNNretrieval(databaseFeatures, metric, queryFeatures.shape[0])
    
    # Inference every query
    (dis, neighbors) = retrieval.getMostSimilar(queryFeatures, queryFeatures.shape[0])
    
    # Compute positive and negative values
    resultList = computePositives(neighbors, database_dataset, query_dataset)
    
    # Compute metrics
    print("P@1: ", mPrecisionK(resultList, 1))
    print("P@5: ", mPrecisionK(resultList, 5))
    print("R@1: ", mRecallK(resultList, 1))
    print("R@5: ", mRecallK(resultList, 5))
    print("MAP: ", MAP(resultList))
    
    inStr = input("Press Enter to continue, other key to exit...")
    while inStr == "":
        # Show results
        query = np.random.choice(list(range(queryFeatures.shape[0])))
        print("Query image:")
        # Get image
        img, _ = query_dataset[query]
        img = np.array(img)
        plt.imshow(img)
        plt.show()
        # Get values
        objIds = query_dataset.getObjs(query)
        objStr = [classes[int(i)] for i in objIds]
        print("Objects: ", objStr)
        
        # Get 5 most close images
        for i in range(5):
            print(i, ". closest image:")
            
            neighbor = neighbors[query, i]
            
            # Get image
            img,_ = database_dataset[neighbor]
            img = np.array(img)
            plt.imshow(img)
            plt.show()
            # Get values
            objIds = database_dataset.getObjs(neighbor)
            objStr = [classes[int(i)] for i in objIds]
            print("Objects: ", objStr)
        
        inStr = input("Press Enter to continue, other key to exit...")
    


    