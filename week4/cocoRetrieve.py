import numpy as np
import os
from cocoTripletDataset import TripletCOCOdatabase, firstStrategy
from sklearn.neighbors import KNeighborsClassifier
from metrics import mPrecisionK, mRecallK, MAP
from tqdm import tqdm

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
    jsonPath = "./COCO/mcv_image_retrieval_annotations.json"
    
    # Init query dataset
    querySection = "val"
    queryImagesPath = "./COCO/val2014/"
    queryImages = os.listdir(queryImagesPath)
    query_dataset = TripletCOCOdatabase(queryImagesPath, queryImages, jsonPath,
                                           None, querySection)
    queryFeaturesPath = "trained_Mask_backbone_lr1e-3_2_val_pool.txt"
    queryFeatures = np.loadtxt(queryFeaturesPath)
    
    # Init database dataset
    databaseSection = "database"
    databaseImagesPath = "./COCO/train2014/"
    databaseImages = os.listdir(databaseImagesPath)
    database_dataset = TripletCOCOdatabase(databaseImagesPath, databaseImages, jsonPath,
                                           None, databaseSection)
    
    databaseFeaturesPath = "trained_Mask_backbone_lr1e-3_2_database_pool.txt"
    databaseFeatures = np.loadtxt(databaseFeaturesPath)

    print(databaseFeaturesPath)
    print(queryFeaturesPath)

    # Init KNN
    metric = "l2"
    knn = KNeighborsClassifier(databaseFeatures.shape[0], metric = metric)
    knn.fit(databaseFeatures, list(range(databaseFeatures.shape[0])))
    
    # Inference every query
    neighbors = knn.kneighbors(queryFeatures, return_distance=False)
    
    # Compute positive and negative values
    resultList = computePositives(neighbors, database_dataset, query_dataset)
    
    # Compute metrics
    print("P@1: ", mPrecisionK(resultList, 1))
    print("P@5: ", mPrecisionK(resultList, 5))
    print("R@1: ", mRecallK(resultList, 1))
    print("R@5: ", mRecallK(resultList, 5))
    print("MAP: ", MAP(resultList))
    
    