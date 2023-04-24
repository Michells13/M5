import numpy as np
import os
from cocoTripletDataset import TripletCOCO_Img2Text
from metrics import mPrecisionK, mRecallK, MAP, precisionRecall
from tqdm import tqdm
from matplotlib import pyplot as plt
from similarRetrieval import FAISSretrieval, KNNretrieval
import time
from sklearn.metrics import PrecisionRecallDisplay


def computePositives(neighbors):
    """
    This function generates the matrix with 1 values when the query and prediction
    are positive and 0 otherwise.

    Parameters
    ----------
    neighbors : numpy array
        Matrix with results of retrieval.

    Returns
    -------
    numpy array
        The results with 1 or 0 values.

    """
    
    resultsQueries = []
    
    for i_query in tqdm(range(neighbors.shape[0])):
        resultQuery = []
        
        for i_db in range(neighbors.shape[1]):
            
            dbIndex = neighbors[i_query, i_db]
            
            # if i_query >= dbIndex*5 and i_query < dbIndex*5 + 5:
            # if i_query >= dbIndex*5 and i_query < dbIndex*5 + 5:
            dbIndex = dbIndex // 5
            if i_query == dbIndex:
                resultQuery.append(1)
            else:
                resultQuery.append(0)
        
        resultsQueries.append(resultQuery)
    
    return np.array(resultsQueries)


if __name__ == "__main__":
    
    # Use FAISS
    useFaiss = False
    metric = "l2"
    
    
    jsonPath = r"D:\coco\annotations_trainval2014\annotations\person_keypoints_val2014.json"
    jsonPathCap = r"D:\coco\annotations_trainval2014\annotations\captions_val2014.json"
    
    
    # Init database dataset
    sections = ["val", "test"]
    databaseImagesPath = r"D:\coco\val2014\val2014"
    databaseImages = os.listdir(databaseImagesPath)
    
    # database_dataset = TripletCOCO_Img2Text(databaseImagesPath, databaseImages, jsonPathCap,
    #                                        jsonPath, None, sections)

    # swap img and txt
    databaseImagesFeaturesPath = r"D:\img2text\img_final.txt"
    databaseCaptionsFeaturesPath = r"D:\img2text\text_final.txt"
    databaseImagesFeatures = np.loadtxt(databaseImagesFeaturesPath)
    databaseImagesFeatures = databaseImagesFeatures.astype(np.float32)
    databaseCaptionFeatures = np.loadtxt(databaseCaptionsFeaturesPath)
    databaseCaptionFeatures = databaseCaptionFeatures.astype(np.float32)

    print(databaseImagesFeaturesPath)
    print(databaseCaptionsFeaturesPath)
    
    # Init similarity method
    times = []
    repeatN = 1
    for i in range(repeatN):
        start = time.time()
        if useFaiss:
            retrieval = FAISSretrieval(databaseImagesFeatures, databaseImagesFeatures.shape[1])
        else:
            retrieval = KNNretrieval(databaseCaptionFeatures, metric, databaseCaptionFeatures.shape[0])
        stop = time.time()
        times.append(stop-start)
    print("Time fit(s): ", np.median(np.array(times)))
    
    # Inference every query
    times = []
    for i in range(repeatN):
        start = time.time()
        (dis, neighbors) = retrieval.getMostSimilar(databaseImagesFeatures, databaseCaptionFeatures.shape[0])
        stop = time.time()
        times.append(stop-start)
    print("Time retrieval(s): ", np.median(np.array(times)))
    
    # Compute positive and negative values
    resultList = computePositives(neighbors)
    
    # Compute metrics
    print("P@1: ", mPrecisionK(resultList, 1))
    print("P@5: ", mPrecisionK(resultList, 5))
    print("R@1: ", mRecallK(resultList, 1))
    print("R@5: ", mRecallK(resultList, 5))
    print("MAP: ", MAP(resultList))
    
    # Compute and plot precision recall curve
    precision, recall = precisionRecall(resultList) 
    display = PrecisionRecallDisplay(precision, recall)
    disp = display.plot()
    _ = disp.ax_.set_title("COCO retrieval precision-recall curve")
    plt.show()
    
    # inStr = input("Press Enter to continue, other key to exit...")
    # while inStr == "":
    #     # Show results
    #     query = np.random.choice(list(range(databaseImagesFeatures.shape[0]))) #695
    #     print("Id: ", query)
    #     num = 0  # np.random.randint(low=0, high=5)
    #     queryIndex = query

    #     # Get img
    #     img, _ = database_dataset[queryIndex]
    #     print("Query image: ", img.shape)
        
    #     # Get 5 most close captions
    #     for i in range(5):
    #         print(i, ". closest caption:")
            
    #         neighbor = neighbors[query, i]
            
    #         # Get caption
    #         img, _ = database_dataset[neighbor]
    #         img = np.array(img)
    #         plt.imshow(img)
    #         plt.show()
            
    #         print("Correct image: ", resultList[query, i] == 1)
        
    #     inStr = input("Press Enter to continue, other key to exit...")
    


    