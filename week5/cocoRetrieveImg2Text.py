import numpy as np
import os
from cocoTripletDataset import TripletCOCOdatabase_Img2Text
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
            
            if dbIndex >= i_query*5 and dbIndex < i_query*5 + 5:
                resultQuery.append(1)
            else:
                resultQuery.append(0)
        
        resultsQueries.append(resultQuery)
    
    return np.array(resultsQueries)
if __name__ == "__main__":
    
    # Use FAISS
    useFaiss = False
    metric = "l2"
    
    
    jsonPathCap = "../WEEK4/COCO/captions_val2014.json"
    
    
    # Init database dataset
    databaseImagesPath = "../WEEK4/COCO/val2014/"
    databaseImages = os.listdir(databaseImagesPath)
    max_images = 4000
    
    database_dataset = TripletCOCOdatabase_Img2Text(databaseImagesPath, databaseImages,
                                           jsonPathCap, None, max_images)
    
    databaseImagesFeaturesPath = "features_img2textfast4000__images.txt"
    databaseCaptionsFeaturesPath = "features_img2textfast4000__captions.txt"
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
            retrieval = FAISSretrieval(databaseCaptionFeatures, databaseCaptionFeatures.shape[1])
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
    
    inStr = input("Press Enter to continue, other key to exit...")
    while inStr == "":
        # Show results
        query = np.random.choice(list(range(databaseImagesFeatures.shape[0]))) #695
        print("Id: ", query)
        print("Query image:")
        # Get image
        data, _ = database_dataset[query]
        img, _ = data
        img = np.array(img)
        plt.imshow(img)
        plt.show()
        
        
        print("GT caption:")
        goodCaption = np.argmax(resultList[query, :])
        goodCaption = neighbors[query,goodCaption]
        goodCaptionInd = goodCaption % 5
        goodCaptionImgInd = goodCaption // 5
        data, _ = database_dataset[goodCaptionImgInd]
        _, captions = data
        print(captions[goodCaptionInd])
        
        # Get 5 most close captions
        for i in range(5):
            print(i, ". closest caption:")
            
            neighbor = neighbors[query, i]
            num = neighbor % 5
            neighbor = neighbor // 5
            
            # Get caption
            data, _ = database_dataset[neighbor]
            _, captions = data

            print("Caption: ", captions[num])
            
            print("Correct caption: ", resultList[query, i] == 1)
        
        inStr = input("Press Enter to continue, other key to exit...")
    


    