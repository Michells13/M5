import faiss
from sklearn.neighbors import KNeighborsClassifier


class FAISSretrieval():
    def __init__(self, features, d):
        self.index = faiss.IndexFlatL2(d)
        self.index.add(features)
    
    def getMostSimilar(self, queryFeatures, k):
        D, I = self.index.search(queryFeatures, k)
        
        return D, I

class KNNretrieval():
    def __init__(self, features, metric, k):
        
        self.knn = KNeighborsClassifier(k, metric = metric, n_jobs= 1)
        self.knn.fit(features, list(range(features.shape[0])))
    
    def getMostSimilar(self, queryFeatures, k = None):
        
    
        # Inference every query
        (dis, neighbors) = self.knn.kneighbors(queryFeatures, return_distance=True)
        
        return dis, neighbors