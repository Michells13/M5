import torch
from tqdm import tqdm
import numpy as np


def cache_outputs(loader, model, cache_filename):
    """Create a database for the retrieval.
    
    Write model's outputs for the loader's data into a file located at
    cache_filename.

    Args:
        loader: PyTorc DataLoader representing your dataset
        model: callable, used to produce feature vectors,
            takes (n_samples, in_features), 
            outputs (n_samles, out_features)
        cache_filename: path to a file, into which feature vectors
            will be written

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        f = open(cache_filename, "wb")
        for data, _ in tqdm(loader):
            output = model(data)
            np.savetxt(f, output)
        f.close()


def retrieve(database_filename, query_idx, k, metric_function, sorted=False):
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
    database = np.loadtxt(database_filename)
    query = database[query_idx]
    distances = metric_function(query, database)
    k += 1  # compensate for the query_idx itself
    if sorted:
        top_k = np.argsort(distances)[:k]
    else:  # faster
        top_k = np.argpartition(distances, k)[:k]
    top_k = np.delete(top_k,
                      np.where(top_k == query_idx)[0])  # delete query_idx
    
    return top_k
