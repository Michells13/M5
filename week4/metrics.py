import numpy as np


def precisionK(results, k):
    """
    This function computes the precision@k for a query
    giving the positive results

    Parameters
    ----------
    results : numpy array
        Array with 1 when retrive was positive, 0 otherwise.
    k : int
        k value to compute.

    Returns
    -------
    float
        p@k value.

    """
    
    return np.sum(results[:k])/k

def mPrecisionK(listResults, k):
    """
    This function computes the mean precision@k over all the queries.

    Parameters
    ----------
    listResults : numpy array
        For each query (row), 1 if retrieve was positive 0 otherwise.
    k : int
        k value to compute.

    Returns
    -------
    float
        Mean p@k value.

    """
    
    valSum = 0
    
    for i in range(listResults.shape[0]):
        valSum += precisionK(listResults[i,:], k)
    
    return valSum / listResults.shape[0]

def recallK(results, k):
    """
    This function computes the recall@k for a query
    giving the positive results

    Parameters
    ----------
    results : numpy array
        Array with 1 when retrive was positive, 0 otherwise.
    k : int
        k value to compute.

    Returns
    -------
    float
        r@k value.

    """
    
    return np.sum(results[:k])/np.sum(results)

def mRecallK(listResults, k):
    """
    This function computes the mean recall@k over all the queries.

    Parameters
    ----------
    listResults : numpy array
        For each query (row), 1 if retrieve was positive 0 otherwise.
    k : int
        k value to compute.

    Returns
    -------
    float
        Mean r@k value.

    """
    
    
    valSum = 0
    
    for i in range(listResults.shape[0]):
        valSum += recallK(listResults[i,:], k)
    
    return valSum / listResults.shape[0]
    
def averagePrecision(results):
    """
    This function computes the average precision for a query
    giving the positive results

    Parameters
    ----------
    results : numpy array
        Array with 1 when retrive was positive, 0 otherwise.

    Returns
    -------
    float
        ap value.

    """
    
    
    ap = (np.cumsum(results) * results)/(np.array(range(results.shape[0])) + 1)
    
    return np.sum(ap)/np.sum(results)

def MAP(listResults):
    """
    This function computes the mean average previcision over all the queries.

    Parameters
    ----------
    listResults : numpy array
        For each query (row), 1 if retrieve was positive 0 otherwise.

    Returns
    -------
    float
        Mean ap value.

    """
    
    valSum = 0
    
    for i in range(listResults.shape[0]):
        valSum += averagePrecision(listResults[i,:])
    
    return valSum / listResults.shape[0]
    
    