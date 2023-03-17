from kittiMotsDataset import get_KITTI_MOTS_dataset
import numpy as np
from sklearn.cluster import KMeans

def get_anchor_ratios_sizes(dataset, num_anchors):
    """
    This function given a dataset and a number of anchors it returns the best anchor box sizes and
    aspect ratios for the dataset

    Parameters
    ----------
    dataset : list(dict)
        Dataset in COCO format.
    num_anchors : int
        Number of anchor boxes.

    Returns
    -------
    anchor_ratios : list
        Estimated best anchor box aspect ratios.
    anchor_sizes : list
        Estimated best anchor box sizes.

    """
    widths = []
    heights = []
    
    for img in dataset:
        for obj in img["annotations"]:
            # Extract width and height from bounding boxes
            widths.append(obj["bbox"][2] - obj["bbox"][0])
            heights.append(obj["bbox"][3] - obj["bbox"][1])

    heights = np.array(heights)
    widths = np.array(widths)
    
    # Compute aspect ratios
    aspect_ratios = widths / heights

    # Concatenate width and height arrays
    sizes = np.stack([widths, heights], axis=1)

    # Use k-means clustering to find anchor box sizes
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(sizes)
    anchor_sizes = kmeans.cluster_centers_
    
    # Obtain sizes and ratios
    anchor_ratios = anchor_sizes[:,0] / anchor_sizes[:,1]
    anchor_sizes = anchor_sizes[:, 1]
    
    # Sort
    anchor_ratios = np.sort(anchor_ratios)
    anchor_sizes = np.sort(anchor_sizes)

    return anchor_ratios, anchor_sizes

if __name__ == '__main__':
    dataset = "/ghome/group04/new_split_dataset/"
    datasetAnnot = dataset + "instances/"
    datasetL = get_KITTI_MOTS_dataset(dataset + "train/", datasetAnnot)
    anchor_ratios, anchor_sizes = get_anchor_ratios_sizes(datasetL, 5)
    print("Best anchor box ratios: ", anchor_ratios)
    print("Best anchor box sizes: ", anchor_sizes)