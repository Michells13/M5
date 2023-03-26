from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm


def copmute_cooccurrence_matrix(coco):
    """Compute a coocurrence matrix for a given coco dataset.
    
    Args:
        coco: a pycocotools' COCO object
    Returns:
        matrix: a NumPy array, symmetric matrix, in which each row and
            column correspond to a category index, and represents how
            many times annotations of these categories occured on the
            same image. Note that index is not necessarily equal to
            a category ID -- see idx2cat for details.
        categories: a NumPy array, containing categories present in matrix.
            It is useful, as categories are not necessarily
            sequentially, and continuously present in a dataset.
    """
    categories = np.array([cat["id"]
                           for cat in coco.loadCats(coco.getCatIds())])
    matrix = np.zeros((len(categories), len(categories)), dtype=np.int32)
    for imgId in tqdm(coco.getImgIds()):
        annIds = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(annIds)
        if not len(anns) > 0:  # for unannotated images
            continue
        catIds = np.array([ann["category_id"] for ann in anns])
        cat_indices = np.argwhere(np.isin(categories, catIds)).flatten()
        matrix[cat_indices[:, None], cat_indices] += 1
    
    return matrix, categories


if __name__ == "__main__":
    annot_path = r"/home/user/coco2017val/annotations/instances_val2017.json"
    matrix_output_path = r"cooccurrence_matrix.npy"
    categories_output_path = r"categories.npy"
    coco = COCO(annot_path)
    matrix, categories = copmute_cooccurrence_matrix(coco)
    np.save(matrix_output_path, matrix)
    np.save(categories_output_path, categories)
