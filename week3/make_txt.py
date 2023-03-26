from pycocotools.coco import COCO
import numpy as np


img_dir = r"/home/user/coco2017val/val2017"
annot_path = r"/home/user/coco2017val/annotations/instances_val2017.json"
matrix_path =r"cooccurrence_matrix.npy"
categories_path =r"categories.npy"


matrix = np.load(matrix_path)
categories = np.load(categories_path)
# mark the lower triangle (symmetry) row as -1 to skip it later
matrix[np.tril_indices_from(matrix)] = -1
top_indices = np.argsort(matrix.flatten())
# find index from where lower triangle coocurrances end
relevant_index = np.where(matrix.flatten()[top_indices] != -1)[0][0]
top_indices = np.unravel_index(top_indices[relevant_index:],
                                    matrix.shape)
top_indices = np.column_stack(top_indices)[::-1]
top_categories = categories[top_indices]


coco = COCO(annot_path)
with open('coocurrence.txt', 'a') as f:
    for row, row_cat in zip(top_indices, top_categories):
        name1 = coco.dataset['categories'][row[0]]["name"]
        name2 = coco.dataset['categories'][row[1]]["name"]
        value = matrix[row[0], row[1]]
        line = f"{name1}, {name2}   {row_cat}      {value}\n"
        f.write(str(line))


