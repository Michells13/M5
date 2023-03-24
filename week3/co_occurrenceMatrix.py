from pycocotools.coco import COCO
from utils import co_ocurrenceOf2classes,coMatriz,co_OcurrenceMatrix
import numpy as np
import matplotlib.pyplot as plt

# LOad annotations

pathToAnnotations='/media/michell/DSet/annotations_trainval2017/annotations/instances_val2017.json'

# get co-occurrence matrix
matrix,matrix2=co_OcurrenceMatrix(pathToAnnotations)

#load matrix
plt.imshow(matrix, cmap='Blues')
# Add x and y axis labels
plt.xlabel('Items')
plt.ylabel('Items')
# Show the plot
plt.show()


