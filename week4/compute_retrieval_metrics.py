import numpy as np
from torchvision.datasets import ImageFolder
from sklearn.neighbors import KNeighborsClassifier
from metrics import *
import matplotlib.pyplot as plt

# dataset_dir = r"D:\MCV-M5-Team04\MIT_train_val_test\MIT_train_val_test\val"
dataset_dir = r"D:\MCV-M5-Team04\MIT_split\test"
cache_filename = r"output_cache_michell_new2.txt"

dataset = ImageFolder(dataset_dir)
database = np.loadtxt(cache_filename)

# get relevance gt
classes = np.array([sample[1] for sample in dataset])
gt = np.tile(classes, len(classes)).reshape(-1, len(classes))
gt = gt == gt.T


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(database, classes)


distances, neighbors = knn.kneighbors(database, n_neighbors=len(database))
reordered_gt = np.array([gt[i][neighbors[i]] for i in range(len(gt))])


class_precisions = []  # list of lists
class_indices = list(range(len(dataset.classes)))
for c in class_indices:
    class_precisions.append([mPrecisionK(reordered_gt[classes==c], k)
                             for k in range(1, 10)])
class_precisions = np.array(class_precisions).T

# Precision@k
plt.plot(class_precisions)
plt.xlabel("k")
plt.ylabel("Precision")
plt.title("Prec@k")
plt.legend(dataset.classes)
plt.show()

# MAP 
maps = np.array([MAP(reordered_gt[classes==c]) for c in class_indices])
for class_idx, class_name in enumerate(dataset.classes):
    print(f"MAP for class {class_name}:    {maps[class_idx]}")

# Precision-Recall Curve
for c in class_indices:
    p, r = precisionRecall(reordered_gt[classes==c])
    plt.plot(r, p)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(dataset.classes)
plt.show()
