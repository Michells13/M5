import numpy as np
from retrieve_utils import retrieve
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

cache_filename = r"output_cache.txt"

# create a metric function
def l2(a, b):
    diff = a - b
    square = np.power(diff, 2)
    sq_sum = np.sum(square, axis=-1)
    return sq_sum

query_idx = 100
top_k = retrieve(cache_filename, query_idx, 5, l2, sorted=True)


# show actual images
dataset_dir = r"/home/user/MIT_train_val_test/val"
dataset = ImageFolder(dataset_dir)
# show query image
plt.imshow(dataset[query_idx][0])
plt.show()
for idx in top_k:
    plt.imshow(dataset[idx][0])
    plt.show()
