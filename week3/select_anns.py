"""Select images with class A present, but not class B"""


from pycocotools.coco import COCO
import numpy as np


img_dir = r"/home/user/coco2017val/val2017"
annot_path = r"/home/user/coco2017val/annotations/instances_val2017.json"

cat_A = 72
cat_B = 7

coco = COCO(annot_path)

imgIds = coco.getImgIds(catIds=cat_A)
imgs = coco.loadImgs(imgIds)
for imgId in imgIds:
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    img_catsIds = [ann["category_id"] for ann in anns]
    if not np.isin(cat_B, img_catsIds):
        for ann in anns:
            if ann["category_id"] == cat_A:
                print(ann["id"])
        #         break
        # else:
        #     continue
        # break

print("\n")
print("\n")
print("\n")
print("\n")


imgIds = coco.getImgIds(catIds=cat_B)
imgs = coco.loadImgs(imgIds)
for imgId in imgIds:
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    img_catsIds = [ann["category_id"] for ann in anns]
    if not np.isin(cat_A, img_catsIds):
        for ann in anns:
            if ann["category_id"] == cat_B:
                # print(ann["id"])
                pass
        #         break
        # else:
        #     continue
        # break

