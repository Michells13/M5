from pathlib import Path
import cv2


# set paths
imdir1 = Path(r"orig_mask_out")
imdir2 = Path(r"finetune_mask_out")


for filename1, filename2 in zip(imdir1.iterdir(), imdir2.iterdir()):
    img1 = cv2.imread(str(filename1))
    img2 = cv2.imread(str(filename2))
    # img1 = cv2.resize(img1, (720, 360))
    # img2 = cv2.resize(img2, (720, 360))
    
    cv2.imshow("finetuned", img2)
    cv2.imshow("orig", img1)
    
    print(filename1)
    print(filename2)
    
    if cv2.waitKey(0) == ord("q"):
        exit()
    cv2.destroyAllWindows()
