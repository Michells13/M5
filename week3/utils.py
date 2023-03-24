import os
import cv2
import random
import numpy as np
from pycocotools.coco import COCO


class coMatriz:
    def __init__(self):
        self.matriz = [[(0, '') for j in range(81)] for i in range(81)]
    
    def get_value(self, row, col):
        return self.matriz[row][col]
    
    
    def set_valor(self, row, col, value, classes):
        self.matriz[row][col] = (value, classes)

def selectRandomImage(path):
    """
    Given a path that contains images, this function return an image randomly selected
    ---------------------
    input:
        path: path of the folder
    output
        image: opencv image
    """

    # Get a list of all image file names in the directory
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

    # Select a random image file name from the list
    image_file = random.choice(image_files)

    # Read the image using OpenCV
    image_path = os.path.join(path, image_file)
    image = cv2.imread(image_path)

    # Return the image
    return image

def co_ocurrenceOf2classes(class1,class2,coco,img_ids):
    """
    Given two classes(strings) this function iterates over all the image ids  to see if  the occurrence of the classes exists 
    if it exists at each image it adds 1 to cnt variable for that pair match and returns that counter 
    ---------------------
    input:
        class1: class 1 to be evaluated
        class2: class 2 to be evaluated
        coco: coco instance
        img_ids: id of all the images in the dataset
    output
        cnt: counter of the occurrence of a pair of classes over all the images
    """
    cnt=0
    
    for img_id in img_ids:
        # carga todas las anotaciones de objetos para la imagen específica
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        # verifica si las dos clases están presentes en la imagen
        class1_present = False
        class2_present = False
        for annotation in annotations:
            if coco.loadCats(annotation['category_id'])[0]['name'] == class1:
                class1_present = True
            elif coco.loadCats(annotation['category_id'])[0]['name'] == class2:
                class2_present = True
        # si ambas clases están presentes, imprime un mensaje
        if class1_present and class2_present:
            #print(f"La imagen {img_id} contiene tanto objetos de la clase {class1} como objetos de la clase {class2}.")
            img_info = coco.loadImgs(img_id)[0]
            #print(f"Nombre de archivo: {img_info['file_name']}")
            cnt=cnt+1
    classes=class1+" and "+class2        
    return cnt, classes
def co_OcurrenceMatrix(pathToAnnotations):
    """
    Given the path of a .json file in coco format, it returns the co-occurrence matrix of all the classes 
    
    ---------------------
    input:
        path: path to the .json file annotations
    output
        coOMatrix: numpy matrix 
        coMatrix:  obj from coMatriz class that contains co-occurrence values and an string 
                   of what classes have been used to get that  co-occurrence value
    """    

    # get coco instance
    coco = COCO(pathToAnnotations)
    # get images IDs
    img_ids = coco.getImgIds()
    # Get classes from annotations
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]
    
    #create numpy matrix   and custom matrix ==> coMatrix.set_value(row, cols, value, classes )
    coMatrix = coMatriz()
    matrixSize=len(category_names)
    coOMatrix= np.zeros((matrixSize, matrixSize))
    #Iterate over all classes twice to see if there is precence of ocurrence for each pair
    #higher values = higher number of ocurrence 
    for x in range(0,len(category_names)):
        for y in range(0,len(category_names)):
            #co_ocurrenceOf2classes checks if there is occurrence between a pair of classes given class1 and class2
            # which are selected by the loop (x,y) position
            value,classes=co_ocurrenceOf2classes(category_names[x], category_names[y], coco,img_ids)    
            coMatrix.set_valor(x, y, value, classes)
            coOMatrix[x, y]=value
            #print(co_ocurrenceOf2classes(category_names[x], category_names[y], coco,img_ids))
        print("Step: "+str(x) + " of 80")

    return coOMatrix, coMatrix



