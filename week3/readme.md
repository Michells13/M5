### Scripts
* make_cooc_matrix.py -- new script for the coocurrence matrix computation. It uses vectorized numpy functions, and much more efficient than the old one (few milliseconds for the val COCO)
* make_txt.py -- create a list representation of a coocurrence matrix
* coocurrence.txt -- the output of the script above
* select_anns.py -- input categories, get annotations that contain them but not together
* transplant.py -- specify object to cut out, and the image to insert it in with annotation id. Get combined image. Run an object-detector inference. Save results.
