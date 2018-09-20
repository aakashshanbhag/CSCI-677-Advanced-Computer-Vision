# -*- coding: utf-8 -*-
"""
Spyder Editor
Author : Aakash Shanbhag with aid from the tutorial for OPENCV watershed
(http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html)
Title: Watershed Segmentation.
"""
  
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random

# Reading image
img = cv2.imread('/users/aakashshanbhag/desktop/computer_vision_usc/300091(12).jpg')
img_copy=img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

height, width = img.shape[:2]

im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

# Thresholding by otsu and binary inverse
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('thresholded image',thresh)
cv2.waitKey(0)
    
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

cv2.imshow('opened image',opening)
cv2.waitKey(0)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

print ("The total number of markers created is %d" %ret)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1 

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,255,255]

# Display input image
cv2.imshow("img", img)
k=cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

for x in range(-1,ret+1):
    img[markers == x] = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]
   
cv2.imshow("img_2", img) 
cv2.waitKey(0)



# Trial script for the marker creation
marker_trial=0;
img_trial = np.zeros([height, width], dtype=np.int32)

i=0
while i<(height-1):
    j=0
    while j<(width-1):
       img_trial[i,j]=marker_trial 
       j+=200
       marker_trial+=1
    i+=150

print ("The total number of markers created is %d" %marker_trial)   
img_trial = cv2.watershed(img_copy,img_trial)
img_copy[img_trial == -1] = [0,0,255]
cv2.imshow("trial", img_copy) 
cv2.waitKey(0)




       


    
    
        
        













