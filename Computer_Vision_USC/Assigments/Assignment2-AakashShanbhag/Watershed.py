#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aakashshanbhag
Title: Watershed Segmentation by Method 2(SuperPixel) creation by randomly selecting Marker points.
"""
import numpy as np
import cv2
import random
from sys import argv

script, first, placement_x, placement_y = argv

print("The script called:", script)
print("Input image: ", first)

# Reading image
img = cv2.imread(first)

# Computing the height and the width of the image 
height, width = img.shape[:2]

#Creating a copy so that the original image is not manipulated
img_copy=img.copy()

# Creation of the marker as a stream of zeros with Marker points set at every m and n values
marker_trial=0;
img_trial = np.zeros([height, width], dtype=np.int32)

i=0
while i<(height-1):
    j=0
    while j<(width-1):
       img_trial[i,j]=marker_trial 
       j+=int(placement_x)
       marker_trial+=1
    i+=int(placement_y)

# Display the total number of markers
print ("The total number of markers created is %d" %marker_trial)

# Carry out the watershed algorithm with marker created above   
img_trial = cv2.watershed(img_copy,img_trial)

# Marking the boundaries of the image with RED
img_copy[img_trial == -1] = [0,0,255]

cv2.imshow("Input", img)
cv2.moveWindow("Input",0,0)

# Watershed segmented image display
cv2.imshow("Output", img_copy)
cv2.moveWindow("Output",np.size(img,1)+10,0) 

# Display the different segemented regions with different colors 
for x in range(-1,marker_trial+1):
    img_copy[img_trial == x] = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]
   
cv2.imshow("Segmented Regions", img_copy)
cv2.moveWindow("Segmented Regions",2*(np.size(img,1))+20,0)

k=cv2.waitKey(0) 
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
cv2.waitKey(1)