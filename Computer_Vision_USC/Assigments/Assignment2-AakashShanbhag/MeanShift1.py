# -*- coding: utf-8 -*-
"""
Spyder Editor
Author : Aakash Shanbhag
Experiment with Mean Shift Segmentation using level one pyramid with different spatial radii and color radii.
"""
import cv2
import numpy as np
from sys import argv

script, first, second = argv

print("The script called:", script)
print("Input image: ", first)
print("Control parameter: ", second)

# Reading image
img = cv2.imread(first)

# Converting to Lab space as expected
lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)


if second=='Spatial_radius':
    # Pyramid mean shifted filtering 
    # with different spatial radius 
    shifted_21_51 = cv2.pyrMeanShiftFiltering(lab, 21, 51)
    shifted_7_51 = cv2.pyrMeanShiftFiltering(lab, 7, 51)
    shifted_35_51 = cv2.pyrMeanShiftFiltering(lab, 35, 51)
    shifted_78_51 = cv2.pyrMeanShiftFiltering(lab, 78, 51)
    shifted_55_51 = cv2.pyrMeanShiftFiltering(lab, 55, 51)


    # Display input image
    cv2.imshow("Input", lab)
    cv2.moveWindow("Input",0,0)
    
    #Display output image
    cv2.imshow("output_21_51",shifted_21_51)
    cv2.moveWindow("output_21_51",np.size(img,1)+10,0)
    
    cv2.imshow("output_7_51",shifted_7_51)
    cv2.moveWindow("output_7_51",2*(np.size(img,1))+20,0)
    
    cv2.imshow("output_35_51",shifted_35_51)
    cv2.moveWindow("output_35_51",0,np.size(img,0)+60)
    
    cv2.imshow("output_78_51",shifted_78_51)
    cv2.moveWindow("output_78_51",(np.size(img,1))+10,np.size(img,0)+60)
    
    cv2.imshow("output_55_51",shifted_55_51)
    cv2.moveWindow("output_55_51",2*(np.size(img,1))+20,np.size(img,0)+60)
    
    # Destroy all the windows
    k=cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        cv2.waitKey(1)

else:
    # Different 
    shifted_21_5 = cv2.pyrMeanShiftFiltering(lab, 21, 51)
    shifted_21_35 = cv2.pyrMeanShiftFiltering(lab, 21, 35)
    shifted_21_51 = cv2.pyrMeanShiftFiltering(lab, 21, 51)
    shifted_21_57 = cv2.pyrMeanShiftFiltering(lab, 21, 57)
    shifted_21_87 = cv2.pyrMeanShiftFiltering(lab, 21, 87)

    # Display input image
    cv2.imshow("Input", lab)
    cv2.moveWindow("Input",0,0)
    
    #Display output image
    cv2.imshow("output_21_5",shifted_21_5)
    cv2.moveWindow("output_21_5",np.size(img,1)+10,0)
    
    cv2.imshow("shifted_21_35",shifted_21_35)
    cv2.moveWindow("shifted_21_35",2*(np.size(img,1))+20,0)
    
    cv2.imshow("shifted_21_51",shifted_21_51)
    cv2.moveWindow("shifted_21_51",0,np.size(img,0)+60)
    
    cv2.imshow("shifted_21_57",shifted_21_57)
    cv2.moveWindow("shifted_21_57",(np.size(img,1))+10,np.size(img,0)+60)
    
    cv2.imshow("shifted_21_87",shifted_21_87)
    cv2.moveWindow("shifted_21_87",2*(np.size(img,1))+20,np.size(img,0)+60)
    
    # Destroy all the windows
    k=cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        cv2.waitKey(1)


