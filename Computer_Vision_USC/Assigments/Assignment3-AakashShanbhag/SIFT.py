# -*- coding: utf-8 -*-
"""

Tutorial utiilised as starter:
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
Assignment 3: SIFT feature matching and homography computation
@ Aakash Shanbhag
"""
import cv2
import numpy as np
from sys import argv
# Command line argumets dealt with
script, train, query = argv

print("The script called:", script)
print("Train image: ", train)
print("Test/Query image: ", query)

# Set up the minimum number of points that need to be matched
MIN_MATCH_COUNT = 20

# Reading the input image and the image to be matched
img = cv2.imread(train)
img2 = cv2.imread(query)

# Converting to gray scale since SIFT works only on gray scale images
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp, des = sift.detectAndCompute(gray,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# Drawing keypoints on the test image and the match_image
img_train=cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_matching=cv2.drawKeypoints(gray2,kp2,gray2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the key point descriptors
cv2.imshow("Test image key point descriptors",img_train)
cv2.imshow("Match_image key point descriptors",img_matching)

# Count the number of keypoint descriptors in test and match image 
keypoint_test,w1=des.shape
keypoint_match,w2=des2.shape

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des,des2, k=2)

# Store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# Sorting on the basis of the least distance concept in ascending order        
topMatches=sorted(good,key=lambda x:x.distance)

# Creating map for the best 20 matches after the good match criterion according to Lowe is set.
img4=cv2.drawMatches(gray,kp,gray2,kp2,topMatches[:20],None,flags=2)
cv2.imshow("Good matches before RANSAC",img4)
       
# Print the total number of good parameters 
print ("The total number of good matches found are: ", len(good))


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Calculating the homography matrix from the good match points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = gray2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    # Computing the perpective transform for matching the points and creating lines
    dst = cv2.perspectiveTransform(pts,M)

    gray2 = cv2.polylines(gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Lesser than required 20 good matches - Good matches found are equal to : ",len(good) )
    matchesMask = None

# Print the total number of values consistent with the found homography(INLIERS)
print("The total number of consistent value inliers with the homography are: ",np.sum(matchesMask))

# Print the final Homography matrix  
print("The homography matrix found is as follows: ")
print(M)

# Draw the relationship match with green color with only inliers under context
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

# After taking into account the RANSAC algorithm
img3 = cv2.drawMatches(gray,kp,gray2,kp2,good,None,**draw_params)

# Output the match that results from RANSAC process. 
cv2.imshow('match',img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
