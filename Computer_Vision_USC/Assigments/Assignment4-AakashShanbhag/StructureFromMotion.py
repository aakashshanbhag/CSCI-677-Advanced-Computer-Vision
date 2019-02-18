#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 4: Structure from motion
@author: aakashshanbhag
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank
from mpl_toolkits.mplot3d import Axes3D
from sys import argv

# Command line argumets dealt with
script, train, query = argv

print("The script called:", script)
print("Train image: ", train)
print("Test/Query image: ", query)

# Reading the input image and the image to be matched
gray = cv2.imread(train,0)
gray2 = cv2.imread(query,0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp, des = sift.detectAndCompute(gray,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# Drawing keypoints on the test image and the match_image
img_train=cv2.drawKeypoints(gray,kp,gray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_matching=cv2.drawKeypoints(gray2,kp2,gray2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
top_matches=sorted(good,key=lambda x:x.distance)

# Creating map for the best matches after the good match criterion according to Lowe is set.
img4=cv2.drawMatches(gray,kp,gray2,kp2,good,None,flags=2)

img4_display=cv2.resize(img4, (1200, 800))
# cv2.namedWindow("Good matches before RANSAC", cv2.WINDOW_NORMAL)
# cv2.imshow("Good matches before RANSAC",img4_display)
       
# Print the total number of good parameters 
print ("The total number of good matches found are: ", len(good))

intrinsic=[[2760.0,0.0,1520.0,0.0,2760.0,1006.0,0.0,0.0,1.0]]
intrinsic=np.array(intrinsic).reshape((3,3))

# Choosing only top choices in the good choices obtained for matching
src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

# Remove barrel and similar non linear distortions for higher accuracy with normalisation
src_pts_undistort = cv2.undistortPoints(src_pts,intrinsic,distCoeffs=None) 
dst_pts_undistort = cv2.undistortPoints(dst_pts,intrinsic,distCoeffs=None)

# Calculating the essentail matrix and the points consistent with the output
E, mask = cv2.findEssentialMat(src_pts_undistort, dst_pts_undistort, method=cv2.RANSAC, prob=0.999, threshold=3.0)
print("The total number of consistent value inliers with the essential matrix are: ",np.sum(mask))

# Display essential matrix 
print("The Essential matrix found is as follows: ")
print(E)

# Verify the rank is 2
print("matrix rank",matrix_rank(E)) 

# Recovering the pose using the undistorted points
_, R, t, mask = cv2.recoverPose(E, src_pts_undistort, dst_pts_undistort)

# Computed Rotation and translation matrix
print("The Rotation matrix found is as follows: ") 
print(R)
print(np.linalg.det(R))

print("The translation vector found is as follows: ")
print(t)

# Creating the camera extrinsics matrices
M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

# Creating the camera projection matrix
P_l = np.dot(intrinsic,  M_l)
P_r = np.dot(intrinsic,  M_r)

# Triangulate the points with regards to the normalisedpoints
point4d=cv2.triangulatePoints(P_l,P_r,src_pts_undistort,dst_pts_undistort)
print (point4d)
point4d_non_hom = point4d / np.tile(point4d[-1, :], (4, 1))

# 3D points calculated 
point_3d = point4d_non_hom[:3, :].T

# Plotting 3D plots
fig=plt.figure()
ax=Axes3D(fig)

#ax.scatter(-0.550724,-0.36449,1.0,'r')
ax.scatter(point_3d[:,0],point_3d[:,1],point_3d[:,2],c='r',marker='.')
plt.show()

# Destroy all the created windows on button press
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


