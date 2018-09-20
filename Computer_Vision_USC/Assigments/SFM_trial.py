
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

img_train_display=cv2.resize(img_train, (1200, 800))
cv2.namedWindow("Test image key point descriptors", cv2.WINDOW_NORMAL)
cv2.imshow("Test image key point descriptors",img_train_display)

img_matching_display=cv2.resize(img_matching, (1200, 800))
cv2.namedWindow("Match_image key point descriptors", cv2.WINDOW_NORMAL)
cv2.imshow("Match_image key point descriptors",img_matching_display)

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
img4=cv2.drawMatches(gray,kp,gray2,kp2,topMatches[:800],None,flags=2)

img4_display=cv2.resize(img4, (1200, 800))
cv2.namedWindow("Good matches before RANSAC", cv2.WINDOW_NORMAL)
cv2.imshow("Good matches before RANSAC",img4_display)
       
# Print the total number of good parameters 
print ("The total number of good matches found are: ", len(good))

intrinsic=[[2760,0,1520,0,2760,1006,0,0,1]]
intrinsic=np.array(intrinsic).reshape((3,3))

# Choosing only top 200 choices in the good choices obtained for matching
src_pts = np.float32([ kp[m.queryIdx].pt for m in topMatches[:800] ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in topMatches[:800] ]).reshape(-1,1,2)

# Remove barrel and similar non linear distortions for higher accuracy
src_pts_undistort = cv2.undistortPoints(src_pts,intrinsic,distCoeffs=None) 
dst_pts_undistort = cv2.undistortPoints(dst_pts,intrinsic,distCoeffs=None)

# Calculating the essentail matrix and the points consistent with the output
E, mask = cv2.findEssentialMat(src_pts_undistort, dst_pts_undistort,2760, (1520,1006),cv2.RANSAC, prob=0.999, threshold=1.0)
print("The total number of consistent value inliers with the essential matrix are: ",np.sum(mask))

# Display essential matrix 
print("The Essential matrix found is as follows: ")
print(E)

# Verify the rank is 2
print("matrix rank",matrix_rank(E)) 

_, R, t, mask = cv2.recoverPose(E, src_pts_undistort, dst_pts_undistort,intrinsic)

# Computed Rotation and translation matrix
print("The Rotation matrix found is as follows: ") 
print(R)

print("The translation vector found is as follows: ")
print(t)

# Creating the camera matrices

M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(intrinsic,  M_l)
P_r = np.dot(intrinsic,  M_r)

src_pts_undistort_array=np.array(src_pts_undistort).reshape((2,800))
dst_pts_undistort_array=np.array(dst_pts_undistort).reshape((2,800))

point4d=cv2.triangulatePoints(P_l,P_r,src_pts_undistort_array,dst_pts_undistort_array)
point4d_non_hom = point4d / np.tile(point4d[-1, :], (4, 1))
point_3d = point4d_non_hom[:3, :].T

fig=plt.figure()
ax=Axes3D(fig)

ax.scatter(point_3d[:,0],point_3d[:,1],point_3d[:,2])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)