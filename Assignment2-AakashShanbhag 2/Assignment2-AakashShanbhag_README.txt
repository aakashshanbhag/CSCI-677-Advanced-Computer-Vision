CSCI 677: Advanced Computer Vision
Assignment 2: Comparison between the Mean shift Segmentation and Watershed segmentation
Author: Aakash Shanbhag
USC ID:3205699915
------------------------------------------------------------------------------------------------
The code was built on MAC-OS Sierra with python 3.5 and OpenCV 3.1

Files included:
•Meanshift1.py for the Mean Shift segmentation Question 1
•Watershed.py for the the Watershed based superpixel segmentation

------------------------------------------------------------------------------------------------

These files can be run with the Opencv version mentioned above with the required packages for python to support them. They can be simply compiled with any Python compiler.

------------------------------------------------------------------------------------------------

Arguments to be passed for Question 1:
./Meanshift1.py Image.jpg ParameterUpdate

argv[0]: The compiled executable.
argv[1]: Image of any format can be read.
argv[2]:ParameterUpdate
		•Spatial_radius -> outputs with different spatial radius with a fixed color radius.
		•Color radius -> outputs with different color radius with a fixed spatial radius.

------------------------------------------------------------------------------------------------

Arguments to be passed for Question 2:
./Watershed.py Image.jpg placement_interval_x placement_interval_y 

argv[0]: The compiled executable.
argv[1]: Image of any format can be read.
argv[2]: placement_interval_x
		•placement_interval_x -> Interval in the x direction where the marker needs to be placed.
argv[3]: placement_interval_y
		•placement_interval_y -> Interval in the y direction where the marker needs to be placed.

------------------------------------------------------------------------------------------------

•Meanshift Segmentation algorithm works with Lab color space with the level 1 filtering.
•Watershed Segmentation is done by method 2 of creating Superpixels in which the marker points are spaced at uniform  intervals using the input from the user in the respective directions




