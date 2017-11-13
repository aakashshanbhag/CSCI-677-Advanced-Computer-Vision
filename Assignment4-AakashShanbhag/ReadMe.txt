CSCI 677: Advanced Computer Vision
Assignment 4: Structure from Motion
Author: Aakash Shanbhag
USC ID:3205699915
------------------------------------------------------------------------------------------------
The code was built on MAC-OS Sierra with python 3.5 and OpenCV 3.1

Files included:
•StructureFromMotion.py for 3D reconstruction of points.

------------------------------------------------------------------------------------------------

These files can be run with the Opencv version mentioned above with the required packages for python to support them. They can be simply compiled with any Python compiler.

------------------------------------------------------------------------------------------------

Arguments to be passed for Question 1:
./StructureFromMotion.py image1.jpg image2.jpg

argv[0]: The compiled executable.
argv[1]: image1 of any format can be read from the first view.
argv[2]: image2 of any format can be read from the first view.
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
Important to note that windows might appear one top of another. Move the windows to best view results. Key press kills all windows. Rotate the scatter plot for multiple views.
3 D reconstruction done with all the inliers that are obtained in this case. Ideal case with Lowe’s ratio of 0.7 and using all 1206 good matches for reconstruction. 