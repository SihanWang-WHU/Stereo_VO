This program can implement camera calibration and calculates Stereo VO

This program is tested on python 3.8, opencv-python 3.4.16.59, and Pyqt5

The test data is kitti, you can reach http://www.cvlibs.net/datasets/kitti for the whole dataset

This program uses LK to track the key frame, and allows SIFT and ORB to detect the features and 
descriptors between frames 

Codes in this branch（master) is the version that runs on the qt window, you can check out the Console_View 
branch for the version that runs on the python console (using the same algorithm)
