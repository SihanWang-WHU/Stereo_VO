This program can implement camera calibration and calculates Stereo VO

This program is tested on python 3.8 and opencv-python 3.4.16.59

The test data is kitti, you can reach http://www.cvlibs.net/datasets/kitti for the whole dataset

This program uses LK to track the key frame, and allows SIFT and ORB to detect the features and 
descriptors

Codes in this branch is the version that runs on the python console, you can check out the master 
branch for the version that has a window written in Pyqt5 (same algorithm)