import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from tkinter import *

import time

import util

start_time = time.time()

image_path = 'depth_photos/cones'
image_path = 'depth_photos/tsukuba'
image_path = 'stereo_photos'
imgL = cv.imread('{}/{}/imgL_undist.png'.format(util.get_resources_directory(), image_path))
imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
# imgL = cv.rotate(imgL, cv.ROTATE_90_CLOCKWISE)
imgR = cv.imread('{}/{}/imgR_undist.png'.format(util.get_resources_directory(), image_path))
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
# imgR = cv.rotate(imgR, cv.ROTATE_90_CLOCKWISE)

# stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL, imgR)
# plt.imshow(disparity, 'gray')
# plt.show()






#
#
# https://docs.opencv.org/4.0.0/dc/dc3/tutorial_py_matcher.html
# sudo -H pip3 install opencv-python==3.4.2.17 opencv-contrib-python==3.4.2.17
#
# Try this next for inliers:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
#



#
# METHOD 1: Brute-force matching with ORB descriptors
#

# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(imgL,None)
kp2, des2 = orb.detectAndCompute(imgR,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(imgL, kp1, imgR, kp2, matches[:100], outImg=None, flags=2)
# plt.imshow(img3),plt.show()







#
# METHOD 2: Brute-force matching with SIFT descriptors
#

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgL,None)
kp2, des2 = sift.detectAndCompute(imgR,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(imgL,kp1,imgR,kp2,good,outImg=None,flags=2)
# plt.imshow(img3),plt.show()








print('execution seconds:', time.time() - start_time)