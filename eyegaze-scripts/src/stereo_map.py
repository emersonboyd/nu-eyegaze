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
imgL = cv.imread('{}/{}/imgL_undist2.png'.format(util.get_resources_directory(), image_path))
imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
imgR = cv.imread('{}/{}/imgR_undist2.png'.format(util.get_resources_directory(), image_path))
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

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
# METHOD 3: Brute-force matching with SIFT descriptors then Ransac
#

MIN_MATCH_COUNT = 10

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()  # need open-cv 3.4.2.17

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgL,None)
kp2, des2 = sift.detectAndCompute(imgR,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# show the matches before ransac filtering
# img3 = cv.drawMatches(imgL, kp1, imgR, kp2, good, outImg=None, flags=2)
# plt.imshow(img3, 'gray'), plt.show()

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    threshold = 20.0  # TODO perfect this number for the RPi
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, threshold)
    matchesMask = mask.ravel().tolist()

    h,w = imgL.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    imgR = cv.polylines(imgR,[np.int32(dst)],True,255,3, cv.LINE_AA)

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
    exit(1)


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

print(M)

# img3 = cv.drawMatches(imgL, kp1, imgR, kp2, good, None, **draw_params)
# plt.imshow(img3, 'gray'), plt.show()

for i in range(len(matchesMask)):
    if matchesMask[i] == 1:
        print('{} in kp1 matches with {} in kp2'.format(good[i].queryIdx, good[i].trainIdx))
        kpL = kp1[good[i].queryIdx]
        kpR = kp2[good[i].trainIdx]
        print('x diff between kp1 and kp2: {}, y diff between kp1 and kp2: {}', kpL.pt[0] - kpR.pt[0], kpL.pt[1] - kpR.pt[1])
        x_disparity_pixel =  kpL.pt[0] - kpR.pt[0]
        micrometer_per_pixel = 1.5
        focal_length_millimeter = 4.15
        focal_length_pixel = focal_length_millimeter / (micrometer_per_pixel / 1000)
        baseline_millimeter = 86.9
        estimated_depth_millimeter = baseline_millimeter * focal_length_pixel / x_disparity_pixel
        print('estimated depth {}'.format(estimated_depth_millimeter))

        # attempt to transform one point to the next
        xL = kpL.pt[0]
        yL = kpL.pt[1]
        pL = np.array([xL, yL, 1])
        pR = np.matmul(M, pL)
        print('old x and y: ({}, {})'.format(xL, yL))
        print('new x and y: ({}, {})'.format(pR[0], pR[1]))
        print('actual x and y: ({}, {})'.format(kpR.pt[0], kpR.pt[1]))


print('execution seconds:', time.time() - start_time)

