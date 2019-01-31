import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
from tkinter import *

import util

image_path = 'depth_photos/cones'
image_path = 'depth_photos/tsukuba'
image_path = 'stereo_photos'
imgL = cv.imread('{}/{}/imgL_undist.png'.format(util.get_resources_directory(), image_path))
imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
imgL = cv.rotate(imgL, cv.ROTATE_90_CLOCKWISE)
imgR = cv.imread('{}/{}/imgR_undist.png'.format(util.get_resources_directory(), image_path))
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
imgR = cv.rotate(imgR, cv.ROTATE_90_CLOCKWISE)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
