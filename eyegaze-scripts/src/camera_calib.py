import numpy as np
import cv2 as cv
import glob

import util

# following this tutorial:
# https://docs.opencv.org/4.0.0/dc/dbb/tutorial_py_calibration.html

# resize_images = True
# input_path = 'iphone_6_plus_emerson_calib/*.JPG'
# resize_images = False
# input_path = 'picam_calib_twisttie/*.jpg'
resize_images = False
input_path = 'picam_no_twisttie_calib/*.jpg'

# these numbers indicated the number of interior corners
num_rows = 8
num_cols = 6

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(num_rows-1,num_cols-1,0)
objp = np.zeros((num_rows*num_cols, 3), np.float32)
objp[:,:2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('{}/{}'.format(util.get_resources_directory(), input_path))




print('Finding corners in image dataset...')
for fname in images:
    img = cv.imread(fname)
    if resize_images:
    	img = cv.resize(img, (0,0), fx=0.3, fy=0.3) 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK # fast check speeds up the processing if no chessboard is found
    ret, corners = cv.findChessboardCorners(gray, (num_rows, num_cols), flags)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (num_rows, num_cols), corners2, ret)
        # cv.imshow(fname, img)
        # cv.waitKey(500)

    else:
        print('Failed to find corners for', fname)
        exit(1)

cv.destroyAllWindows()




print('Calculating camera calibration matrix...')
# TODO the gray.shape could be buggy because it's using image dimensions from the last acquired image
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)




print('Undistorting the photos...')
for fname in images:
    img = cv.imread(fname)
    if resize_images:
    	img = cv.resize(img, (0,0), fx=0.3, fy=0.3) 
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # check for errors
    if (roi[0] == 0 and roi[1] == 0 and roi[2] == 0 and roi[3] == 0):
        print('Failed to get valid roi for', fname)
        exit(1)

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
   
    # crop the image based on valid pixels calculated from getOptimalNewCameraMatrix()
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('{}/calibresult_{}.png'.format(util.get_output_directory(), fname.split('/')[-1]), dst)




print('Calculating reprojection error...')
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )




print ('Done...')
