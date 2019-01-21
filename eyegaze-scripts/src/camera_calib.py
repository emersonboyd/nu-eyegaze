import numpy as np
import cv2 as cv
import glob

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
images = glob.glob('../res/pycam_calib_twisttie/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK # fast check speeds up the processing if no chessboard is found
    ret, corners = cv.findChessboardCorners(gray, (num_rows, num_cols), flags)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (num_rows, num_cols), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print('failed to find corners for', fname)

cv.destroyAllWindows()
