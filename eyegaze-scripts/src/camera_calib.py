import numpy as np
import cv2 as cv
import glob

# following this tutorial:
# https://docs.opencv.org/4.0.0/dc/dbb/tutorial_py_calibration.html

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
images = glob.glob('../res/picam_calib_twisttie/*.jpg')

for fname in images:
    img = cv.imread(fname)
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
        cv.drawChessboardCorners(img, (num_rows, num_cols), corners2, ret)
        cv.imshow(fname, img)
        cv.waitKey(5000)

    else:
        print('Failed to find corners for', fname)
        exit(1)

cv.destroyAllWindows()




print('Calculating camera calibration matrix...')
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)




print('Undistorting the picam photos...')
for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    print(newcameramtx)
    print(roi)

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
   
    # crop the image based on valid pixels calculated from getOptimalNewCameraMatrix()
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult.png', dst)

    # # Draw the image
    # cv.imshow('img'. )




print('Calculating reprojection error...')
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )




print ('Done...')
