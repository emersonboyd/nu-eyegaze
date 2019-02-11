import numpy as np
import cv2 as cv
import glob
import pickle
from constants import CameraType
import image_helper

import util

# following this tutorial:
# https://docs.opencv.org/4.0.0/dc/dbb/tutorial_py_calibration.html


def calculateCameraCalibrationData(input_path):
    # these numbers indicated the number of interior corners
    num_rows = 8
    num_cols = 6

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(num_rows-1,num_cols-1,0)
    objp = np.zeros((num_rows * num_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    input_images = glob.glob('{}/{}'.format(util.get_resources_directory(), input_path))

    print('Finding corners in image dataset...')
    for fname in input_images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK  # fast check speeds up the processing if no chessboard is found
        ret, corners = cv.findChessboardCorners(gray, (num_rows, num_cols), flags)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
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
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def run():
    camera_type = CameraType.PICAM_LEFT

    if camera_type == CameraType.EMERSON_IPHONE_6_PLUS:
        input_path = 'iphone_6_plus_emerson_calib/*.JPG'
        test_path = 'iphone_6_plus_emerson_calib_test/*.JPG'
        calibration_data_path = 'iphone_6_plus_emerson.pickle'
    elif camera_type == CameraType.PICAM_RIGHT:
        input_path = 'picam_right_calib/*.jpg'
        test_path = 'picam_right_calib/*.jpg'
        calibration_data_path = 'picam_right.pickle'
    elif camera_type == CameraType.PICAM_LEFT:
        input_path = 'picam_left_calib/*.jpg'
        test_path = 'picam_left_calib/*.jpg'
        calibration_data_path = 'picam_left.pickle'
    else:
        print('Invalid camera type')
        exit(1)

    full_calibration_data_path = '{}/{}'.format(util.get_resources_directory(), calibration_data_path)
    if not util.file_exists(full_calibration_data_path):
        with open(full_calibration_data_path, 'wb') as calibration_data_file:
            ret, mtx, dist, rvecs, tvecs = calculateCameraCalibrationData(input_path)
            pickle.dump((ret, mtx, dist, rvecs, tvecs), calibration_data_file)
    else:
        with open(full_calibration_data_path, 'rb') as calibration_data_file:
            ret, mtx, dist, rvecs, tvecs = pickle.load(calibration_data_file)

    test_images = glob.glob('{}/{}'.format(util.get_resources_directory(), test_path))

    print('Undistorting the photos...')
    for fname in test_images:
        img = cv.imread(fname)
        dst = image_helper.undistort(img, mtx, dist)

        output_image_path = '{}/calibresult_{}.png'.format(util.get_output_directory(), fname.split('/')[-1])
        cv.imwrite(output_image_path, dst)


    # print('Calculating reprojection error...')
    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    #     mean_error += error
    # print("total error: {}".format(mean_error/len(objpoints)))

    print('Done...')


if __name__ == '__main__':
    run()
