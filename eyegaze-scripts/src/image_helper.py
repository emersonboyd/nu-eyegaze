import cv2 as cv
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import itertools
import time

import util
from constants import CameraType


IMAGE_HELPER_MIN_MATCH_COUNT = 10


def get_calib_data_for_camera_type(camera_type):
    res_path = util.get_resources_directory()

    if camera_type == CameraType.EMERSON_IPHONE_6_PLUS:
        calib_data_path = '{}/iphone_6_plus_emerson.pickle'.format(res_path)
    elif camera_type == CameraType.PICAM_LEFT:
        calib_data_path = '{}/picam_left.pickle'.format(res_path)
    elif camera_type == CameraType.PICAM_RIGHT:
        calib_data_path = '{}/picam_right.pickle'.format(res_path)
    else:
        print('Invalid camera type entered.')
        exit(1)


    with open(calib_data_path, 'rb') as calib_data_file:
        ret, mtx, dist, rvecs, tvecs = pickle.load(calib_data_file)

    return mtx, dist


def undistort(input_image, mtx, dist):
    h, w = input_image.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # check for errors
    if roi[0] == 0 and roi[1] == 0 and roi[2] == 0 and roi[3] == 0:
        print('Failed to get valid roi for', fname)
        exit(1)

    # undistort
    dst = cv.undistort(input_image, mtx, dist, None, newcameramtx)

    # crop the image based on valid pixels calculated from getOptimalNewCameraMatrix()
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    return dst


def get_homography_matrix(image_left, image_right):
    time_1 = time.time()

    size_factor = 0.5
    downsample_images = True

    if downsample_images:
        image_left = np.copy(image_left)
        image_right = np.copy(image_right)
        image_left = cv.blur(image_left, (5,5))
        image_right = cv.blur(image_right, (5,5))
        image_left = cv.resize(image_left, dsize=None, fx=size_factor, fy=size_factor)
        image_right = cv.resize(image_right, dsize=None, fx=size_factor, fy=size_factor)

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()  # need opencv-python==3.4.2.17 and opencv-contrib-python==3.4.2.17 or if that doesn't work then 3.4.2.16 for both

    time_1 = time.time() - time_1
    time_2 = time.time()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)

    time_2 = time.time() - time_2
    time_3 = time.time()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    time_3 = time.time() - time_3
    time_4 = time.time()

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # show the matches before ransac filtering
    # img3 = cv.drawMatches(image_left, kp1, image_right, kp2, good, outImg=None, flags=2)
    # plt.imshow(img3, 'gray'), plt.show()
    # cv.imwrite('/tmp/image_correspondences_raw.jpg', img3)

    time_4 = time.time() - time_4
    time_5 = time.time()

    if len(good) > IMAGE_HELPER_MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        threshold = 20.0  # TODO perfect this number for the RPi
        if downsample_images:
            threshold = threshold * size_factor

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, threshold)
        matchesMask = mask.ravel().tolist()

        h, w, _ = image_left.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        image_right = np.ascontiguousarray(image_right, dtype=np.uint8)
        image_right = cv.polylines(image_right, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), IMAGE_HELPER_MIN_MATCH_COUNT))
        matchesMask = None
        exit(1)

    time_5 = time.time() - time_5
    time_6 = time.time()

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    flags=2)
    # img3 = cv.drawMatches(image_left, kp1, image_right, kp2, good, None, **draw_params)
    # plt.imshow(img3, 'gray')
    # plt.show()
    # cv.imwrite('/tmp/image_correspondences_cleaned.jpg', img3)

    if downsample_images:
        for kp in itertools.chain(kp1, kp2):
            kp.pt = (kp.pt[0] / size_factor, kp.pt[1] / size_factor)

    time_6 = time.time() - time_6

    print('\n\n\n')
    print('Time 1: {} seconds'.format(time_1))
    print('Time 2: {} seconds'.format(time_2))
    print('Time 3: {} seconds'.format(time_3))
    print('Time 4: {} seconds'.format(time_4))
    print('Time 5: {} seconds'.format(time_5))
    print('Time 6: {} seconds'.format(time_6))
    print('\n\n\n')

    return M, kp1, kp2, good, matchesMask


# determines the corresponding pixel in the right image based off the homography matrix
def calculate_corresponding_pixel_right(pixel_left, M):
    x_left, y_left = pixel_left

    estimated_pixel_right = np.matmul(M, np.array([x_left, y_left, 1]))[:2]
    print(estimated_pixel_right)

    return estimated_pixel_right


# calculates the depth of the location given in millimeters based on the pixels in two photos
def calculate_depth(pixel_left, pixel_right, camera_type):
    x_left, y_left = pixel_left
    x_right, y_right = pixel_right

    micrometer_per_pixel = camera_type.get_pixel_size()
    focal_length_millimeter = camera_type.get_focal_length()
    focal_length_pixel = focal_length_millimeter / (micrometer_per_pixel / 1000)
    baseline_millimeter = camera_type.get_baseline()

    # TODO consider NOT using both x disparity and y disparity combined
    # x_disparity_pixel = x_left - x_right
    import math
    x_disparity_pixel = math.sqrt(math.pow(x_left - x_right, 2) + math.pow(y_left - y_right, 2))
    estimated_depth_millimeter = baseline_millimeter * focal_length_pixel / x_disparity_pixel

    return estimated_depth_millimeter / 1000


# calculates the angle between a the user and a pixel in the image
def calculate_angle_to_pixel(image, pixel, horizontal_fov_degrees):
    degrees_per_half = horizontal_fov_degrees / 2

    pixel_x = pixel[0]
    center_x = image.shape[1] / 2

    pixels_from_center_x = abs(center_x - pixel_x)
    degrees_from_center = degrees_per_half * pixels_from_center_x / center_x

    if pixel_x < center_x:
        return -1 * degrees_from_center

    return degrees_from_center


def show_image_with_mark(image, pixel):
    x, y = pixel

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(image, cmap=mpl.cm.gray)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    circ = Circle((x, y), 25)
    ax.add_patch(circ)

    # Show the image
    plt.show()
