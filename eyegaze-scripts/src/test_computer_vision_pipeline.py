import cv2 as cv

from constants import CameraType
import util
import image_helper as ih
import time

def run():
    start = time.time()

    camera_type = CameraType.EMERSON_IPHONE_6_PLUS
    test_path = '{}/iphone_6_plus_emerson_calib_test'.format(util.get_resources_directory())
    image_left = cv.imread('{}/IMG_5736.JPG'.format(test_path), cv.IMREAD_GRAYSCALE)
    image_right = cv.imread('{}/IMG_5737.JPG'.format(test_path), cv.IMREAD_GRAYSCALE)

    mtx, dist = ih.get_calib_data_for_camera_type(camera_type)
    image_left = ih.undistort(image_left, mtx, dist)
    image_right = ih.undistort(image_right, mtx, dist)

    M = ih.get_homography_matrix(image_left, image_right)

    pixel_left = (2400.90, 921.68)
    est_pixel_right = ih.calculate_corresponding_pixel_right(pixel_left, M)
    if not util.pixel_in_bounds(image_right, est_pixel_right):
        print('The corresponding pixel for the given pixel does not exist')
        exit(1)
    depth = ih.calculate_depth(pixel_left, est_pixel_right, camera_type)

    # ih.show_image_with_mark(image_left, pixel_left)

    print('estimated depth in left image pixel ({}, {}): {} meters'.format(pixel_left[0], pixel_left[1], depth))

    print(time.time() - start)

if __name__ == '__main__':
    run()
