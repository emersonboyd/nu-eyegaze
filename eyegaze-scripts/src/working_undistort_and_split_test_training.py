import glob
from os.path import expanduser
import random
from constants import CameraType
import image_helper as ih
import cv2 as cv
import matplotlib.pyplot as plt
import util


PERCENTAGE_TEST_IMAGES = 0.07
CAMERA_TYPE = CameraType.PICAM_RIGHT

HOME_PATH = expanduser("~")
ORIGINAL_IMAGE_PATH = glob.glob('{}/Desktop/PiCam Signs/*.jpg'.format(HOME_PATH))
UNDISTORTED_IMAGE_PATH_TRAIN = '{}/sign_photos/trainTHISPROTECTSAGAINSTOVERWRITING'.format(util.get_resources_directory())
UNDISTORTED_IMAGE_PATH_TEST = '{}/sign_photos/testTHISPROTECTSAGAINSTOVERWRITING'.format(util.get_resources_directory())

mtx, dist = ih.get_calib_data_for_camera_type(CAMERA_TYPE)

i = 1
for original_image_path in ORIGINAL_IMAGE_PATH:
    original_image = cv.imread(original_image_path)
    undistorted_image = ih.undistort(original_image, mtx, dist)

    undistorted_image_name = 'image{}.jpg'.format(i)

    # check whether we should save the photo as a test photo or a train photo
    if random.uniform(0, 1) > PERCENTAGE_TEST_IMAGES:
        undistorted_image_path = '{}/{}'.format(UNDISTORTED_IMAGE_PATH_TRAIN, undistorted_image_name)
    else:
        undistorted_image_path = '{}/{}'.format(UNDISTORTED_IMAGE_PATH_TEST, undistorted_image_name)

    cv.imwrite(undistorted_image_path, undistorted_image)

    i += 1
