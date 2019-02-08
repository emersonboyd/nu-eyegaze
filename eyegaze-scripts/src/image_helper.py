import cv2 as cv
import util


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
