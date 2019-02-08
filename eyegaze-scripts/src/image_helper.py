import cv2 as cv
import util
from constants import CameraType


IMAGE_HELPER_MIN_MATCH_COUNT = 10


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
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()  # need open-cv 3.4.2.17

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # show the matches before ransac filtering
    # img3 = cv.drawMatches(imgL, kp1, imgR, kp2, good, outImg=None, flags=2)
    # plt.imshow(img3, 'gray'), plt.show()

    if len(good) > IMAGE_HELPER_MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        threshold = 20.0  # TODO perfect this number for the RPi
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, threshold)
        matchesMask = mask.ravel().tolist()

        h, w = imgL.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        imgR = cv.polylines(imgR, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), IMAGE_HELPER_MIN_MATCH_COUNT))
        matchesMask = None
        exit(1)

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    flags=2)
    # img3 = cv.drawMatches(imgL, kp1, imgR, kp2, good, None, **draw_params)
    # plt.imshow(img3, 'gray'), plt.show()

    return M


# calculates the depth of the object given based on the two pixel locations
def calculate_depth(pixel_left, pixel_right, camera_type):
    x_left, y_left = pixel_left
    x_right, y_right = pixel_right

    if camera_type == CameraType.EMERSON_IPHONE_6_PLUS:
        micrometer_per_pixel = 1.5
        focal_length_millimeter = 4.15
        focal_length_pixel = focal_length_millimeter / (micrometer_per_pixel / 1000)
        baseline_millimeter = 86.9
    else:
        print('Invalid camera type entered')
        exit(1)

    return depth
