import numpy as np
import point_cloud
import constants
import logging
import time
import cv2

import calibrate_cameras
from datetime import datetime


class DroneController(object):
    """Drone controller object"""

    REMAP_INTERPOLATION = cv2.INTER_LINEAR
    RENDER_POINT_CLOUD = False

    def __init__(self, rtsp_domain_left: str, rtsp_domain_right: str, image_shape: tuple):
        # domains to rtsp stream
        self.rtsp_domain_left = rtsp_domain_left
        self.rtsp_domain_right = rtsp_domain_right

        # image shape
        self.image_shape = image_shape

        # mappings for image rectification
        self.left_map_x, self.left_map_y = 0, 0
        self.right_map_x, self.right_map_y = 0, 0

        if self.RENDER_POINT_CLOUD:
            self.point_cloud_renderer = point_cloud.PointCloud("3D-representation", 2.0)

        # stereo brute force matcher
        self.stereo_matcher = cv2.StereoBM_create(constants.NUM_DISPARITIES, constants.BLOCK_SIZE)

    def compute_depth_map_with_images(self, left_image, right_image):
        """"Compute depth map from given images

        :param left_image: the left image to use
        :param right_image: the right image to use
        """
        return self.compute_depth_map(left_image, right_image)

    def compute_depth_map(self, left_image, right_image):
        """Computes depth map and returns it

        :param left_image:  the left image of the stereo camera
        :param right_image: the right image of the stereo camera

        :returns the generated depth map
        """
        disparity_map = self.stereo_matcher.compute(left_image, right_image)
        # correct places that are only shown on one of the images
        disparity_map[disparity_map == np.min(disparity_map)] = np.mean(disparity_map)

        # remap values from 0 to 1 + convert disparity to depth
        return 1.0 - disparity_map / np.max(disparity_map)

    def calibrate_cameras(self, path_to_images_left, path_to_images_right):
        """Calibrates cameras and sets intrinsic parameters

        :param path_to_images_left:  Path to folder of images for calibration of left camera
        :param path_to_images_right: Path to folder of images for calibration of the right camera
        """
        self.left_map_x, self.left_map_y, self.right_map_x, self.right_map_y = calibrate_cameras.calibrate_cameras(
            path_to_images_left, path_to_images_right)

    def load_calibration(self, path_to_calibration):
        """Load calibration from calibration file

        :param path_to_calibration: filepath of the calibration file
        """
        # load saved calibration
        calibration = np.load(path_to_calibration)

        assert calibration['img_size'] == self.image_shape[:2]
        self.left_map_x, self.left_map_y = calibration["left_map_x"], calibration["left_map_y"]
        self.right_map_x, self.right_map_y = calibration["right_map_x"], calibration["right_map_y"]

    def rectify_image_left(self, image: np.ndarray):
        """Undistorts and rectifies image from left camera

        :param image: image from left camera to be rectified
        """
        assert image.shape == self.image_shape
        return cv2.remap(image, self.left_map_x, self.left_map_y, self.REMAP_INTERPOLATION)

    def rectify_image_right(self, image: np.ndarray):
        """Undistorts and rectifies image from right camera

        :param image: image from the right camera to be rectified
        """
        assert image.shape == self.image_shape
        return cv2.remap(image, self.right_map_x, self.right_map_y, self.REMAP_INTERPOLATION)


if __name__ == '__main__':
    drone_controller = DroneController('0.0.0.0:1234', '0.0.0.0:1234', (300, 300))

    left_image = cv2.imread('tests/test_left.png', 0)
    right_image = cv2.imread('tests/test_right.png', 0)
    depth_map = drone_controller.compute_depth_map(left_image, right_image)

    while True:
        cv2.imshow('Depth Map', depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
