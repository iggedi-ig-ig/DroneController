import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

from itertools import chain
import glob

# constants
CHECKERBOARD_PATTERN_SIZE = (7, 6)  # size of checkerboard pattern to search

OBJECT_POINT_ZERO = np.zeros((CHECKERBOARD_PATTERN_SIZE[0] * CHECKERBOARD_PATTERN_SIZE[1], 3), dtype=np.float32)
OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHECKERBOARD_PATTERN_SIZE[0], 0: CHECKERBOARD_PATTERN_SIZE[1]].T.reshape(-1, 2)

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

OPTIMIZE_ALPHA = 0.25


def parse_images(directory: str):
    obj_points = []
    img_points = []

    shape = None

    for image_path in chain(glob.glob(f"{directory}/*.jpg"), glob.glob(f"{directory}/*.png")):
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if shape is None:
            shape = image_gray.shape
        else:
            if shape != image_gray.shape:
                raise ValueError(f"Image is wrong size: {image_path}\n\tExpected: {shape}")

        has_corners, corners = cv2.findChessboardCorners(image_gray, CHECKERBOARD_PATTERN_SIZE, cv2.CALIB_CB_FAST_CHECK)

        if has_corners:
            print(f"Corners found: {image_path}")

            cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)

            obj_points.append(OBJECT_POINT_ZERO)
            img_points.append(corners)
        else:
            print(f"No corners found: {image_path}")

    return obj_points, img_points, shape


def get_camera_data(obj_points: list, img_points: list, img_shape: tuple):
    _, matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

    return matrix, distortion_coefficients


def get_camera_relation(obj_points, points_l, points_r, matrix_l, distortion_l, matrix_r, distortion_r, img_shape):
    (_, _, _, _, _, rotation_matrix, translation_vector, _, _) = cv2.stereoCalibrate(obj_points,
                                                                                     points_l, points_r,
                                                                                     matrix_l, distortion_l,
                                                                                     matrix_r, distortion_r,
                                                                                     img_shape,
                                                                                     flags=cv2.CALIB_FIX_INTRINSIC,
                                                                                     criteria=TERMINATION_CRITERIA)

    return rotation_matrix, translation_vector


def get_rectification(matrix_l, distortion_l, matrix_r, distortion_r, shape, rotation_matrix, translation_vector):
    (rectified_l, rectified_r, projection_l, projection_r, disparity_to_depth_map, roi_l, roi_r) = cv2.stereoRectify(
        matrix_l, distortion_l,
        matrix_r, distortion_r,
        shape,
        rotation_matrix, translation_vector,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=OPTIMIZE_ALPHA)

    return rectified_l, projection_l, rectified_r, projection_r, roi_l, roi_r


def get_mapping(camera_matrix, distortion, rectification, projection, shape):
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion, rectification, projection,
                                               shape, cv2.CV_32FC1)

    return map_x, map_y


def calibrate_cameras(left_path: str, right_path: str):
    obj_points_l, img_points_l, shape_l = parse_images(left_path)
    obj_points_r, img_points_r, shape_r = parse_images(right_path)

    # make sure cameras have the same resolution
    assert shape_l == shape_r

    # fixed parameters
    left_matrix, left_distortion = get_camera_data(obj_points_l, img_points_l, shape_l)
    right_matrix, right_distortion = get_camera_data(obj_points_r, img_points_r, shape_r)

    # varying parameters
    rotation_matrix, translation_vector = get_camera_relation(obj_points_l, img_points_l, img_points_r, left_matrix,
                                                              left_distortion, right_matrix, right_distortion, shape_l)

    rectified_l, projection_l, rectified_r, projection_r, roi_l, roi_r = get_rectification(left_matrix, left_distortion,
                                                                                           right_matrix,
                                                                                           right_distortion,
                                                                                           shape_l,
                                                                                           rotation_matrix,
                                                                                           translation_vector)

    map_x_l, map_y_l = get_mapping(left_matrix, left_distortion, rectified_l, projection_l, shape_l)
    map_x_r, map_y_r = get_mapping(right_matrix, right_distortion, rectified_r, projection_r, shape_r)

    np.savez_compressed(sys.argv[3], img_size=shape_l,
                        left_map_x=map_x_l, left_map_y=map_y_l, left_roi=roi_l,
                        right_map_x=map_x_r, right_map_y=map_y_r, right_roi=roi_r)

    print(f"Saved configuration to file: {sys.argv[3]}")


if __name__ == '__main__':
    calibrate_cameras(sys.argv[1].strip('/'), sys.argv[2].strip('/'))
