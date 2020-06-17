# noinspection PyUnresolvedReferences
import point_cloud
import numpy as np
import constants
import time
import cv2

from cv2 import imread as read

REMAP_INTERPOLATION = cv2.INTER_LINEAR
POINT_CLOUD_SCALE = 0.05

DEBUG = True
POINTCLOUD = False


def get_depth_map(img_left: np.ndarray, img_right: np.ndarray, stereo: cv2.StereoBM) -> np.ndarray:
    """
    Returns depth map from two rectified images

    :param img_left: left image
    :param img_right: right image
    :param stereo: StereoBM object

    :return: depth map (normalized and inverted)
    """

    disparity_map = stereo.compute(img_left, img_right)
    disparity_map[disparity_map == np.min(disparity_map)] = np.mean(disparity_map)  # correct not found pixels

    # if we divide by that, values will be between 0 and 1
    depth_scale = np.max(disparity_map)

    # normalize between 0 and 1 and invert
    disparity_map = 1.0 - (disparity_map / depth_scale)

    return disparity_map


def plot_image(pc: point_cloud.PointCloud, img: np.ndarray, depth_map: np.ndarray, scale=0.2, step=1):
    """
    Renders pixels in 3D with depth as Z

    :param pc: PointCloud object
    :param img: colorized image to plot (shape (w, h, 3))
    :param depth_map: depth map (shape (w, h))
    :param scale: distance between points
    :param step: step size of loop through pixels

    :return: void
    """
    for y in range(0, img.shape[1], step):
        for x in range(0, img.shape[0], step):
            if x >= img.shape[1] or y >= img.shape[0]:
                break

            pc.add_point((x * scale, y * scale, depth_map[x, y]), tuple(img[x, y]))


def get_direction(depth_map: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray, iterations=150, step_size=3.0) -> (
        np.ndarray, np.ndarray):
    """
    Returns the direction the drone should fly in as a 3D vector

    :param depth_map: depth map
    :param grad_x: The partial derivative in respect to x of the depth map
    :param grad_y: The partial derivative in respect to y of the depth map
    :param iterations: The number of iterations / steps that should be taken


    :return: The input vector
    """
    origin = np.array([depth_map.shape[1] / 2, depth_map.shape[0] / 2])
    pos = origin

    positions = []
    for _ in range(iterations):
        step_x = grad_x[int(pos[1]), int(pos[0])]
        step_y = grad_y[int(pos[1]), int(pos[0])]

        pos += np.array([step_x, step_y]) * step_size
        positions.append(pos.copy())

    vec = origin - pos
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0.0 else norm, np.array(positions)


def main():
    """main function"""

    def process_frame_pair(frames, color) -> bool:
        left = frames[0]
        right = frames[1]

        # make sure images are the same size
        assert left.shape == right.shape

        # get depth map with rectified images
        depth_map = get_depth_map(left, right, stereo)

        # plot image with depth
        if POINTCLOUD:
            plot_image(pc, np.rot90(colorized, -1), np.rot90(depth_map, -1) * (1.0 / POINT_CLOUD_SCALE), POINT_CLOUD_SCALE, step=1)

        grad_x, grad_y = np.gradient(cv2.GaussianBlur(depth_map, (25, 25), 25.0) * 33.3, 1)
        direction, points = get_direction(depth_map, grad_x, grad_y, 250, 1.0)

        debug = np.zeros(depth_map.shape)
        for point in points:
            cv2.circle(debug, (int(point[0]), int(point[1])), 1, (255, 255, 255), 1)

        if cv2.waitKey(1) & 0xFF != ord('q') and (not POINTCLOUD or pc.render()):
            cv2.imshow('left', rectified_left)
            cv2.imshow('depth', depth_map)
            cv2.imshow('debug', debug)

        return False

    # object to compute disparity map
    stereo: cv2.StereoBM = cv2.StereoBM_create(constants.NUM_DISPARITIES, constants.BLOCK_SIZE)

    # custom point cloud renderer
    if POINTCLOUD:
        pc: point_cloud.PointCloud = point_cloud.PointCloud("3D Representation", 2.0)

    # load saved calibration
    calibration = np.load("test_calibration.npz")

    img_size = calibration["img_size"]
    left_map_x = calibration["left_map_x"]
    left_map_y = calibration["left_map_y"]

    right_map_x = calibration["right_map_x"]
    right_map_y = calibration["right_map_y"]
    right_map_y = calibration["right_map_y"]

    # noinspection PyUnboundLocalVariable
    frames = 0
    durations = []
    frames_per_second = []
    while True:  # not cameras or (cameras and capture_l.isOpened() and cameras and capture_r.isOpened()):
        pre = time.time()

        # save colorized image to render point cloud
        colorized = read('tests/test_left.png') / 255.0
        img_left, img_right = read('tests/test_left.png', cv2.CV_8UC1), read('tests/test_right.png', cv2.CV_8UC1)

        # remap images to align normals of image planes
        rectified_left = cv2.remap(img_left, left_map_x, left_map_y, REMAP_INTERPOLATION)
        rectified_right = cv2.remap(img_right, right_map_x, right_map_y, REMAP_INTERPOLATION)

        if DEBUG:
            rectified_left = img_left
            rectified_right = img_right

        if process_frame_pair((rectified_left, rectified_right), colorized):
            break

        durations.append(dur := round(time.time() - pre, 3))
        frames_per_second.append(fps := round(1.0 / dur, 3))
        print(
            "\rProcessed Frame: {frames} (took {dur:<5} ms, avg: {avg:<7} | fps: {fps:<7}, avg fps: {avg_fps:<7})".format(
                frames=(frames := frames + 1),
                dur=dur,
                avg=round(float(np.mean(durations)), 3),
                fps=fps,
                avg_fps=round(float(np.mean([1.0 / x for x in durations])), 3)),
            end='')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # test()
