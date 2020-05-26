import point_cloud
import numpy as np
import constants
import cv2

from cv2 import imread as read

REMAP_INTERPOLATION = cv2.INTER_LINEAR
POINT_CLOUD_SCALE = 0.05


def get_depth_map(img_left: np.ndarray, img_right: np.ndarray, stereo: cv2.StereoBM) -> np.ndarray:
    """
    Returns depth map from two rectified images

    :param img_left: left image
    :param img_right: right image
    :param stereo: StereoBM object

    :return: depth map (normalized and inverted)
    """

    disparity_map = stereo.compute(img_left, img_right)

    # if we divide by that, values will be between 0 and 1
    depth_scale = np.max(disparity_map)

    # normalize between 0 and 1 and invert
    disparity_map = 1.0 - (disparity_map / depth_scale)

    # smooth edges (Gaussian blur, radius 25, 25 sigma)
    # disparity_map = cv2.GaussianBlur(disparity_map, (25, 25), 2.5)

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


def get_direction(depth_map: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Returns the direction the drone should fly in as a 3D vector

    :param depth_map: depth map
    :param grad_x: The partial derivative in respect to x of the depth map
    :param grad_y: The partial derivative in respect to y of the depth map


    :return: The origin and the vector
    """

    h, w = (grad_x + grad_y).shape

    # do something like gradient descent form the center
    center_pos = np.array([w // 2, h // 2], dtype=np.float32)

    debug_img = np.zeros((w, h, 3), dtype=np.float32)

    pos = center_pos
    for i in range(50):
        d_x = grad_x[int(round(pos[0])), int(round(pos[1]))] * 750
        d_y = grad_y[int(round(pos[0])), int(round(pos[1]))] * 750

        v = np.array([d_x, d_y], dtype=np.float32)
        norm = np.linalg.norm(v)

        if norm < 1.0:
            break

        pos += v
        cv2.circle(debug_img, tuple(np.round(pos)), 3, (0., 1., 0.), 1)

    origin = np.array([w // 2, h // 2, 0])
    target = np.array((*pos[:2], 0))
    target[2] = depth_map[int(target[0]), int(target[1])]

    return origin, target


def main():
    """main function"""

    # object to compute disparity map
    stereo: cv2.StereoBM = cv2.StereoBM_create(constants.NUM_DISPARITIES, constants.BLOCK_SIZE)

    # custom point cloud renderer
    pc: point_cloud.PointCloud = point_cloud.PointCloud("test", 2.0)

    cameras = False
    if cameras:
        capture_l = cv2.VideoCapture(0)
        capture_r = cv2.VideoCapture(1)

    # load saved calibration
    calibration = np.load("test_calibration.npz")

    img_size = calibration["img_size"]
    left_map_x = calibration["left_map_x"]
    left_map_y = calibration["left_map_y"]
    # left_roi = calibration["left_roi"]

    right_map_x = calibration["right_map_x"]
    right_map_y = calibration["right_map_y"]
    # right_roi = calibration["right_roi"]

    # noinspection PyUnboundLocalVariable
    while not cameras or (cameras and capture_l.isOpened() and cameras and capture_r.isOpened()):
        # save colorized image to render point cloud
        colorized = read('tests/test_left.png') / 255.0
        img_left, img_right = read('tests/test_left.png', cv2.CV_8UC1), read('tests/test_right.png', cv2.CV_8UC1)

        # remap images to align normals of image planes
        rectified_left = cv2.remap(img_left, left_map_x, left_map_y, REMAP_INTERPOLATION)
        rectified_right = cv2.remap(img_right, right_map_x, right_map_y, REMAP_INTERPOLATION)

        # TODO: remove this
        rectified_left = img_left
        rectified_right = img_right

        # make sure images are the same size
        assert rectified_left.shape == rectified_right.shape

        # make sure configuration fits camera specifications
        # assert rectified_left.shape == img_size == rectified_right.shape

        # get depth map with rectified images
        depth_map = get_depth_map(rectified_left, rectified_right, stereo)

        # plot image with depth
        plot_image(pc, np.rot90(colorized, -1), np.rot90(depth_map, -1) * (1.0 / POINT_CLOUD_SCALE), POINT_CLOUD_SCALE, step=1)

        grad_x, grad_y = np.array(np.gradient(cv2.GaussianBlur(depth_map, (99, 99), 15), 1))
        gradient = (grad_x + grad_y) / np.max(grad_x + grad_y)

        origin, direction = get_direction(np.rot90(depth_map, -1), grad_x, grad_y)
        pc.add_line(tuple(origin * POINT_CLOUD_SCALE),
                    tuple([direction[0] * POINT_CLOUD_SCALE,
                           direction[1] * POINT_CLOUD_SCALE,
                           direction[2] * (1.0 / POINT_CLOUD_SCALE)]),(255, 0, 0))

        while cv2.waitKey(1) & 0xFF != ord('q') and pc.render():
            cv2.imshow("debug", direction)
            cv2.imshow("gradient", gradient * 0.5 + 0.5)
            cv2.imshow("depth", depth_map)

        cv2.destroyAllWindows()

        if not cameras:
            break


def test():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")

    video_capture = cv2.VideoCapture('videos/10 Hours of Walking in NYC as a Woman-b1XGPvbWn0A.mp4')

    while video_capture.isOpened():
        _, img = video_capture.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img_gray, 1.5, 6)

        for (x, y, w, h) in faces:
            if w * h <= 5:
                continue

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # test()
