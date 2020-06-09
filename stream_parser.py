#!/usr/bin/python3.8

import cv2  # for computer vision
import sys  # for command line arguments


# save RTSP stream domain to global variable (format: rtsp://0.0.0.0:8080/out.h264)
RTSP_STREAM_DOMAIN = sys.argv[1]


def process_frame(frame):
    """processes frame and sends controll data to drone"""
    pass


def main():
    """Main function called on run"""
    video_capture = cv2.VideoCapture(RTSP_STREAM_DOMAIN)

    while True:
        return_code, pixel_matrix = video_capture.read()

        if not return_code:
            break

        process_frame(pixel_matrix)


if __name__ == '__main__':
    main()

