import argparse
import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to image to analyze')
    args = parser.parse_args()

    logging.info('Reading image')
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    detector = cv2.SimpleBlobDetector()

    logging.info('Detecting blobs')
    keypoints = detector.detect(image)

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    logging.info('Drawing keypoints')
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", im_with_keypoints)
    print('here')
    cv2.waitKey()
