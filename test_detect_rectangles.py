import argparse
import logging
import os
import sys

import cv2
import numpy

FORMAT = '[%(asctime)s] [%(levelname)s] : %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
WIDTH = 800

WINDOW_TITLE = "Test contour approximation"
APPROX_TRACKBAR_TITLE = 'Find contours approximation type\n' \
                        '0 = APPROX_NONE\n' \
                        '1 = APPROX_SIMPLE\n' \
                        '2 = APPROX_TC89_L1\n' \
                        '3 = APPROX_TC89_KCOS'
COEF_TRACKBAR_TITLE = "Polygon approximation coefficient"
AREA_TRACKBAR_TITLE = 'Area threshold'


approximations = {
    0: cv2.CHAIN_APPROX_NONE,
    1: cv2.CHAIN_APPROX_SIMPLE,
    2: cv2.CHAIN_APPROX_TC89_L1,
    3: cv2.CHAIN_APPROX_TC89_KCOS
}


approx_to_string = {
    cv2.CHAIN_APPROX_NONE: 'CHAIN_APPROX_NONE',
    cv2.CHAIN_APPROX_SIMPLE: 'CHAIN_APPROX_SIMPLE',
    cv2.CHAIN_APPROX_TC89_L1: 'CHAIN_APPROX_TC89_L1',
    cv2.CHAIN_APPROX_TC89_KCOS: 'CHAIN_APPROX_TC89_KCOS'
}


def operation(_):
    processed = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    image = original.copy()

    approximation_type_n = cv2.getTrackbarPos(APPROX_TRACKBAR_TITLE, WINDOW_TITLE)
    approximation_type = approximations[approximation_type_n]

    logging.info(f'Finding contours with approximation type: {approx_to_string[approximation_type]}')
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, approximation_type)

    coef = cv2.getTrackbarPos(COEF_TRACKBAR_TITLE, WINDOW_TITLE) / 100 * 0.1
    logging.info(f'Approximating polygons with coef: {coef}')

    area_thresh = cv2.getTrackbarPos(AREA_TRACKBAR_TITLE, WINDOW_TITLE)
    logging.info(f'Area threshold: {area_thresh}')
    for contour in contours:
        approx = cv2.approxPolyDP(contour, coef * cv2.arcLength(contour, True), True)

        if len(approx) == 4:
            try:
                if cv2.contourArea(approx) > area_thresh:
                    x, y, w, h = cv2.boundingRect(approx)
                    rect_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                    rect_contour = numpy.array(rect_points).reshape((-1, 1, 2)).astype(numpy.int32)

                    cv2.drawContours(image, [rect_contour], 0, (0, 0, 255), 2)
            except Exception as e:
                logging.exception('Error when trying to draw contour {}'.format(e))

    images = numpy.hstack((original, image))
    cv2.imshow('Shapes', images)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Path to image to analyze')
args = parser.parse_args()

if not os.path.isfile(args.image_path):
    print('Specified image does not exist.')
    exit(1)

original = cv2.imread(args.image_path)

logging.info('Converting to gray')
gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

cv2.namedWindow(WINDOW_TITLE)
cv2.createTrackbar(COEF_TRACKBAR_TITLE, WINDOW_TITLE, 0, 100, operation)
cv2.createTrackbar(APPROX_TRACKBAR_TITLE, WINDOW_TITLE, 0, 3, operation)
cv2.createTrackbar(AREA_TRACKBAR_TITLE, WINDOW_TITLE, 0, 20000, operation)


operation(0)

cv2.waitKey()
cv2.destroyAllWindows()
