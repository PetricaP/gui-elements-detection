import argparse
import logging
import sys
import time

import cv2
import numpy
import os

FORMAT = '[%(asctime)s] [%(levelname)s] : %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
WIDTH = 800


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to image to analyze')
    parser.add_argument('--processing_type', nargs=1, help='Type of processing to do on image (thresh / canny)')
    parser.add_argument('--width', help='Resize image to the specified width', type=int, default=WIDTH)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print('path not exist')

    image = cv2.imread(args.image_path)

    w, h = image.shape[0], image.shape[1]

    image = cv2.resize(image, (args.width, int(w / h * args.width)))

    logging.info('Copying original image')
    original = image.copy()

    logging.info('Converting to gray')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if args.processing_type and args.processing_type[0] == 'canny':
        logging.info('Applying canny processing')
        processed = cv2.Canny(gray_image, 60, 120)
    else:
        logging.info('Applying adaptive threshold')
        processed = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    logging.info('Finding contours')
    t = time.perf_counter()
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Finding contours in: {time.perf_counter() - t}')

    for contour in contours:
        logging.debug('Approximating contour {}'.format(contour))
        t = time.perf_counter()
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        print(f'Approximating contour in: {time.perf_counter() - t}')

        if len(approx) == 4:

            try:
                if cv2.contourArea(approx) > 80:
                    x, y, w, h = cv2.boundingRect(approx)
                    rect_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                    rect_contour = numpy.array(rect_points).reshape((-1, 1, 2)).astype(numpy.int32)

                    logging.info('Drawing bounding rect for {}'.format(str(approx).replace('\n', '')))
                    cv2.drawContours(image, [rect_contour], 0, (0, 0, 255), 2)

                    color_threshold = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(color_threshold, [rect_contour], 0, (0, 0, 255), 2)

                    images = numpy.hstack((original, color_threshold, image))
                    cv2.imshow('Shapes', images)

                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
                else:
                    logging.debug('Skipping too small image')
            except Exception as e:
                logging.exception('Error when trying to draw contour {}'.format(e))

    images = numpy.hstack((original, cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR), image))
    cv2.imshow('Shapes', images)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
