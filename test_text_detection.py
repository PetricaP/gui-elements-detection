import argparse
import contextlib
import itertools
import logging
import time

import cv2
import imutils
import numpy
import pytesseract
from imutils import object_detection

logging.basicConfig(level=logging.INFO)


def overlap(rect1, rect2):
    x11 = rect1[0]
    y11 = rect1[1]
    x12 = rect1[2]
    y12 = rect1[3]

    x21 = rect2[0]
    y21 = rect2[1]
    x22 = rect2[2]
    y22 = rect2[3]

    d1 = min(y12, y22) - max(y11, y21)
    d2 = min(x12, x22) - max(x11, x21)

    if d1 <= 0 or d2 <= 0:
        return 0

    a1 = (x12 - x11) * (y12 - y11)
    a2 = (x22 - x21) * (y22 - y21)

    # The ration between the overlap area and the smaller rectangle area
    return d1 * d2 / min(a1, a2)


def join(rect1, rect2):
    return min(rect1[0], rect2[0]), min(rect1[1], rect2[1]), max(rect1[2], rect2[2]), max(rect1[3], rect2[3])


@contextlib.contextmanager
def timer(operation_name):
    old = time.time()
    yield
    diff = time.time() - old
    logging.info(f'{operation_name} finished in {diff:.4f} seconds')


def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, numCols):
            if scores_data[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = angles_data[x]
            cos = numpy.cos(angle)
            sin = numpy.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rects, confidences


def get_text_rects(padding, boxes, orig_height, orig_width, rel_x, rel_y):
    results = []
    # We need to rescale all the boxes with text to have the coordinates of the original image
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * rel_x)
        start_y = int(start_y * rel_y)
        end_x = int(end_x * rel_x)
        end_y = int(end_y * rel_y)

        dx = int((end_x - start_x) * padding)
        dy = int((end_y - start_y) * padding)

        start_x = max(0, start_x - dx)
        start_y = max(0, start_y - dy)
        end_x = min(orig_width, end_x + (dx * 2))
        end_y = min(orig_height, end_y + (dy * 2))

        results.append((start_x, start_y, end_x, end_y))
    return results


def apply_ocr_on_rects(image, joined):
    results = []
    for start_x, start_y, end_x, end_y in joined:
        start_x = int(start_x)
        start_y = int(start_y)
        end_x = int(end_x)
        end_y = int(end_y)

        roi = image[start_y:end_y, start_x:end_x]

        # config = "-l eng --oem 1 --psm 7"
        config = "-l eng --oem 1"
        text = pytesseract.image_to_string(roi, config=config)

        results.append(((start_x, start_y, end_x, end_y), text))
    return results


def join_overlapping_rectangles(results):
    joined = set()
    processed = set()

    while True:
        found_overlap = False
        for rect1, rect2 in itertools.combinations(results, 2):
            if rect1 not in processed and rect2 not in processed:
                if overlap(rect1, rect2):
                    joined.add(join(rect1, rect2))
                    processed.add(rect1)
                    processed.add(rect2)
                    found_overlap = True

        for rect in results:
            if rect not in processed:
                joined.add(rect)

        if not found_overlap:
            break
        else:
            results = joined
            joined = set()
            processed = set()

    return joined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to east model pb file')
    parser.add_argument('image_path', help='image to analyze')
    parser.add_argument('--join_overlapping', action='store_true', help='join overlapping text rectangles')
    parser.add_argument('--min_confidence', type=float, help='minimum confidence for text position', default=0.8)
    parser.add_argument('--padding', type=float, help='padding to add when looking for text', default=0.1)
    parser.add_argument('--draw_text', action='store_true', help='Draw the detected text near the boxes.')
    parser.add_argument('--apply_ocr', action='store_true', help="Don't apply ocr, just draw the rects")
    args = parser.parse_args()

    original_image = cv2.imread(args.image_path)
    height, width = original_image.shape[:2]

    with timer('Read network'):
        net = cv2.dnn.readNet(args.model_path)

    # Image dimensions must be multiples of 32 for this to work
    new_height, new_width = height - height % 32, width - width % 32

    rel_x, rel_y = width / new_width, height / new_height

    image = cv2.resize(original_image, (new_width, new_height))

    with timer('Blob from image'):
        blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), True, False)

    output_layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    net.setInput(blob)

    with timer('Forward layers to net'):
        scores, geometry = net.forward(output_layers)

    with timer('Decode predictions'):
        (rects, confidences) = decode_predictions(scores, geometry, args.min_confidence)

    with timer('Non max suppression'):
        boxes = imutils.object_detection.non_max_suppression(numpy.array(rects), probs=confidences)

    with timer('Getting text rects'):
        results = get_text_rects(args.padding, boxes, height, width, rel_x, rel_y)

        results = sorted(results, key=lambda r: r[0])

    if args.join_overlapping:
        with timer('Join overlapping rectangles'):
            results = join_overlapping_rectangles(results)

    with timer('Drawing rectangles'):
        for start_x, start_y, end_x, end_y in results:
            cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

    if args.apply_ocr:
        with timer('Apply OCR on rects'):
            results = apply_ocr_on_rects(image, results)

        results = sorted(results, key=lambda r: r[0][1])

        if args.draw_text:
            with timer('Drawing text'):
                for ((start_x, start_y, end_x, end_y), text) in results:
                    cv2.putText(original_image, text, (start_x, start_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for (_, text) in results:
            logging.info(text)

    cv2.imshow("Text Detection", original_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
