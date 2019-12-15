import itertools

import cv2
import imutils
import numpy as np
from imutils import object_detection

from utils import timer, join, overlap, circle, rectangle, point


def detect_circles(gray_image, min_radius, max_radius, param1, param2):
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, min_radius, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    results = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            circ = circle(point(i[0], i[1]), i[2])
            results.append(circ)

    return results


def detect_rectangles(image, area_thresh, coef, approximation_type=cv2.CHAIN_APPROX_SIMPLE):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, approximation_type)

    results = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, coef * cv2.arcLength(contour, True), True)

        if len(approx) != 4:
            continue

        # (x11, y11)        (x22, y22)
        #   o-----------------o
        #   |                 |
        #   |                 |
        #   o-----------------o
        # (x12, y12)       (x21, y21)

        ((x11, y11),), ((x12, y12),), ((x21, y21),), ((x22, y22),) = approx

        # Check that the sides are equal
        w1 = x22 - x11
        w2 = x21 - x12
        h1 = y12 - y11
        h2 = y21 - y22

        diff = abs(w1 - w2)
        if diff > 0.1 * w1 or diff > 0.1 * w2:
            continue

        diff = abs(h1 - h2)
        if diff > 0.1 * h1 or diff > 0.1 * h2:
            continue

        # Check that the sides are HORIZONTAL or PARALLEL
        thresh_x = (x22 - x11) * 0.1
        thresh_y = (y12 - y11) * 0.1
        if any([
            abs(x11 - x12) > thresh_x,
            abs(x22 - x21) > thresh_x,
            abs(y11 - y22) > thresh_y,
            abs(y12 - y21) > thresh_y
        ]):
            continue

        if cv2.contourArea(approx) > area_thresh:
            x, y, w, h = cv2.boundingRect(approx)
            results.append(rectangle(x, y, w, h))

    return results


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
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rects, confidences


def rescale_text_rects(boxes, rel_x, rel_y):
    # We need to rescale all the boxes with text to have the coordinates of the original image
    results = []
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * rel_x)
        start_y = int(start_y * rel_y)
        end_x = int(end_x * rel_x)
        end_y = int(end_y * rel_y)

        results.append(rectangle(start_x, start_y, end_x - start_x, end_y - start_y))
    return results


def apply_padding(boxes, join_padding, original_dimensions):
    results = []
    padding_x, padding_y = join_padding
    orig_width, orig_height = original_dimensions

    # We need to rescale all the boxes with text to have the coordinates of the original image
    for (start_x, start_y, w, h) in boxes:
        dx = int(w * padding_x)
        dy = int(h * padding_y)

        start_x = max(0, start_x - dx)
        start_y = max(0, start_y - dy)

        end_x = min(orig_width, start_x + w + dx)
        end_y = min(orig_height, start_y + h + dy)

        results.append(rectangle(start_x, start_y, end_x - start_x, end_y - start_y))
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


def detect_text(image, model_path, min_confidence, join_padding, join_overlapping=False):
    height, width = image.shape[:2]
    with timer('Read network'):
        net = cv2.dnn.readNet(model_path)

    image, rel_x, rel_y = resize_image_for_net(image, height, width)
    new_height, new_width = image.shape[:2]

    boxes = apply_east_text_detection(image, min_confidence, net, new_height, new_width)

    resize_results = rescale_text_rects(boxes, rel_x, rel_y)
    results = apply_padding(resize_results, join_padding, (new_width, new_height))

    results = sorted(results, key=lambda r: r.x)

    if join_overlapping:
        results = join_overlapping_rectangles(results)
    return results


def apply_east_text_detection(image, min_confidence, net, new_height, new_width):
    with timer('Blob from image'):
        blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), True, False)
    output_layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
    net.setInput(blob)
    with timer('Forward layers to net'):
        scores, geometry = net.forward(output_layers)
    with timer('Decode predictions'):
        (rects, confidences) = decode_predictions(scores, geometry, min_confidence)
    with timer('Non max suppression'):
        boxes = imutils.object_detection.non_max_suppression(np.array(rects), probs=confidences)
    return boxes


def resize_image_for_net(image, height, width):
    # Image dimensions must be multiples of 32 for this to work
    new_height, new_width = height - height % 32, width - width % 32
    rel_x, rel_y = width / new_width, height / new_height
    image = cv2.resize(image, (new_width, new_height))

    return image, rel_x, rel_y


def is_checked(image_rect, wanted_ratio=0.7):
    _, thresh = cv2.threshold(image_rect, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = cv2.countNonZero(thresh)
    total = thresh.shape[0] * thresh.shape[1]
    ratio = white / total
    checked = ratio < wanted_ratio
    return checked
