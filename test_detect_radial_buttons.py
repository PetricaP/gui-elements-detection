import argparse
import itertools
import json

import cv2

from detection import detect_circles, detect_text, is_checked
from utils import point, circle, is_inside_circle

RADIAL_BUTTON_PARAM1 = 14
RADIAL_BUTTON_PARAM2 = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the .pb file')
    parser.add_argument('image_path', help='Path to image to find buttons in.')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = detect_circles(gray, 5, 7, 10, 15)

    text_rects = detect_text(image, args.model_path, 0.8, (0.4, 0.1), True)

    for start_x, start_y, w, h in text_rects:
        cv2.rectangle(image, (start_x, start_y), (start_x + w, start_y + h), (0, 0, 255), 2)

    for circ in circles:
        # circle outline
        cv2.circle(image, circ.center, circ.radius, (255, 0, 0), 3)

    first_results = []

    for rect, circ in itertools.product(text_rects, circles):
        if circ.center.y > rect.y and is_inside_circle(point(rect.x, rect.y), circle(circ.center, circ.radius * 3)):
            cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)
            first_results.append((rect, circ))

    results = []
    for rect, circ in first_results:
        image_rect = gray[circ.center.y - circ.radius: circ.center.y + circ.radius,
                          circ.center.x - circ.radius: circ.center.x + circ.radius]
        checked = is_checked(image_rect)

        results.append({
            'rectangle': rect.to_json(),
            'button_circle': circ.to_json(),
            'is_checked': checked
        })

    print(json.dumps(results, indent=4))

    cv2.imshow("Result", image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
