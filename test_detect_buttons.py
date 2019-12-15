import argparse
import itertools
import json

import cv2

from detection import detect_rectangles, detect_text
from utils import overlap, is_rect_inside_rect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to .pb file')
    parser.add_argument('image_path', help='Path to image to open')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    rects = detect_rectangles(processed, 50, 0.0)

    rects_to_remove = set()
    for inner_rect, outer_rect in itertools.combinations(rects, 2):
        if inner_rect not in rects_to_remove \
                and outer_rect not in rects_to_remove \
                and is_rect_inside_rect(inner_rect, outer_rect):
            rects_to_remove.add(outer_rect)

    rects = set(rects).difference(rects_to_remove)

    horizontal_rects = [rect for rect in rects if rect.w > rect.h]

    for rect in horizontal_rects:
        cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (255, 0, 0), 2)

    text_rects = detect_text(image, args.model_path, 0.8, (0.1, 0.05), True)

    for start_x, start_y, w, h in text_rects:
        cv2.rectangle(image, (start_x, start_y), (start_x + w, start_y + h), (0, 0, 255), 2)

    results = []
    for text_rect, rect in itertools.product(text_rects, horizontal_rects):
        if overlap(text_rect, rect) > 0.8 and abs(text_rect.h - rect.h) < 0.5 * rect.h:
            cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)
            results.append({
                'text_rectangle': text_rect.to_json(),
                'rectangle': rect.to_json(),
            })

    print(json.dumps(results, indent=4))

    cv2.imshow("Result", image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
