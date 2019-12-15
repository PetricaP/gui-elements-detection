import argparse
import itertools
import json

import cv2

from detection import detect_rectangles, detect_text, is_checked
from utils import is_inside_circle, point, circle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to .pb file')
    parser.add_argument('image_path', help='Path to image to open')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    rects = detect_rectangles(processed, 50, 0.0)

    squares = [rect for rect in rects if abs(rect.w - rect.h) < 5 and rect.w * rect.h < 200]

    for rect in squares:
        cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (255, 0, 0), 2)

    text_rects = detect_text(image, args.model_path, 0.8, (0.3, 0.05), True)

    for start_x, start_y, w, h in text_rects:
        cv2.rectangle(image, (start_x, start_y), (start_x + w, start_y + h), (0, 0, 255), 2)

    first_results = []
    for rect, square in itertools.product(text_rects, squares):
        if square.y > rect.y and is_inside_circle(point(rect.x, rect.y),
                                                  circle(
                                                      point(square.x + square.w / 2,
                                                            square.y + square.h / 2),
                                                      3 * square.w)):
            cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)
            first_results.append((rect, square))

    results = []
    for rect, square in first_results:
        image_rect = gray[square.y: square.y + square.h,
                          square.x: square.x + square.w]
        checked = is_checked(image_rect, 0.8)

        results.append({
            'rectangle': rect.to_json(),
            'check_button': square.to_json(),
            'is_checked': checked
        })

    print(json.dumps(results, indent=4))

    cv2.imshow("Result", image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
