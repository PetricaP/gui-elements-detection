import argparse
import json

import cv2

from detection import detect_rectangles, detect_text, join_padded_rectangles, detect_buttons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to .pb file')
    parser.add_argument('image_path', help='Path to image to open')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    rects = detect_rectangles(processed, 50, 0.0)

    net_results = detect_text(image, args.model_path, 0.8)
    text_rects = join_padded_rectangles(net_results, (0.05, 0.05), image.shape[:2])

    results = detect_buttons(image, rects, text_rects)

    print(json.dumps(results, indent=4))

    cv2.imshow("Result", image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
