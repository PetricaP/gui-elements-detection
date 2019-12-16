import argparse
import json

import cv2

from detection import detect_text, join_padded_rectangles, detect_radial_buttons

RADIAL_BUTTON_PARAM1 = 14
RADIAL_BUTTON_PARAM2 = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the .pb file')
    parser.add_argument('image_path', help='Path to image to find buttons in.')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    net_results = detect_text(image, args.model_path, 0.8)
    text_rects = join_padded_rectangles(net_results, (0.4, 0.1), image.shape[:2])

    # for start_x, start_y, w, h in text_rects:
        # cv2.rectangle(image, (start_x, start_y), (start_x + w, start_y + h), (0, 0, 0), 2)

    results = detect_radial_buttons(gray, text_rects)

    for button in results:
        circ = button['button_circle']
        cv2.circle(image, (circ['center']['x'], circ['center']['y']), circ['radius'], (0, 0, 255), 2)

    print(json.dumps(results, indent=4))

    cv2.imshow("Result", image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
