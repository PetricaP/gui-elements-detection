import argparse
import json

import cv2

from detection import detect_rectangles, detect_text, join_padded_rectangles, detect_buttons, detect_check_buttons, \
    detect_radial_buttons, apply_ocr_on_rectangle
from utils import rectangle


def analyze_image(image, model_path):
    net_results = detect_text(image, model_path, 0.1)

    text_rects = join_padded_rectangles(net_results, (0.05, 0.05), image.shape[:2])

    results = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    rectangles = detect_rectangles(processed, 50, 0.0)

    buttons = detect_buttons(image, rectangles, text_rects)
    results['buttons'] = buttons

    check_buttons = detect_check_buttons(gray, rectangles, text_rects)
    results['check_buttons'] = check_buttons

    radial_buttons = detect_radial_buttons(gray, text_rects)
    results['radial_buttons'] = radial_buttons

    processed_text_rects = []
    for button in results['buttons']:
        text_rect = rectangle.from_json(button['rectangle'])

        processed_text_rects.append(text_rect)
        text = apply_ocr_on_rectangle(image, text_rect, (0.0, 0.0))

        button['text'] = text

    for check_button in results['check_buttons']:
        text_rect = check_button['associated_text_rect']

        if text_rect:
            text_rect = rectangle.from_json(text_rect)

            processed_text_rects.append(text_rect)
            text = apply_ocr_on_rectangle(image, text_rect, (0.1, 0.1))
        else:
            text = None

        check_button['text'] = text

    for radial_button in radial_buttons:
        text_rect = rectangle.from_json(radial_button['associated_text_rect'])

        processed_text_rects.append(text_rect)
        text = apply_ocr_on_rectangle(image, text_rect, (0.1, 0.1))

        radial_button['text'] = text

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the pb file for the neural network')
    parser.add_argument('image_path', help='Path to the image to analyze')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    results = analyze_image(image, args.model_path)

    with open('results.json', 'w') as file:
        file.write(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
