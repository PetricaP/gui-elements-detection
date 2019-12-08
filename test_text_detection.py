import argparse
import logging

import cv2
import pytesseract

from detection import detect_text
from utils import timer

logging.basicConfig(level=logging.INFO)


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

    image, results = detect_text(original_image, args.model_path, args.min_confidence, args.padding,
                                 args.join_overlapping)

    with timer('Drawing rectangles'):
        for start_x, start_y, w, h in results:
            cv2.rectangle(original_image, (start_x, start_y), (start_x + w, start_y + h), (0, 0, 255), 2)

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
