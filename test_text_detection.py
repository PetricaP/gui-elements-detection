import argparse
import logging

import cv2
import pytesseract

from detection import detect_text
from utils import timer, rectangle

logging.basicConfig(level=logging.INFO)


def apply_ocr_on_rects(image, joined, padding):
    results = []
    padding_x, padding_y = padding

    clone = image.copy()

    image_h, image_w = image.shape[:2]

    for rect in joined:
        start_x, start_y, w, h = rect

        end_x = start_x + w
        end_y = start_y + h

        dx = int(w * padding_x)
        dy = int(h * padding_y)

        start_x = max(0, start_x - dx)
        start_y = max(0, start_y - dy)
        end_x = min(image_w, end_x + dx)
        end_y = min(image_h, end_y + dy)

        roi = image[start_y:end_y, start_x:end_x]

        # config = "-l eng --oem 1 --psm 7"
        config = "-l eng --oem 1"
        text = pytesseract.image_to_string(roi, config=config)

        cv2.rectangle(clone, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

        # cv2.imshow(text, roi)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        results.append((rectangle(start_x, start_y, w, h), text))

    cv2.imshow("OCR padding", clone)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to east model pb file')
    parser.add_argument('image_path', help='image to analyze')
    parser.add_argument('--join_overlapping', action='store_true', help='join overlapping text rectangles')
    parser.add_argument('--min_confidence', type=float, help='minimum confidence for text position', default=0.8)
    parser.add_argument('--join_padding', nargs=2, type=float, help='padding to add when joining rectangles x y',
                        default=[0.1, 0.1])
    parser.add_argument('--ocr_padding', nargs=2, type=float, help='padding to add when applying ocr x y',
                        default=[0.0, 0.0])
    parser.add_argument('--draw_text', action='store_true', help='Draw the detected text near the boxes.')
    parser.add_argument('--apply_ocr', action='store_true', help="Don't apply ocr, just draw the rects")
    args = parser.parse_args()

    original_image = cv2.imread(args.image_path)

    results = detect_text(original_image, args.model_path, args.min_confidence, args.join_padding,
                          args.join_overlapping)

    if args.apply_ocr:
        with timer('Apply OCR on rects'):
            ocr_results = apply_ocr_on_rects(original_image, results, args.ocr_padding)

        ocr_results = sorted(ocr_results, key=lambda r: r[0][1])

        if args.draw_text:
            with timer('Drawing text'):
                for ((start_x, start_y, end_x, end_y), text) in ocr_results:
                    cv2.putText(original_image, text, (start_x, start_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        for (rect, text) in ocr_results:
            print(f'Result at {rect}:')
            print(text)

    with timer('Drawing rectangles'):
        for start_x, start_y, w, h in results:
            cv2.rectangle(original_image, (start_x, start_y), (start_x + w, start_y + h), (0, 0, 255), 2)

    cv2.imshow("Text Detection", original_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
