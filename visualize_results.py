import argparse
import json

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image which generated the results')
    parser.add_argument('results_path', help='Path to generated json file')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    result_image = image.copy()

    with open(args.results_path, 'r') as file:
        results = json.load(file)
        for button in results['buttons']:
            rect = button['rectangle']
            cv2.rectangle(result_image, (rect['x'], rect['y']),
                          (rect['x'] + rect['w'], rect['y'] + rect['h']), (0, 255, 0), 2)

        for button in results['check_buttons']:
            rect = button['rectangle']
            color = (255, 0, 0) if button['is_checked'] else (255, 100, 100)
            cv2.rectangle(result_image, (rect['x'], rect['y']),
                          (rect['x'] + rect['w'], rect['y'] + rect['h']), color, 2)

        for button in results['radial_buttons']:
            circ = button['button_circle']
            color = (0, 100, 255) if button['is_checked'] else (100, 220, 255)
            cv2.circle(result_image, (circ['center']['x'], circ['center']['y']),
                       circ['radius'], color, 2)

    cv2.imshow("Result", result_image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
