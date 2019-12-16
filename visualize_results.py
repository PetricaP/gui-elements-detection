import argparse
import json

import cv2


def visualize_results(image_path, results_path):
    result_image = cv2.imread(image_path)

    with open(results_path, 'r') as file:
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

    return result_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image which generated the results')
    parser.add_argument('results_path', help='Path to generated json file')
    parser.add_argument('--save', help='Save the resulting image to this path')
    args = parser.parse_args()

    result_image = visualize_results(args.image_path, args.results_path)

    cv2.imshow("Result", result_image)
    cv2.waitKey()

    if args.save:
        cv2.imwrite(args.save, result_image)


if __name__ == '__main__':
    main()
