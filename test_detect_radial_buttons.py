import argparse
import cv2
from detection import detect_circles


RADIAL_BUTTON_PARAM1 = 14
RADIAL_BUTTON_PARAM2 = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image to find buttons in.')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = detect_circles(gray, 5, 7, 10, 15)

    for circ in results:
        # circle outline
        cv2.circle(image, circ.center, circ.radius, (255, 0, 255), 3)
    cv2.imshow("detected circles", image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
