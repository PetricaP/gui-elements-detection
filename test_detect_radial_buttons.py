import argparse
import cv2
from detection import detect_circles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image to find buttons in.')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    results = detect_circles(gray, 2, 50)

    for circ in results:
        # circle center
        cv2.circle(image, circ.center, 1, (0, 100, 100), 3)
        # circle outline
        cv2.circle(image, circ.center, circ.radius, (255, 255, 255), 3)
    cv2.imshow("detected circles", image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
