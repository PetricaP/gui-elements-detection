import argparse

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    args = parser.parse_args()

    original = cv2.imread(args.image_path)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU);

    cv2.imshow('Original', original)
    cv2.imshow('Gray', gray)
    cv2.imshow('Gaussian', gaussian)
    cv2.imshow('Otsu', otsu)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
