import sys

import cv2 as cv
import numpy as np

title_trackbar_param1 = 'param1 for HoughCircles'
title_trackbar_param2 = 'param2 for HoughCircles'
title_trackbar_minRadius = 'minRadius for HoughCircles'
title_trackbar_maxRadius = 'maxRadius for HoughCircles'
title_window = 'HoughCircles Demo'

src_org = 0
gray_org = 0


def houghcircles_operation(val):
    src = src_org.copy()
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    param1 = cv.getTrackbarPos(title_trackbar_param1, title_window)
    param2 = cv.getTrackbarPos(title_trackbar_param2, title_window)
    min_radius = cv.getTrackbarPos(title_trackbar_minRadius, title_window)
    max_radius = cv.getTrackbarPos(title_trackbar_maxRadius, title_window)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=param1, param2=param2,
                              minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 255, 255), 3)

    cv.imshow("detected circles", src)


def main(argv):
    filename = argv[0]
    # Loads an image
    global src_org
    src_org = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src_org is None:
        print('Error opening image!')
        return -1
    cv.namedWindow(title_window)
    cv.createTrackbar(title_trackbar_param1, title_window, 100, 255, houghcircles_operation)
    cv.createTrackbar(title_trackbar_param2, title_window, 30, 255, houghcircles_operation)
    cv.createTrackbar(title_trackbar_minRadius, title_window, 1, 127, houghcircles_operation)
    cv.createTrackbar(title_trackbar_maxRadius, title_window, 80, 127, houghcircles_operation)

    houghcircles_operation(0)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
