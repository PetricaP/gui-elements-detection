import argparse
import cv2
import numpy

title_trackbar_c = 'C constant for threshold'
title_trackbar_block_size = 'Block size:'
title_window = 'Thresholding Demo'


def threshold_operations(_):
    c = cv2.getTrackbarPos(title_trackbar_c, title_window)
    block_size = cv2.getTrackbarPos(title_trackbar_block_size, title_window)
    print("[c, block_size] = [{c}, {block_size}]".format(c=c, block_size=2*block_size + 1))
    dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2 * block_size+1, c)
    dst2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 2 * block_size + 1, c)
    images = numpy.hstack((gray, dst, dst2))
    cv2.imshow(title_window, images)


parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Path to image to analyze')
args = parser.parse_args()

original = cv2.imread(args.image_path)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

cv2.namedWindow(title_window)
cv2.createTrackbar(title_trackbar_c, title_window, 2, 255, threshold_operations)
cv2.createTrackbar(title_trackbar_block_size, title_window, 1, 127, threshold_operations)

threshold_operations(0)


cv2.waitKey()
cv2.destroyAllWindows()


