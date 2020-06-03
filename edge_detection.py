import cv2
import numpy as np


# edge detection using Sobel and Thresholding of results
def detect_edges_sobel(img):
    # blr=cv2.GaussianBlur(img,(5,5),0)
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    sobel_8u = cv2.cvtColor(sobel_8u, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(sobel_8u, 35, 255, cv2.THRESH_BINARY)

    return thresh1