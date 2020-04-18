import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('0001.png')

sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
sobel_8u = cv2.cvtColor(sobel_8u, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(sobel_8u,100,255,cv2.THRESH_BINARY)
plt.subplot(2,2,1),plt.imshow(thresh1,cmap = 'gray')
plt.title('Sobel, 1, 0, ksize=5')

sobelx64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
sobel_8u = cv2.cvtColor(sobel_8u, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(sobel_8u,100,255,cv2.THRESH_BINARY)
plt.subplot(2,2,2),plt.imshow(thresh1,cmap = 'gray')
plt.title('Sobel, 0, 1, ksize=5')

sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
sobel_8u = cv2.cvtColor(sobel_8u, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(sobel_8u,100,255,cv2.THRESH_BINARY)
plt.subplot(2,2,3),plt.imshow(thresh1,cmap = 'gray')
plt.title('Sobel, 1, 0, ksize=3')

sobelx64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
sobel_8u = cv2.cvtColor(sobel_8u, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(sobel_8u,100,255,cv2.THRESH_BINARY)
plt.subplot(2,2,4),plt.imshow(thresh1,cmap = 'gray')
plt.title('Sobel, 0, 1, ksize=3')

plt.show()