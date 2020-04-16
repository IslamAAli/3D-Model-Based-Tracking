#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:36:57 2020

@author: junaid_ia
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cereal.jpg')
img0=img
gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
img = cv2.GaussianBlur(gray,(3,3),0)
edges = cv2.Canny(img,200,100,apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 120, None, 0, 0)
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 180, None, 120, 20)
# remove noise
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)  # y
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(edges,cmap = 'gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])
for i in range(lines.shape[0]):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv2.line(img0,(x1,y1),(x2,y2),(255,255,0),2)
cv2.imwrite('houghlines.jpg',img0)
img1=cv2.imread('cereal.jpg')
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(img1, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2)
    cv2.imwrite('LinesParam.jpg',img1)
plt.show()