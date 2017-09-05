# -*- coding: utf-8 -*-
#Functions for sobel gradient operations

import cv2
import numpy as np

## Sobel Thresholding functions
def cvtThresholdBinary(img, threshold):
    img  = np.uint8(255*img/np.max(img))
    binary_img = np.zeros_like(img)
    binary_img[(img >= threshold[0]) & (img <= threshold[1])] = 1
    return binary_img

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        print("ERROR: Incorrect gradient orientation selected. Grayscale image returned")
        sobel = gray
    abs_sobel = np.absolute(sobel)
    binary_img = cvtThresholdBinary(abs_sobel, thresh)
    return binary_img

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    binary_img = cvtThresholdBinary(abs_sobel, mag_thresh)
    return binary_img

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_grad = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_img = np.zeros_like(dir_grad)
    binary_img[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return binary_img