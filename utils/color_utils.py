# -*- coding: utf-8 -*-
#Functions for color transformations of images

import cv2
import numpy as np

def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Wrapper for BGR to Gray conversion
def BGR2GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def BGR2YUV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

#Wrapper for converting to HLS
def BGR2HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

#Wrapper for converting to HSV
def BGR2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Wrapper for RGB to Gray conversion
def RGB2GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Wrapper for RGB to Gray conversion for set of images
def RGB2GRAY_set(img_set):
    gray_img_set = []
    for img in img_set:
            gray_img_set.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return gray_img_set

#Wrapper for converting to HLS
def RGB2HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

#Wrapper for converting to HSV
def RGB2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def RGB2YUV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

#Wrapper for converting Gray to RGB
def GRAY2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

#Perform histogram equalization using CLAHE algorithm on grayscale images
def clahe_equalize_set(img_set, climit=2.0, gridsize=(4,4)):
    clahe_img_set = []
    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=gridsize)
    for img in img_set:
        clahe_img_set.append(np.expand_dims(clahe.apply(img), axis = 2))
    return clahe_img_set

#Perform histogram equalization using CLAHE algorithm on single grayscale image
def clahe_equalize(img, climit=2.0, gridsize=(4,4)):
    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=gridsize)
    img = clahe.apply(img)
    return img

#Perform histogram equalization using CLAHE algorithm on RGB->YUV->RGB images
def clahe_equalize_RGB_set(img_set, climit=2.0, gridsize=(4,4)):
    clahe_RGB_img_set = []
    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=gridsize)
    for img in img_set:
        y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
        y = clahe.apply(y)
        img = cv2.merge((y,u,v))
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        clahe_RGB_img_set.append(img)
    return clahe_RGB_img_set

#References:
#https://medium.com/towards-data-science/
#robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3
def white_selector(img):
    #img = RGB2HLS(img)
    #lower_white_threshold = np.array([0,200,0], dtype = np.uint8)
    #upper_white_threshold = np.array([255,255,255], dtype = np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_white_threshold = np.array([0,0,200], dtype = np.uint8)
    upper_white_threshold = np.array([255,30,255], dtype = np.uint8)
    
    binary_img = cv2.inRange(img, lower_white_threshold, upper_white_threshold)
    return binary_img/255

#References:
#http://aishack.in/tutorials/tracking-colored-objects-opencv/
#https://medium.com/@royhuang_87663/how-to-find-threshold-f05f6b697a00
def yellow_selector(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #img = RGB2HLS(img)
    #lower_yellow_threshold = np.uint8([ 10,   0, 100])
    #upper_yellow_threshold = np.uint8([ 40, 255, 255])
    lower_yellow_threshold = np.asarray([10, 100, 100]) #HSV threshold S 20-70%
    upper_yellow_threshold = np.asarray([50, 255, 255]) #HSV threshold
    binary_img = cv2.inRange(img, lower_yellow_threshold, upper_yellow_threshold)
    #binary_img = cv2.bitwise_and(img, img, mask=mask)
    return binary_img/255

def OR_binaries(img1, img2):
    new_img = np.zeros_like(img1)
    new_img[(img1 == 1) | (img2 == 1)] = 1
    return new_img

#AND two binary images
def AND_binaries(img1, img2):
    new_img = np.zeros_like(img1)
    new_img[(img1 == 1) & (img2 == 1)] = 1
    return new_img

def HLS_threshold(img, channel='S', thresh=(0,255)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H,L,S = cv2.split(img)
    binary_img = np.zeros_like(S)
    if channel == 'S':
        binary_img[(S > thresh[0]) & (S <= thresh[1])] = 1
    elif channel == 'H':
        binary_img[(H > thresh[0]) & (H <= thresh[1])] = 1
    elif channel == 'L':
        binary_img[(L > thresh[0]) & (L <= thresh[1])] = 1
    else:
        print("ERROR: Incorrect channel selected, empty image returned")
    return binary_img

def LAB_threshold(img, channel='A', thresh=(0,255)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L,A,B = cv2.split(img)
    binary_img = np.zeros_like(L)
    if channel == 'A':
        binary_img[(A > thresh[0]) & (A <= thresh[1])] = 1
    elif channel == 'B':
        binary_img[(B > thresh[0]) & (B <= thresh[1])] = 1
    elif channel == 'L':
        binary_img[(L > thresh[0]) & (L <= thresh[1])] = 1
    else:
        print("ERROR: Incorrect channel selected, empty image returned:", channel )
    return binary_img