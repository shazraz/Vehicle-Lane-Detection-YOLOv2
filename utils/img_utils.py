# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.image as mpimg

##Read a set of training images (1:1 aspect ratio) in a specific color-space given a list of file paths
def read_training_images(files, size, color_space='RGB'):
    # Create aarray to hold images
    n_images = len(files)
    if color_space == 'GRAY':
        images = np.empty([n_images, size, size], dtype = np.uint8)
    else:
        images = np.empty([n_images, size, size, 3], dtype = np.uint8)
    # Iterate through the list of images
    for idx, file in enumerate(files):
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif color_space == 'GRAY':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        images[idx] = feature_image
    return images

##Drawing functions
def draw_ROI(img, vertices, color = (0,255,0), thickness=3 ):
    return cv2.polylines(img, np.int32([vertices]), 1, color = color, thickness=thickness, lineType=4)

#For use with labeled heatmaps
def draw_labeled_bboxes(img, labels):
    draw_img = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (100,255,0), 3)
    # Return the image
    return draw_img

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#Transform functions
#Resize set of images to (32x32) for input into the neural network
def resize_set(img_set, size):
    resized_img_set = []
    for img in img_set:
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA )
        resized_img_set.append(resized)
    return np.asarray(resized_img_set, dtype = np.uint8)


#Warp the image to get a plan view
def warp_image(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
    return img

def blur_gradient(img, rad=5):
    return cv2.GaussianBlur(img,(rad,rad),0)    
    
#Undistort an image
def undistort(img, img_points, obj_points, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def cal_camera(cal_img_path, board_size):
    img_points = [] # 2D points in image
    obj_points = [] # 3D points in real-world space
    
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2)
    
    #Read in all images and append to image points and object points arrays
    for image_path in cal_img_path:
        img = mpimg.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (board_size[0],board_size[1]), None)
        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return img_points, obj_points, mtx, dist

