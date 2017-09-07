# -*- coding: utf-8 -*-
#Utilities and classes for lane finding
import cv2
import numpy as np
import datetime

from collections import deque

import utils.color_utils as color_utils
#import utils.grad_utils as grad_utils
import utils.img_utils as img_utils


#Line Class
class Line():
    def __init__(self):
        self.centers = [] #array of center points from current fit
        self.coeff = [] #co-efficients from latest center points
        self.smooth_count= 10 # number of frames to smooth over
        self.coeff_list = deque(maxlen=self.smooth_count) #deque to hold co-efficients to average
        self.base = None #base point for latest frame
        self.ROC = deque(maxlen=self.smooth_count)
    
    def best_fit(self):
        return np.mean(self.coeff_list, 0)
        
#Scan an image slice
def scan_slice(img_slice, conv_win):
    
    image_mid = int(img_slice.shape[0]/2) #used to scan the left and right halfs of the slice separately
    l_conv_output = np.convolve(conv_win, img_slice[:image_mid]) 
    l_center = np.argmax(l_conv_output)
    #Drop the result if the max value is less than a threshold
    if l_conv_output[l_center] < 10:
        l_center = 0
    #print("Raw slice left center is:", l_center)
    r_conv_output = np.convolve(conv_win, img_slice[int(5*image_mid/4):]) #change image-mid to 800
    r_center = np.argmax(r_conv_output)
    #Drop the result if the max value is less than a threshold
    if r_conv_output[r_center] < 10:
        r_center = 0
    #print("Raw slice right center is:", r_center)
    return l_center, r_center


#Scan a margin around previously known good centers
def scan_margin(img_slice, conv_win, margin, xl, xr):
    offset = int(len(conv_win)/2)
    conv_output = np.convolve(conv_win, img_slice) 
    #Initialize centers to zero in case both input values are zero
    l_center = 0
    r_center = 0
    #Perform margin scan only if input values are non-zero
    if xl != 0:
        #Define boundaries for left line scan & limit left line from 0 to middle of slice
        l_min_index = int(max(xl+offset-margin,0))
        l_max_index = int(min(xl+offset+margin,img_slice.shape[0]/2))
        l_center = np.argmax(conv_output[l_min_index:l_max_index])+l_min_index
        #Set returned values to 0 if nothing is found in margin scan to trigger a slice scan
        if l_center == l_min_index:
            l_center = 0
        #print("Raw margin left center is:", l_center)
    
    if xr != 0:
        #Define boundaries for right line scan and limit right line from middle to end of slice 
        r_min_index = int(max(xr+offset-margin,5*img_slice.shape[0]/8)) #Change 1/2 to 5/8 (640 to 800)
        r_max_index = int(min(xr+offset+margin,img_slice.shape[0]))
        r_center = np.argmax(conv_output[r_min_index:r_max_index])+r_min_index
        #Set returned values to 0 if nothing is found in margin scan to trigger a slice scan
        if r_center == r_min_index:
            r_center = 0
        #print("Raw margin right center is:", r_center)
    
    return l_center, r_center

def find_centers(img):
    #TO-DO: Use previous base points as starting point   
    n_windows = 10 #should divide 720 evenly - 6, 8, 9, 12 - # of slices into which the image will be split 
    window_height = img.shape[0]/n_windows #height of the convolution window/image slice
    image_mid = int(img.shape[1]/2)
    window_width = 45 #width of the convolution window
    offset = window_width/2 #used to re-position the center after performing the convolution
    margin = 50 #margin around which to scan for lane lines

    l_centers = [] #left lane line center points [xl, y] are stored from the bottom to the top of an image
    r_centers = [] #right lane line center points [xr, y] are stored from the bottom to the top of an image
    conv_win = np.ones(window_width) #define the convolution window
    
    #Find the centers starting at the bottom of the image to use as a starting point
    image_base_slice = np.sum(img[int((n_windows - 1)*img.shape[0]//n_windows):
                                  int((n_windows)*img.shape[0]//n_windows),:], axis=0)
    
    #Find base x-coordinates and append only if a peak is found
    xl, xr = scan_slice(image_base_slice, conv_win)
    y = img.shape[0] - int(window_height/2)
    if (xl != 0):
        xl = xl - offset
        l_centers.append([xl, y])
        #print("Left base point appended")
        
    if (xr != 0):
        xr = xr - offset + image_mid + 160 #add 160 to compensate for change from 640 -> 800
        r_centers.append([xr, y])
        #print("Right base point appended")
        
    #Iterate over the levels moving up from the bottom of the image
    for level in np.arange(1, n_windows):
        image_slice = np.sum(img[int((n_windows - level - 1)*img.shape[0]//n_windows):
                                  int((n_windows - level)*img.shape[0]//n_windows),:], axis=0)
        
        #Scan around a margin to find the center point for the next level
        y = int((n_windows - level - 1)*img.shape[0]//n_windows + window_height/2)
        xl, xr = scan_margin(image_slice, conv_win, margin, xl, xr)
        #Do a full slice scan if margin scan doesn't return any values
        if (xl == 0) | (xr == 0):
            #print("Margin scan failed. Scanning slice.", xl, xr)
            xl, xr = scan_slice(image_slice, conv_win)
            
            #Append points only if a peak is found
            if xl != 0:
                xl = xl - offset
                l_centers.append([xl, y])
                #print("Slice left point {:d} appended".format(level))

            if xr != 0:
                xr = xr - offset + image_mid + 160
                r_centers.append([xr, y])
                #print("Slice right point {:d} appended".format(level))
        else:
            xl = xl - offset
            xr = xr - offset #+ image_mid
            l_centers.append([xl, y])
            r_centers.append([xr, y])
            #print("Margin left & right points {:d} appended".format(level))
            
    return np.asarray(l_centers, dtype=np.int32), np.asarray(r_centers, dtype=np.int32)

def get_coeff(centers, deg=2):
    #log.write('{} Y-values are {}\n'.format(datetime.datetime.now(),centers[:,1]))
    #log.write('{} X-values are {}\n'.format(datetime.datetime.now(),centers[:,0]))
    if centers.shape[0]>3:
        coeff = np.polyfit(centers[:,1], centers[:,0], deg)
    else:
        coeff = np.polyfit(centers[:,1], centers[:,0], 1)
        coeff = np.insert(coeff, 0, 0)
    #log.write('{} Co-efficients are {}\n'.format(datetime.datetime.now(),coeff))
    return coeff

def get_ROC_offset(img, l_centers, r_centers, deg=2):
    
    #Convert from pixel space to real space
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = img.shape[0] #Evaluate the ROC and offset at the bottom of the image
    
    # Fit new polynomials to x,y in world space
    left_line_cr = np.polyfit(l_centers[:,1]*ym_per_pix, l_centers[:,0]*xm_per_pix, 2)
    right_line_cr = np.polyfit(r_centers[:,1]*ym_per_pix, r_centers[:,0]*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_ROC = ((1 + (2*left_line_cr[0]*y_eval*ym_per_pix + left_line_cr[1])**2)**1.5) / np.absolute(2*left_line_cr[0])
    right_ROC = ((1 + (2*right_line_cr[0]*y_eval*ym_per_pix + right_line_cr[1])**2)**1.5) / np.absolute(2*right_line_cr[0])
    
    #Measure offset
    x_left = left_line_cr[0]*(y_eval*ym_per_pix)**deg + left_line_cr[1]*y_eval*ym_per_pix + left_line_cr[2]
    x_right = right_line_cr[0]*(y_eval*ym_per_pix)**deg + right_line_cr[1]*y_eval*ym_per_pix + right_line_cr[2]
    lane_center = (x_left + x_right)/2
    image_center = int(img.shape[1]/2)*xm_per_pix
    offset =  (lane_center - image_center)
    
    return left_ROC, right_ROC, offset

def draw_lines(img, l_coeff, r_coeff, deg=2):
    
    draw_img = np.zeros_like(img)
    
    #Calculate the line points for each pixel
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_line = l_coeff[0]*ploty**deg + l_coeff[1]*ploty + l_coeff[2]
    right_line = r_coeff[0]*ploty**deg + r_coeff[1]*ploty + r_coeff[2]
    
    #Reshape the arrays into (x,y) coordinates that can be used by cv2 functions
    ploty = ploty.reshape((len(ploty),1))
    left_line = left_line.reshape((len(ploty),1))
    right_line = right_line.reshape((len(ploty),1))
    left_pts = np.concatenate([left_line, ploty], axis = 1)
    right_pts = np.concatenate([right_line, ploty], axis = 1)
    
    #Draw lane boundaries and region on original image
    poly_pts = np.concatenate([left_pts, right_pts[::-1]], axis = 0)
    cv2.polylines(draw_img, np.int32([left_pts]), 0, color = (255,0,0), thickness=20, lineType=4)
    cv2.polylines(draw_img, np.int32([right_pts]), 0, color = (0,0,255), thickness=20, lineType=4)
    cv2.fillPoly(draw_img, np.int32([poly_pts]), color = (0,100,0), lineType=8)
    
    return draw_img

def check_centers(l_centers, r_centers, z_score):
    l_mu = np.mean(l_centers[:,0])
    r_mu = np.mean(r_centers[:,0])
    l_sig = np.std(l_centers[:,0])
    r_sig = np.std(r_centers[:,0])
    
    lz_score = np.abs(l_centers[:,0] - l_mu)/l_sig
    rz_score = np.abs(r_centers[:,0] - r_mu)/r_sig
    
    return l_centers[lz_score<z_score], r_centers[rz_score<z_score]

def create_binary(img, L_thresh = (215, 255), B_thresh = (135, 255)):
        
    #Use L-channel to select white pixels
    W_binary = color_utils.LAB_threshold(img, 'L', L_thresh)
    #Use B-channel to select yellow pixels
    Y_binary = color_utils.LAB_threshold(img, 'B', B_thresh) 
    
    output = color_utils.OR_binaries(W_binary, Y_binary)
    
    return output