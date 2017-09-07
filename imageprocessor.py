import datetime
import time
import numpy as np
import cv2
from collections import deque
from scipy.ndimage.measurements import label
from PIL import Image
from keras import backend as K
from utils import img_utils, lane_utils, car_utils, yolo_utils
from utils.lane_utils import Line

##Create an image processor class
class ImageProcessor: 
    
    def __init__(self, img_points, obj_points, mtx, dist, src, dst):
        
        #General Attributes
        self.frame = 0
        #self.log = open(".\\log\\detector_log.txt","w+")
        self.undistort_time = 0
        
        #Attributes for Lane detection

        self.left_line = Line()
        self.right_line = Line()
        self.img_points = img_points
        self.obj_points = obj_points
        self.mtx = mtx
        self.dist = dist
        self.src = src
        self.dst = dst
        self.offset = deque(maxlen=self.left_line.smooth_count)
        self.L_thresh = (215,255)
        self.B_thresh = (150,255)
        self.lane_time = 0
        self.binary_time = 0
        self.draw_time = 0
        self.lane_total = 0

        
        #Common Attributes for vehicle detection
        self.model = None #model to use for predictions
        self.predict_time = 0
        self.box_time = 0
        self.win_time = 0
        self.heat_time = 0
        self.car_total = 0
        
        #Attributes for CNN vehicle detection
        self.xy_window = (80, 80) #size of initial window
        self.window_scale = (1.0, 1.5) #window scales to use
        self.x_start_stop = [[575, 1280], [400, 1280]] #start and stop x-coordinates to search
        self.y_start_stop = [[375, 550], [450, 650]] #start and stop y-coordinates to search
        self.xy_overlap = (0.75, 0.75) #overlap of search windows
        self.pred_threshold = 0.6
        self.smooth_count = 15 # Number of frames to average over
        self.threshold = 9 #threshold for detection
        self.heatmaps_list = deque(maxlen=self.smooth_count) #deque of heatmaps to smooth
        
        #Attributes for YOLO vehicle detection
        self.anchors = []
        self.anchors_path = '.\\model\\PASCAL\\yolov2-voc_anchors.txt'
        self.class_names = []
        self.class_path = '.\\model\\PASCAL\\pascal_classes.txt'
        self.sess = K.get_session()
        
    def lane_detection(self, img):
        self.frame += 1
        z_score1 = 1.9
        tb1 = time.time()
        
        #Create binary
        binary = lane_utils.create_binary(img, self.L_thresh, self.B_thresh)
    
        #Warp Image
        warped_binary = img_utils.warp_image(binary, self.src, self.dst)
        tb2 = time.time()
        self.binary_time += (tb2-tb1)
        tl1 = time.time()
        #Find centroids
        self.left_line.centers, self.right_line.centers = lane_utils.find_centers(warped_binary)
        
        #self.log.write('{} ############# NEXT FRAME ##############\n'.format(datetime.datetime.now()))
        #self.log.write('{} FRAME {:d} - # of LEFT centers is: {:d}\n'.format(datetime.datetime.now(), 
        #                                                                     self.frame, self.left_line.centers.shape[0]))
        #self.log.write('{} FRAME {:d} - # of RIGHT centers is: {:d}\n'.format(datetime.datetime.now(), 
        #                                                                      self.frame, self.right_line.centers.shape[0]))
        #Get line co-efficients if more than three centers are found for each line:
        if (self.left_line.centers.shape[0] > 2 ) & (self.right_line.centers.shape[0] > 2):
            
            #self.log.write('{} ENTERED IF BLOCK FOR FRAME: {}\n'.format(datetime.datetime.now(), self.frame))
            #self.log.write('{} Left centers shape: {}\n'.format(datetime.datetime.now(), self.left_line.centers.shape[0]))
            #self.log.write('{} Right centers shape: {}\n'.format(datetime.datetime.now(), self.right_line.centers.shape[0]))
            #self.log.write('{} Left centers are: {}\n'.format(datetime.datetime.now(), self.left_line.centers))
            #self.log.write('{} Right centers are: {}\n'.format(datetime.datetime.now(), self.right_line.centers))
            
            self.left_line.centers, self.right_line.centers = lane_utils.check_centers(self.left_line.centers, 
                                                                                       self.right_line.centers, z_score1)
            #self.log.write('{} CHECK CENTERS CALLED\n'.format(datetime.datetime.now()))
            #self.log.write('{} Left centers shape: {}\n'.format(datetime.datetime.now(), self.left_line.centers.shape[0]))
            #self.log.write('{} Right centers shape: {}\n'.format(datetime.datetime.now(), self.right_line.centers.shape[0]))
        
            #self.log.write('{} Getting Left Line Co-efficients\n'.format(datetime.datetime.now()))
            self.left_line.coeff = lane_utils.get_coeff(self.left_line.centers)
            #self.log.write('{} Getting Right Line Co-efficients\n'.format(datetime.datetime.now()))
            self.right_line.coeff = lane_utils.get_coeff(self.right_line.centers)
            #self.log.write('{} GOT CO-EFFICIENTS\n'.format(datetime.datetime.now()))
            self.left_line.coeff_list.append(self.left_line.coeff)
            self.right_line.coeff_list.append(self.right_line.coeff)
            #self.log.write('{} CO-EFFICIENTS APPENDED\n'.format(datetime.datetime.now()))

            #Measure ROC and Offset
            l_ROC, r_ROC, current_offset = lane_utils.get_ROC_offset(img, self.left_line.centers, 
                                                                                        self.right_line.centers)
            self.left_line.ROC.append(l_ROC)
            self.right_line.ROC.append(r_ROC)
            self.offset.append(current_offset)
            
        avg_ROC = (np.mean(self.left_line.ROC) + np.mean(self.right_line.ROC))/2
        avg_offset = np.mean(self.offset)
        tl2 = time.time()
        self.lane_time += (tl2 - tl1)
        #TO-DO: Sanity checks
        #Check that ROCs for left_line & right_line in this frame are within a threshold of ROC of average ROC
        #Check that left_line and right_line are equidistant in warped_binary at bottom, middle & top of frame
        #If sanity checks fail, remove the latest appended offset, ROC and co-efficients
        
        td1 = time.time()
        #Draw Lines & measure RoC
        warped_img = img_utils.warp_image(img, self.src, self.dst)
        draw_img = lane_utils.draw_lines(warped_img, self.left_line.best_fit(), self.right_line.best_fit())
        
        #Unwarp Image & add to original
        ##diag_img_size = 250
        ##stacked_warped_binary = np.dstack((warped_binary*255, warped_binary*255, warped_binary*255)).astype(np.uint8)
        unwarped_draw_img = img_utils.warp_image(draw_img, self.dst, self.src)
        output_img = cv2.addWeighted(img, 1, unwarped_draw_img, 0.5, 0)
        ##diag_img = cv2.resize(cv2.addWeighted(stacked_warped_binary, 1, draw_img, 0.9, 0), 
        ##                      (diag_img_size, diag_img_size), interpolation=cv2.INTER_AREA) 
        ##output_img[:diag_img_size,(1280-diag_img_size):1280,:] = diag_img
        
        #Add text to image
        cv2.putText(output_img,"Frame: {:d}".format(self.frame), (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(output_img,"Offset: {:0.2f} m.".format(avg_offset), (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,0,0), 2, cv2.LINE_AA)
        if avg_ROC>4000:
            cv2.putText(output_img,"RoC: Straight", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255,0,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(output_img,"RoC: {:0.2f} m.".format(avg_ROC), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255,0,0), 2, cv2.LINE_AA)
        
        #self.log.write('{} COMPLETED PROCESSING FRAME\n'.format(datetime.datetime.now()))
        td2 = time.time()
        self.draw_time += (td2 - td1)
        
        return output_img
        
    def vehicle_detection_NN(self,img):
        windows = []
        detection_windows = []
        trans_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        
        tw1 = time.time()
        for i, scale in enumerate(self.window_scale):
        #Generate (x,y) coordinates for all windows that will be used
            windows.extend(car_utils.slide_window(trans_img, 
                                                  x_start_stop=self.x_start_stop[i], y_start_stop=self.y_start_stop[i], 
                                                  xy_window=[int(dim*scale) for dim in self.xy_window], 
                                                  xy_overlap=self.xy_overlap))
        windows = np.asarray(windows)
        #Generate views based on the same parameters as the slide_windows() function
        views = car_utils.create_views(trans_img, self.xy_window, self.xy_overlap, self.x_start_stop, self.y_start_stop, 
                                       self.window_scale)
        
        tw2 = time.time()
        self.win_time += (tw2 - tw1)
        
        #Get predictions on all the views and reshape for boolean masking
        tp1 = time.time()
        predictions = car_utils.search_windows(views, self.model)
        tp2 = time.time()
        self.predict_time += (tp2 - tp1)
        
        th1 = time.time()
        #If detections are found, append the detected windows if the probability is greater than a threshold
        if len(predictions[predictions>=self.pred_threshold]) > 0:
            detection_windows.extend(windows[predictions>=self.pred_threshold])

        heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        heatmap = car_utils.add_heat(heatmap ,detection_windows)

        #####Use for running on images####
        #heatmap = car_utils.apply_threshold(heatmap, self.threshold)
        #labels = label(heatmap)

        ###Use for running on videos####
        self.heatmaps_list.append(heatmap)
        smooth_heatmap = car_utils.sum_heatmap(self.heatmaps_list, self.threshold)
        labels = label(smooth_heatmap)
        th2 = time.time()
        self.heat_time += (th2 - th1)
        
        return labels
    
    def lane_and_vehicle_detection_NN(self,img):
        #Undistort the image
        total1 = time.time()
        tu1 = time.time()
        img = img_utils.undistort(img, self.img_points, self.obj_points, self.mtx, self.dist)
        tu2 = time.time()
        self.undistort_time += (tu2 - tu1)
        lane_img = self.lane_detection(img)
        total2 = time.time()
        self.lane_total += (total2 - total1)
        
        ctotal1 = time.time()
        cars = self.vehicle_detection_NN(img)
        tdraw1 = time.time()
        output_frame = img_utils.draw_labeled_bboxes(lane_img, cars)
        cv2.putText(output_frame,"Cars Detected: {:d}".format(cars[1]), (50,70), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,0,0), 2, cv2.LINE_AA)
        tdraw2 = time.time()
        self.box_time += (tdraw2 - tdraw1)
        ctotal2 = time.time()
        self.car_total += (ctotal2 - ctotal1)
		
        return output_frame

    def vehicle_detection_YOLO(self,img):
        #Read in the class names and anchor locations
        self.class_names = yolo_utils.read_class_names(self.class_path)
        self.anchors = yolo_utils.read_anchors(self.anchors_path)
        
        #Pre-process the incoming frame
        resized_img = cv2.resize(img,(416,416))
        batch_img = resized_img/255.
        batch_img = np.expand_dims(batch_img, axis=0)
        
        #Perform the prediction
        n_classes = len(self.class_names)
        out_boxes, out_scores, out_classes = yolo_utils.YOLO_predict(img, batch_img, self.sess, self.model, self.anchors, n_classes)
        
        return out_boxes, out_scores, out_classes
    
    def lane_and_vehicle_detection_YOLO(self, img):
        
        #Undistort the image
        img = img_utils.undistort(img, self.img_points, self.obj_points, self.mtx, self.dist)
        #Perform lane detection on image
        lane_img = self.lane_detection(img)
        out_boxes, out_scores, out_classes = self.vehicle_detection_YOLO(img)
        
        #Populate variables for drawing YOLO boxes
        colors = yolo_utils.generate_colors(img, self.class_names)
        
        #Convert image to format expected by PIL
        lane_img_PIL = Image.fromarray(lane_img)
        
        #Draw YOLO boxes on lane image
        output_frame = yolo_utils.draw_YOLO_boxes(lane_img_PIL, out_boxes, out_classes, out_scores, self.class_names, colors)
        output_frame = np.array(output_frame)
        cv2.putText(output_frame,"Cars Detected: {:d}".format(len(out_boxes)), (50,70), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255,0,0), 2, cv2.LINE_AA)
        
        return output_frame