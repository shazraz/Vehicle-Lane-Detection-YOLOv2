# Vehicle & Lane Detection

## Overview
This repository contains a combined pipeline for lane finding and vehicle detection. The lane finding algorithm is based off the Advanced Lane Lines project done for Udacity's SDC Term 1 but improved with better thresholding techniques and smoothing techniques. The vehicle detection portion compares LeNet-5 to YOLOv2. The YOLOv2 model was built using a modified version of the YAD2K project to change the Keras calls to v1.2.1 for compatibility with the SDC Term 1 conda environment. This is only tested for converting the DarkNet models to Keras models, re-training of models has not been tested. Complete write-up [here](https://medium.com/@raza.shahzad/vehicle-detection-lane-finding-using-opencv-lenet-5-2-2-cfc4fea330b4).

## Usage

### Creating YOLOv2 Keras models
1. Download the configuration files & weights for YOLOv2 trained on the COCO and PASCAL data sets from [here](https://pjreddie.com/darknet/yolo/). 

1. Use ```yad2k.py``` in the YAD2K folder to convert the darknet cfg files and weights to a Keras 1.2.1 ```.h5``` model. This will also generate anchor files. 

1. Use ```test_yolo.py``` in the YAD2K folder to test a directory of images (use test_images). Specify the path for the appropriate anchor file and classes (VOC or COCO). 

### Processing the videos
1. Launch the jupyter notebook, select the ROI src/dst points & color thresholds for the appropriate video.

1. Use either the saved LeNet-5 model or the generated YOLOv2 models and instantiate the appropriate ImageProcessor object. Be sure to specify the correct anchor and classes path depending on whether the model trained on the PASCAL VOC dataset or the COCO dateset is being used.

1. Run the appropriate code block to instantiate the ImageProcessor object and call the correct method (lane_and_vehicle_detection_NN for LeNet-5 and lane_and_vehicle_detection_YOLO for YOLOv2) to process the video images.

## Notes

1. The lane finding algorithm uses thresholding in the LAB colorspace only instead of multiple HSV, RGB and Sobel gradient thresholds [previously](https://github.com/shazraz/Advanced-Lane-Finding) used to obtain a combined binary. Ipywidgets were used to determine the appropriate thresholding parameters for each video (shout out to [Ed Voas](https://medium.com/@edvoas/advanced-lane-finding-a4bb8356824d) for the idea.)

1. Various helper functions are located in the utils folder. These include a Line class in ```lane_utils.py``` to implement smoothing over multiple video frames. The number of frames can be set using the ```Line.smooth_count``` attribute.

1. Several improvements can be made to the lane finding algorithm that have not been implemented since the algorithm works well on both the project and challenge videos. These include:
    - Re-using the last best polynomial fit to identify a region in the image that should be checked for lane lines
    - Checking whether the lines are equidistant at the bottom, center and top of the warped binary image.
    - Checking the ROC of each Line object against it's average ROC in previous frames
    - Using the line co-efficients as part of the ```Line.coeff_list``` deque only if the checks above pass 
  

