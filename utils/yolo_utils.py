# Functions for YOLO predictions and drawings
import numpy as np
import colorsys
import random
from keras import backend as K
from PIL import ImageDraw, ImageFont
from YAD2K.yad2k.models.keras_yolo import yolo_eval, yolo_head


#Read Anchors and Classes
def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def read_class_names(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def generate_colors(img, class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def YOLO_predict(img, batch_img, sess, model, anchors, n_classes):
    yolo_outputs = yolo_head(model.output, anchors, n_classes)
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape)
    out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    model.input: batch_img,
                    input_image_shape: [img.shape[0], img.shape[1]],
                    K.learning_phase(): 0
                })
    return out_boxes, out_scores, out_classes

def draw_YOLO_boxes(img, out_boxes, out_classes, out_scores, class_names, colors):
    thickness = (img.size[1] + img.size[0]) // 300
    font = ImageFont.truetype(font='.\\YAD2K\\font\\FiraMono-Medium.otf', 
                              size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
    #info_font = ImageFont.truetype(font='.\\YAD2K\\font\\FiraMono-Medium.otf', 
    #                          size=35)
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
    
        label = '{} {:.2f}'.format(predicted_class, score)
    
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(label, font)
    
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(img.size[0], np.floor(right + 0.5).astype('int32'))
    
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
    
        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #draw.text(np.asarray([50,35]), "Cars Detected: {:d}".format(len(out_boxes)), fill=(255, 0, 0, 255), font=info_font)
        del draw
    return img

