# importing the requests library
import requests
import json
import cv2
import os
import cv2
import numpy as np
import pdb
from scorers_configs import get_config
from flask import jsonify, make_response
from flask_restful import Resource, Api
from flask import Flask, request
import cv2
import json
import numpy as np
# from ssd_utils.utils import inference_utils
from flask_cors import CORS
import sys
import base64
from flask import Flask, Response
import cv2
import threading
import time

# class SSD_model():
#     def __init__(self):
#         CONFIG = get_config("SSD_1")
#
#         self.CONFIDENCE_THRESHOLD = CONFIG["confidence_threshold"]
#
#         self.num_predictions = CONFIG["num_predictions"]
#         self.show_class_label = CONFIG["show_class_label"]
#         weights = CONFIG["weights_path"]
#
#         self.class_names = list(CONFIG["labels"].keys())
#         self.THRESHS = CONFIG["labels"]
#
#         args = {
#             'config': CONFIG["config"],
#             'weights': weights,
#             'label_maps': self.class_names,
#             'confidence_threshold': self.CONFIDENCE_THRESHOLD ,
#             'num_predictions': self.num_predictions,
#             'show_class_label': self.show_class_label
#         }
#
#         with open(CONFIG["config"], "r") as config_file:
#             config = json.load(config_file)
#
#         self.input_size = config["model"]["input_size"]
#         model, process_input_fn = inference_utils.inference_ssd_vgg16(config, args)
#         model.load_weights(args["weights"])
#         self.model = model
#         self.process_input_fn = process_input_fn

class YOLOv4_model():
    def __init__(self):
        CONFIG = get_config("yolov4")

        self.CONFIDENCE_THRESHOLD = CONFIG["confidence_threshold"]
        self.NMS_THRESHOLD = CONFIG["nms_threshold"]
        self.class_names = list(CONFIG["labels"].keys())
        self.THRESHS = CONFIG["labels"]

        weights = CONFIG["weights_path"]
        model_config = CONFIG["cfg_path"]
        input_width = CONFIG["input_width"]
        input_height = CONFIG["input_height"]

        net = cv2.dnn.readNet(weights, model_config)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(input_width, input_height), scale=1 / 255, swapRB=True)

        self.model = model


# ssd_model = SSD_model()
YOLOv4_model = YOLOv4_model()

# def predict_SSD(img):
#     image = cv2.resize(img, (ssd_model.input_size, ssd_model.input_size))
#     image_height, image_width, _ = img.shape
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = ssd_model.process_input_fn(image)
#     image = np.expand_dims(image, axis=0)
#
#     height_scale, width_scale = ssd_model.input_size / image_height, ssd_model.input_size / image_width
#     y_pred = ssd_model.model.predict(image)
#
#     temp = ["bg"] + ssd_model.class_names
#     classes, class_names, scores, boxes = [], [], [], []
#
#     for pred in y_pred[0]:
#         class_name = temp[int(pred[0])]
#         if pred[1] >= ssd_model.THRESHS[class_name]:
#             classes.append(int(pred[0]))
#             class_names.append(temp[int(pred[0])])
#             scores.append(pred[1].astype(float))
#             xmin = max(int(pred[2] / width_scale), 1)
#             ymin = max(int(pred[3] / height_scale), 1)
#             xmax = min(int(pred[4] / width_scale), image_width - 1)
#             ymax = min(int(pred[5] / height_scale), image_height - 1)
#             boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
#
#     if class_names:
#         res = {"decisions": "threat detected",
#                "detections": {
#                    "classes": class_names,
#                    "scores": scores,
#                    "boxes": boxes,
#                    "class_ids": classes,
#                    "thresh": ssd_model.CONFIDENCE_THRESHOLD
#                }}
#     else:
#         res = {"decisions": "no threat detected"}
#
#     return res

def predict_YOLO(img):
    classes, scores, boxes = YOLOv4_model.model.detect(img, YOLOv4_model.CONFIDENCE_THRESHOLD,
                                                       YOLOv4_model.NMS_THRESHOLD)
    class_names, res_classes, res_scores, res_boxes, res_thresholds = [], [], [], [], []
    if type(classes) != tuple:
        for i, _class in enumerate(classes):
            if scores[i][0] >= YOLOv4_model.THRESHS[YOLOv4_model.class_names[_class[0]]]:
                class_names.append(YOLOv4_model.class_names[_class[0]])
                res_scores.append(scores[i][0].astype(float))
                res_classes.append(int(_class[0]))
                res_boxes.append(boxes[i].astype(int).tolist())
                res_thresholds.append(YOLOv4_model.THRESHS[YOLOv4_model.class_names[_class[0]]])

    if class_names:
        res = {"decisions": "threat detected",
               "detections": {
                   "classes": class_names,
                   "scores": res_scores,
                   "boxes": res_boxes,
                   "class_ids": res_classes,
                   "thresholds": res_thresholds
               }}
    else:
        res = {"decisions": "no threat detected"}

    return res

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

cap = cv2.VideoCapture('video/video1.mp4')


# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # send http request with image and receive response
        # start = time.time()
        # data = predict_SSD(frame)
        # if "detections" in data:
        #     detections = data["detections"]
        #     classids, classes, scores, boxes = detections["class_ids"], detections["classes"], detections["scores"], \
        #                                        detections["boxes"]
        #     for (classid, class_name, score, box) in zip(classids, classes, scores, boxes):
        #         color = COLORS[int(classid) % len(COLORS)]
        #         color = COLORS[1]
        #         label = "[SSD] %s : %f" % (class_name, score)
        #         print(label)
        #         cv2.rectangle(frame, box, color, 2)
        #         cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # end = time.time()
        # print("SSD:\t",end-start)

        start = time.time()
        data = predict_YOLO(frame)
        if "detections" in data:
            detections = data["detections"]
            classids, classes, scores, boxes = detections["class_ids"], detections["classes"], detections["scores"], \
                                               detections["boxes"]
            for (classid, class_name, score, box) in zip(classids, classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                color = COLORS[0]
                label = "[YOLO] %s : %f" % (class_name, score)
                print(label)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        end = time.time()
        print("YOLO:\t", end-start)

        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



