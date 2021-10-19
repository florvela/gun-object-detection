from flask import jsonify
from flask_restful import Resource
from flask import request
import cv2
import numpy as np
import os, pdb
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('weights_file', type=str, help='path to weights file.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
parser.add_argument('test_set_dir', type=str, help='path to test set dir.')
parser.add_argument('cfg_path', type=str, help='path to test set dir.')
args = parser.parse_args()

def get_config(model):
    if model == "yolov4":
        return {
            "model_type": "yolo",
            "input_width": 416,
            "input_height": 416,
            "confidence_threshold": 0.5,
            "nms_threshold": 0.5,
            "weights_path": args.weights_file,
            "cfg_path": args.cfg_path,
            "labels":{
                "KNIFE":0.7,
                "GUN":0.8,
                "RIFLE":0.8,
            }
        }
    return None

class YOLOv4_model():
    def __init__(self):
        CONFIG = get_config("yolov4")

        self.CONFIDENCE_THRESHOLD = CONFIG["confidence_threshold"]
        self.NMS_THRESHOLD = CONFIG["nms_threshold"]
        self.class_names = ["KNIFE","GUN","RIFLE"]#list(CONFIG["labels"].keys())
        self.THRESHS = CONFIG["labels"]

        weights = CONFIG["weights_path"]
        model_config = CONFIG["cfg_path"]
        input_width = CONFIG["input_width"]
        input_height = CONFIG["input_height"]

        net = cv2.dnn.readNet(weights, model_config)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(input_width, input_height), scale=1 / 255, swapRB=True)

        self.model = model

YOLOv4_model = YOLOv4_model()

def detect(img):
    classes, scores, boxes = YOLOv4_model.model.detect(img, YOLOv4_model.CONFIDENCE_THRESHOLD, YOLOv4_model.NMS_THRESHOLD)
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



test_dir = args.test_set_dir
predictions_dir = args.output_dir
os.makedirs(predictions_dir, exist_ok=True)
counter = 0
for f in [file for file in os.listdir(test_dir) if file.endswith(".jpg")]:
    counter += 1
    dir = os.path.join(os.getcwd(),test_dir,f)
    frame = cv2.imread(dir)
    detections = detect(frame)

    if detections["decisions"] == "threat detected":
        d = detections["detections"]
        num_det = len(d["scores"])
        if num_det >= 1:
            for i in range(num_det):
                line_elements = []
                line_elements.append(str(d["classes"][i]))
                line_elements.append(str(d["scores"][i]))
                for b in d["boxes"][i]:
                    line_elements.append(str(b))
                line = " ".join(line_elements) + "\n"
                with open(predictions_dir + f + '.txt', 'a') as write_file:
                    write_file.write(line)
    else:
        with open(predictions_dir + f + '.txt', 'a') as write_file:
            write_file.write("")