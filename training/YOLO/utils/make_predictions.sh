#!/bin/bash
python make_predictions.py ../darknet/output/yolov4-custom-flor_1000.weights ../../predictions/complete_test_set/yolo_1000_bacthes/ ../../datasets/test/ ../darknet/cfg/yolov4-custom-flor.cfg &&
python make_predictions.py ../darknet/output/yolov4-custom-flor_2000.weights ../../predictions/complete_test_set/yolo_2000_bacthes/ ../../datasets/test/ ../darknet/cfg/yolov4-custom-flor.cfg &&
python make_predictions.py ../darknet/output/yolov4-custom-flor_3000.weights ../../predictions/complete_test_set/yolo_3000_bacthes/ ../../datasets/test/ ../darknet/cfg/yolov4-custom-flor.cfg &&
python make_predictions.py ../darknet/output/yolov4-custom-flor_4000.weights ../../predictions/complete_test_set/yolo_4000_bacthes/ ../../datasets/test/ ../darknet/cfg/yolov4-custom-flor.cfg &&
python make_predictions.py ../darknet/output/yolov4-custom-flor_5000.weights ../../predictions/complete_test_set/yolo_5000_bacthes/ ../../datasets/test/ ../darknet/cfg/yolov4-custom-flor.cfg &&
python make_predictions.py ../darknet/output/yolov4-custom-flor_6000.weights ../../predictions/complete_test_set/yolo_6000_bacthes/ ../../datasets/test/ ../darknet/cfg/yolov4-custom-flor.cfg