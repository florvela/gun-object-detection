from scorers_configs import get_config
from flask import jsonify, make_response
from flask_restful import Resource, Api
from flask import Flask, request
import cv2
import json
import numpy as np
from ssd_utils.utils import inference_utils
from flask_cors import CORS
import sys
import base64
from flask import Flask, Response
import cv2
import threading

# creating the flask app
app = Flask(__name__)
CORS(app)
# creating an API object
api = Api(app)

lock = threading.Lock()

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    buf_decode = base64.b64decode(encoded_data)
    nparr = np.fromstring(buf_decode, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

class Hello(Resource):

    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    def get(self):
        return jsonify({'message': 'hello world'})

    # Corresponds to POST request
    def post(self):
        data = request.get_json()  # status code
        return make_response(jsonify({'data': data}), 201)
        # return jsonify(), 201


class SSD_model():
    def __init__(self):
        CONFIG = get_config("SSD_1")

        self.CONFIDENCE_THRESHOLD = CONFIG["confidence_threshold"]

        self.num_predictions = CONFIG["num_predictions"]
        self.show_class_label = CONFIG["show_class_label"]
        weights = CONFIG["weights_path"]

        self.class_names = list(CONFIG["labels"].keys())
        self.THRESHS = CONFIG["labels"]

        args = {
            'config': CONFIG["config"],
            'weights': weights,
            'label_maps': self.class_names,
            'confidence_threshold': self.CONFIDENCE_THRESHOLD ,
            'num_predictions': self.num_predictions,
            'show_class_label': self.show_class_label
        }

        with open(CONFIG["config"], "r") as config_file:
            config = json.load(config_file)

        self.input_size = config["model"]["input_size"]
        model, process_input_fn = inference_utils.inference_ssd_vgg16(config, args)
        model.load_weights(args["weights"])
        self.model = model
        self.process_input_fn = process_input_fn

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
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(input_width, input_height), scale=1 / 255, swapRB=True)

        self.model = model


ssd_model = SSD_model()
YOLOv4_model = YOLOv4_model()


def predict_SSD(img):
    image = cv2.resize(img, (ssd_model.input_size, ssd_model.input_size))
    image_height, image_width, _ = img.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ssd_model.process_input_fn(image)
    image = np.expand_dims(image, axis=0)

    height_scale, width_scale = ssd_model.input_size / image_height, ssd_model.input_size / image_width
    y_pred = ssd_model.model.predict(image)

    temp = ["bg"] + ssd_model.class_names
    classes, class_names, scores, boxes = [], [], [], []

    for pred in y_pred[0]:
        class_name = temp[int(pred[0])]
        if pred[1] >= ssd_model.THRESHS[class_name]:
            classes.append(int(pred[0]))
            class_names.append(temp[int(pred[0])])
            scores.append(pred[1].astype(float))
            xmin = max(int(pred[2] / width_scale), 1)
            ymin = max(int(pred[3] / height_scale), 1)
            xmax = min(int(pred[4] / width_scale), image_width - 1)
            ymax = min(int(pred[5] / height_scale), image_height - 1)
            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

    if class_names:
        res = {"decisions": "threat detected",
               "detections": {
                   "classes": class_names,
                   "scores": scores,
                   "boxes": boxes,
                   "class_ids": classes,
                   "thresh": ssd_model.CONFIDENCE_THRESHOLD
               }}
    else:
        res = {"decisions": "no threat detected"}

    return res

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

class YOLOv4_resource(Resource):

    def post(self):
        r = request
        nparr = np.fromstring(r_data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res = predict_YOLO(img)
        return jsonify(res)

class SSD_resource(Resource):

    def post(self):
        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res = predict_SSD(img)
        return jsonify(res)


def generate():
    # grab global references to the lock variable
    global lock
    # initialize the video stream
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # FPS = 1 / 30
    # FPS_MS = int(FPS * 1000)

    # check camera is open
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # while streaming
    while rval:
        # wait until the lock is acquired
        with lock:
            # read next frame
            rval, frame = vc.read()
            # if blank frame
            if frame is None:
                continue

            data = predict_SSD(frame)
            if "detections" in data:
                detections = data["detections"]
                classids, classes, scores, boxes = detections["class_ids"], detections["classes"], detections["scores"], \
                                                   detections["boxes"]
                for (classid, class_name, score, box) in zip(classids, classes, scores, boxes):
                    color = COLORS[int(classid) % len(COLORS)]
                    color = COLORS[1]
                    label = "[SSD] %s : %f" % (class_name, score)
                    print(label)
                    cv2.rectangle(frame, box, color, 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    # release the camera
    vc.release()

class Stream_resource(Resource):

    def get(self):
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# adding the defined resources along with their corresponding urls
api.add_resource(Hello, '/')
api.add_resource(YOLOv4_resource, '/api/v1/yolov4')
api.add_resource(SSD_resource, '/api/v1/ssd')
api.add_resource(Stream_resource, '/stream')

# driver function
# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == '__main__':
   host = "127.0.0.1"
   port = 8000
   debug = False
   options = None
   app.run(host, port, debug, options)