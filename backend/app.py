
from flask import Flask, make_response, jsonify, request
app = Flask(__name__)

import cv2
import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras.applications import vgg16, mobilenet, mobilenet_v2
import numpy as np
from networks import SSD_VGG16, SSD_MOBILENET, SSD_MOBILENETV2
from utils import inference_utils, textboxes_utils, command_line_utils
import pdb

parser = argparse.ArgumentParser(description='run inference on an input image.')
parser.add_argument('--input_image', type=str, help='path to the input image.', default='./datasets/SasankYadati-Guns-Dataset-0eb7329/Images/267.jpeg')
# parser.add_argument('--label_file', type=str, help='path to the label file.', default='../datasets/fruits/train_zip/train/apple_1.xml')
parser.add_argument('--config', type=str, help='path to config file.', default='./configs/ssd300_mobilenetv2.json')
parser.add_argument('--weights', type=str, help='path to the weight file.', default='./output/cp_ep_100_loss_39.8607.h5')
# default='./output/cp_ep_88_loss_15.1093.h5') #cp_ep_100_loss_15.8296.h5')
parser.add_argument('--label_maps', type=str, help='path to label maps file.', default=['gun'])
parser.add_argument('--confidence_threshold', type=float, help='the confidence score a detection should match in order to be counted.', default=0.9)
parser.add_argument('--num_predictions', type=int, help='the number of detections to be output as final detections', default=10)
parser.add_argument('--show_class_label',  type=command_line_utils.str2bool, nargs='?', help='whether or not to show class labels over each detected object.', default=True)
parser.add_argument('--show_quad',  type=command_line_utils.str2bool, nargs='?', help='whether or not to show the quadrilaterals for textboxes++ models', default=False)
args = parser.parse_args()

assert os.path.exists(args.input_image), "config file does not exist"
assert os.path.exists(args.config), "config file does not exist"
assert args.num_predictions > 0, "num_predictions must be larger than zero"
assert args.confidence_threshold > 0, "confidence_threshold must be larger than zero."
assert args.confidence_threshold <= 1, "confidence_threshold must be smaller than or equal to 1."

with open(args.config, "r") as config_file:
    config = json.load(config_file)

input_size = config["model"]["input_size"]
model_config = config["model"]

label_maps = args.label_maps

model = SSD_VGG16(
    config,
    label_maps,
    is_training=False,
    num_predictions=args.num_predictions)
process_input_fn = vgg16.preprocess_input

model.load_weights(args.weights)

@app.route('/', methods=['POST'])
def inference():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # image = cv2.imread(input_image)  # read image in bgr format
    image = np.array(image, dtype=np.float)
    image = np.uint8(image)
    display_image = image.copy()
    image_height, image_width, _ = image.shape
    height_scale, width_scale = input_size/image_height, input_size/image_width

    image = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = process_input_fn(image)

    image = np.expand_dims(image, axis=0)
    y_pred = model.predict(image)

    results = []
    # response = make_response(y_pred.tobytes())
    # response.headers.set('Content-Type', 'application/octet-stream')
    for i, pred in enumerate(y_pred[0]):
        classname = label_maps[int(pred[0]) - 1].upper()
        confidence_score = pred[1]

        score = f"{'%.2f' % (confidence_score * 100)}%"
        print(f"-- {classname}: {score}")

        if confidence_score <= 1 and confidence_score > args.confidence_threshold:
            xmin = max(int(pred[2] / width_scale), 1)
            ymin = max(int(pred[3] / height_scale), 1)
            xmax = min(int(pred[4] / width_scale), image_width-1)
            ymax = min(int(pred[5] / height_scale), image_height-1)
            # xmin = int(pred[2])
            # ymin = int(pred[3])
            # xmax = int(pred[4])
            # ymax = int(pred[5])
            results.append({'classname': classname, 'score':score, 'xmin':xmin, 'ymin': ymin, 'xmax':xmax, 'ymax':ymax})
    return jsonify(results=results)



if __name__ == '__main__':
    app.run(debug=True)