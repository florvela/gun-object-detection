
from flask import Flask, jsonify, request
app = Flask(__name__)

import cv2
import os
import json
import numpy as np
from utils import get_inference_model
import pdb

args = {
    'config': './configs/vgg16_flor.json',
    'weights': './output/cp_ep_100_loss_39.8607.h5',
    'label_maps': ['gun'],
    'confidence_threshold': 0.9,
    'num_predictions': 10,
    'show_class_label': True
}

assert os.path.exists(args["config"]), "config file does not exist"
assert args["num_predictions"] > 0, "num_predictions must be larger than zero"
assert args["confidence_threshold"] > 0, "confidence_threshold must be larger than zero."
assert args["confidence_threshold"] <= 1, "confidence_threshold must be smaller than or equal to 1."

with open(args["config"], "r") as config_file:
    config = json.load(config_file)

input_size = config["model"]["input_size"]
model, process_input_fn = get_inference_model(config, args)
model.load_weights(args["weights"])

@app.route('/', methods=['POST'])
def inference():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = np.array(image, dtype=np.float)
    image = np.uint8(image)
    image_height, image_width, _ = image.shape
    height_scale, width_scale = input_size/image_height, input_size/image_width

    image = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = process_input_fn(image)

    image = np.expand_dims(image, axis=0)
    y_pred = model.predict(image)

    results = []
    for i, pred in enumerate(y_pred[0]):
        classname = args["label_maps"][int(pred[0]) - 1].upper()
        confidence_score = pred[1]

        score = f"{'%.2f' % (confidence_score * 100)}%"
        print(f"-- {classname}: {score}")

        if confidence_score <= 1 and confidence_score > args["confidence_threshold"]:
            xmin = max(int(pred[2] / width_scale), 1)
            ymin = max(int(pred[3] / height_scale), 1)
            xmax = min(int(pred[4] / width_scale), image_width-1)
            ymax = min(int(pred[5] / height_scale), image_height-1)
            results.append({'classname': classname, 'score':score, 'xmin':xmin, 'ymin': ymin, 'xmax':xmax, 'ymax':ymax})
    return jsonify(results=results)



if __name__ == '__main__':
    app.run(debug=True)