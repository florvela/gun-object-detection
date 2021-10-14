import cv2
import os
import json
import pdb
import pickle
import numpy as np

def test(test_config, only_pred_imgs=True):
    from utils import inference_utils
    print(test_config)
    weights, predictions_path, test_images_path = test_config

    args = {
        'config': './configs/vgg16_flor.json',
        'weights': weights,
        'label_maps': ['KNIFE', 'GUN', 'RIFLE'],
        'thresholds': [0.7, 0.8, 0.8],
        'confidence_threshold': 0.7,
        'num_predictions': 10,
        'show_class_label': True
    }

    assert os.path.exists(args["config"]), "config file does not exist"
    assert args["num_predictions"] > 0, "num_predictions must be larger than zero"

    with open(args["config"], "r") as config_file:
        config = json.load(config_file)

    input_size = config["model"]["input_size"]
    model, process_input_fn = inference_utils.inference_ssd_vgg16(config, args)
    model.load_weights(args["weights"])

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    counter = 0
    for file in [file for file in os.listdir(test_images_path) if file.endswith(".jpg")]:
        counter += 1
        print("Testing image number", counter)

        image = cv2.imread(test_images_path + file)  # read image in bgr format
        # image = np.array(image, dtype=np.float)
        # image = np.uint8(image)
        image_height, image_width, _ = image.shape
        height_scale, width_scale = input_size / image_height, input_size / image_width
        image = cv2.resize(image, (input_size, input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = process_input_fn(image)
        image = np.expand_dims(image, axis=0)

        y_pred = model.predict(image)

        has_predictions = False
        for pred in y_pred[0]:
            # pdb.set_trace()
            class_id = int(pred[0].astype(float))
            class_name = args['label_maps'][class_id-1]
            confidence = pred[1].astype(float)
            if confidence >= args["thresholds"][class_id-1]:
                xmin = max(int(pred[2] / width_scale), 1)
                ymin = max(int(pred[3] / height_scale), 1)
                xmax = min(int(pred[4] / width_scale), image_width - 1)
                ymax = min(int(pred[5] / height_scale), image_height - 1)
                # line_elements = [class_, confidence, xmin, ymin, xmax, ymax]

                line_elements = [class_name, confidence, xmin, ymin, xmax-xmin, ymax-ymin]
                # line_elements = [class_, confidence, int(pred[2]), int(pred[3]), int(pred[4]), int(pred[5])]
                line_elements = [str(e) for e in line_elements]
                line = " ".join(line_elements) + "\n"
                with open(predictions_path + file + '.txt', 'a') as write_file:
                    write_file.write(line)
                has_predictions = True
        if not has_predictions:
            with open(predictions_path + file + '.txt', 'a') as write_file:
                write_file.write("")

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('weights_file', type=str, help='path to weights file.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
parser.add_argument('test_set_dir', type=str, help='path to test set dir.')
args = parser.parse_args()
test((args.weights_file, args.output_dir, args.test_set_dir))