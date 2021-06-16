import cv2
import os
import json
import numpy as np
from utils import inference_utils


args = {
    'input_image': './datasets/0a4a34cff82148b5_jpg.rf.5c2962aeabcb6d5ea0b6291976ce42c3.jpg',
    'config': './configs/vgg16_flor.json',
    'weights': '../output_2/cp_ep_100_loss_17.3806.h5',
    'label_maps': ['gun'],
    'confidence_threshold': 0.4,
    'num_predictions': 10,
    'show_class_label': True
}

assert os.path.exists(args["input_image"]), "config file does not exist"
assert os.path.exists(args["config"]), "config file does not exist"
assert args["num_predictions"] > 0, "num_predictions must be larger than zero"
assert args["confidence_threshold"] > 0, "confidence_threshold must be larger than zero."
assert args["confidence_threshold"] <= 1, "confidence_threshold must be smaller than or equal to 1."

with open(args["config"], "r") as config_file:
    config = json.load(config_file)

input_size = config["model"]["input_size"]
model_config = config["model"]

model, process_input_fn = inference_utils.inference_ssd_vgg16(config, args)
image = cv2.imread(args["input_image"])  # read image in bgr format
image = np.array(image, dtype=np.float)
image = np.uint8(image)

label_maps = args["label_maps"]

model.load_weights(args["weights"])

display_image = image.copy()
image_height, image_width, _ = image.shape
height_scale, width_scale = input_size/image_height, input_size/image_width

image = cv2.resize(image, (input_size, input_size))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = process_input_fn(image)

image = np.expand_dims(image, axis=0)
y_pred = model.predict(image)


for i, pred in enumerate(y_pred[0]):
    classname = label_maps[int(pred[0]) - 1].upper()
    confidence_score = pred[1]

    score = f"{'%.2f' % (confidence_score * 100)}%"
    print(f"-- {classname}: {score}")

    if confidence_score <= 1 and confidence_score > args["confidence_threshold"]:
        # pdb.set_trace()
        xmin = max(int(pred[2] / width_scale), 1)
        ymin = max(int(pred[3] / height_scale), 1)
        xmax = min(int(pred[4] / width_scale), image_width-1)
        ymax = min(int(pred[5] / height_scale), image_height-1)

        cv2.putText(
            display_image,
            score,
            (int(xmin), int(ymin)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (100, 100, 255),
            1, 1)
        cv2.rectangle(
            display_image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 255),
            1)

cv2.imshow("image", display_image)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
