import os
import json
import pdb
import numpy as np
import argparse
from glob import glob
import matplotlib.pyplot as plt
from utils import bbox_utils
import pandas as pd
import pdb
# importing the requests library
import requests
import json
import cv2
import os

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

addr = "http://127.0.0.1:8000"
test_url = addr + '/api/v1/yolov4'
ssd_url = addr + '/api/v1/ssd'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

parser = argparse.ArgumentParser(description='Start the evaluation process of a particular network.')
parser.add_argument("--iou_threshold", type=float, help="iou between gt box and pred box to be counted as a positive.", default=0.5)
parser.add_argument("--set", type=str, help="set to evaluate", default="test")
args = parser.parse_args()


assert args.iou_threshold > 0, "iou_threshold must be larger than zeros"

if args.set == "valid":
    coco_annotations = "datasets/valid/_annotations.coco.json"
else:
    coco_annotations = "datasets/valid/_annotations.coco.json"
f = open(coco_annotations, )
data = json.load(f)
annotations = [ann for ann in data["annotations"]]
images = [img for img in data["images"]]
images_dict = {}
for img in data["images"]:
    images_dict[img["id"]] = img["file_name"]

import cv2

ground_truths_dict = {}
label_maps = ["", "KNIFE", "GUN", "RIFLE"]
for annotation in annotations:
    key = images_dict[annotation["image_id"]]
    if key not in ground_truths_dict:
        # print(key)
        ground_truths_dict[key] = []
    x = annotation["bbox"]
    ground_truths_dict[key].append({
        "class": label_maps[int(annotation["category_id"])],
        "bbox": [int(x[0]), int(x[1]), int(x[2]), int(x[3])]
    })


for image_type in ["KNIFE", "GUN", "RIFLE"]:
    master_df = pd.DataFrame()
    images_dir = "../../src/test_images/" + image_type + "/"

    frames = os.listdir(images_dir)
    counter = 0
    for f in frames:
        counter += 1
        frame = cv2.imread(images_dir + f)
        # encode image as jpeg
        _, img_encoded = cv2.imencode('.jpg', frame)

        image_results = list()

        # # send http request with image and receive response
        r = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

        data = r.json()
        if "detections" in data:
            detections = data["detections"]
            print(detections)
            classids, classes, scores, boxes = detections["class_ids"], detections["classes"], detections["scores"], \
                                               detections["boxes"]
            for (classid, class_name, score, box) in zip(classids, classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                color = COLORS[0]
                label = "[YOLO] %s : %f" % (class_name, score)
                print(label)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


                for gt in ground_truths_dict[f]:
                    if class_name == gt["class"]:
                        iou = bbox_utils.iou( np.expand_dims(box, axis=0), np.expand_dims(gt["bbox"], axis=0), )[0]

                        detection = dict()
                        detection["iou"] = iou
                        detection["model"] = "YOLO"
                        detection["score"] = score
                        detection["image_number"] = counter
                        image_results.append(detection)

                    else:
                        print(gt["class"],class_name)

        r = requests.post(ssd_url, data=img_encoded.tostring(), headers=headers)

        data = r.json()
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

                for gt in ground_truths_dict[f]:
                    color = COLORS[2]
                    label = "[GT]"
                    cv2.rectangle(frame, gt["bbox"], color, 2)

                    if class_name == gt["class"]:
                        iou = bbox_utils.iou( np.expand_dims(box, axis=0), np.expand_dims(gt["bbox"], axis=0), )[0]

                        detection = dict()
                        detection["iou"] = iou
                        detection["model"] = "SSD"
                        detection["score"] = score
                        detection["class_name"] = class_name
                        detection["image_number"] = counter
                        image_results.append(detection)

                    else:
                        print(gt["class"],class_name)

        # cv2.imshow("detections", frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(os.path.join("../../src/test_images/predictions", f), frame)

        master_df = pd.concat([master_df, pd.DataFrame(image_results)])

    master_df.to_csv(image_type + "_test_results.csv", index=False)

