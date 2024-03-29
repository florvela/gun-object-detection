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

parser = argparse.ArgumentParser(description='Start the evaluation process of a particular network.')
parser.add_argument('predictions_dir', type=str, help='path to the predictions dir.')
parser.add_argument('output_dir', type=str, help='path to output file.', default="output")
parser.add_argument("--iou_threshold", type=float, help="iou between gt box and pred box to be counted as a positive.", default=0.5)
parser.add_argument("--set", type=str, help="set to evaluate", default="test")
args = parser.parse_args()

assert os.path.exists(args.predictions_dir), "predictions_dir does not exist"
assert args.iou_threshold > 0, "iou_threshold must be larger than zeros"
df = pd.DataFrame()

predictions_dir_parent = args.predictions_dir
for elem in os.listdir(predictions_dir_parent):
    predictions_dir = predictions_dir_parent + elem
    output_dir = args.output_dir + "/" + elem
    print(elem)

    csv_row = dict()
    model_type, batches, learning_rate, optimizer, epochs, augmented_flag = elem.split("_")
    csv_row["model_type"] = model_type
    csv_row["batches"] = batches
    csv_row["learning_rate"] = learning_rate
    csv_row["optimizer"] = optimizer
    csv_row["epochs"] = epochs
    csv_row["augmented_flag"] = augmented_flag


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_files = glob(os.path.join(predictions_dir, "*txt"))

    predictions_dict = {}
    for prediction_file in prediction_files:
        filename = prediction_file.split("\\")[1].strip(".txt")
        # pdb.set_trace()
        # filename = os.path.basename(prediction_file)
        # filename = filename[:filename.index(".")]
        with open(prediction_file, "r") as pred_file:
            predictions = pred_file.readlines()
            predictions = [p.strip("\n").split(" ") for p in predictions]
            predictions = [{
                "class": p[0],
                "confidence_score": float(p[1]),
                "bbox": [int(p[2]), int(p[3]), int(p[2])+int(p[4]), int(p[3])+int(p[5])] #[int(p[2]), int(p[3]), int(p[2])+int(p[4]), int(p[3])+int(p[5])] #[int(p[2]), int(p[3]), int(float(p[4])-float(p[2])), int(float(p[5])-float(p[3]))],
            } for p in predictions]
            predictions_dict[filename] = predictions

    if args.set == "valid":
        coco_annotations = "datasets/valid/_annotations.coco.json"
    else:
        coco_annotations = "datasets/test/_annotations.coco.json"
    f = open(coco_annotations, )
    data = json.load(f)
    annotations = [ann for ann in data["annotations"]]
    images = [img for img in data["images"]]
    images_dict = {}
    for img in data["images"]:
        images_dict[img["id"]] = img["file_name"]


    import cv2
    ground_truths_dict = {}
    label_maps = ["","KNIFE","GUN","RIFLE"]
    for annotation in annotations:
        key = images_dict[annotation["image_id"]]
        if key in predictions_dict:
            if key not in ground_truths_dict:
                # print(key)
                ground_truths_dict[key]= []
            x = annotation["bbox"]
            ground_truths_dict[key].append({
                "class": label_maps[int(annotation["category_id"])],
                "bbox": [x[0],x[1],x[0]+x[2],x[1]+x[3]]
            })

    assert len(list(ground_truths_dict.keys())) == len(list(predictions_dict.keys())), "prediction files does not equal to ground truth files"

    aps = []
    metrics = {}
    label_maps = ["KNIFE","GUN","RIFLE"]

    for classname in label_maps:
        detections = []
        all_ground_truths = 0
        for filename in list(predictions_dict.keys()):
            predictions_in_file = [p for p in predictions_dict[filename] if p["class"] == classname]
            ground_truth_in_file = [g for g in ground_truths_dict[filename] if g["class"] == classname]

            all_ground_truths += len(ground_truth_in_file)

            gt_pred_matrix = np.zeros((len(ground_truth_in_file), len(predictions_in_file)))
            for pred_i, pred in enumerate(predictions_in_file):
                for gt_i, gt in enumerate(ground_truth_in_file):
                    gt_pred_matrix[gt_i][pred_i] = bbox_utils.iou(
                        np.expand_dims(pred["bbox"], axis=0),
                        np.expand_dims(gt["bbox"], axis=0),
                    )[0]

            t = np.where(gt_pred_matrix >= args.iou_threshold, 1, 0)
            for gt_i in range(t.shape[0]):
                row = t[gt_i]
                true_positives_per_gt = np.argwhere(row == 1)
                if len(true_positives_per_gt) > 1:
                    t[gt_i] = np.zeros_like(row)
                    largest_overlap_idx = np.argmax(row)
                    t[gt_i][largest_overlap_idx] = 1

            for pred_idx in range(t.shape[1]):
                cols = t[:, pred_idx]
                # pdb.set_trace()
                if (len(np.argwhere(cols == 1)) > 0):
                    detections.append([predictions_in_file[pred_idx]["confidence_score"], 1])  # 1 for tp
                else:
                    detections.append([predictions_in_file[pred_idx]["confidence_score"], 0.48])  # 1 for tp

        detections.sort(key=lambda x: x[0], reverse=True)  # highest confidence score first
        detections = np.array(detections)

        precisions, recalls = [], []
        intp_precisions, intp_recalls = [], []

        for i in range(detections.shape[0]):
            detections_for_threshold = detections[:i+1]
            if (detections_for_threshold.shape[0] > 0):
                tp = sum(detections_for_threshold[:, 1])
                precision = tp / len(detections_for_threshold)
                recall = tp / all_ground_truths
                precisions.append(precision)
                recalls.append(recall)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        for recall in np.arange(0, 1.1, 0.1):
            intp_recalls.append(recall)
            temp = precisions[np.where(recalls >= recall)]
            if len(temp) == 0:
                intp_precisions.append(0)
            else:
                intp_precisions.append(max(temp))

        intp_precisions = np.array(intp_precisions)
        intp_recalls = np.array(intp_recalls)

        ap = intp_precisions.mean()
        aps.append(ap)

        metrics[classname] = {
            "precisions": precisions,
            "recalls": recalls,
        }

        print(f"-- {classname}: {'%.2f' % (ap * 100)}%")
        csv_row[classname] = f'%.2f' % (ap * 100)

        plt.figure(figsize=(7, 7))
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.plot(recalls, precisions, label="Precision")
        plt.scatter(intp_recalls, intp_precisions, label="11-Point Interpolated Precision", color="red")
        plt.title(f"Precision-Recall Curve\nClass: {str(classname).lower()}, AP: {'%.2f' % (ap * 100)}%")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig(os.path.join(output_dir, f"{str(classname).lower()}_ap-{'%.2f' % ap}.png"))


    aps = np.array(aps)
    mAP = aps.mean()
    print(f"mAP: {'%.2f' % (mAP * 100)}%")
    csv_row["mAP"] = f'%.2f' % (mAP * 100)

    df = df.append(csv_row, ignore_index=True)

    plt.figure(figsize=(7, 7))
    for classname in metrics.keys():
        plt.plot(metrics[classname]["recalls"], metrics[classname]["precisions"], label=classname)
    plt.title(f"Precision-Recall Curve\n{len(label_maps)} classes, mAP: {'%.2f' % (mAP * 100)}%")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"_all_map-{'%.2f' % mAP}.png"))

df.to_csv(f"{args.set}_evaluation_results.csv", index=False)