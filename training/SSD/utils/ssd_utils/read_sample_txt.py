import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pdb
import pandas as pd


def read_sample_txt(image_path, label_path, name='gun'):
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    # print(label_path, image_path)
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)  # read image in bgr format
    bboxes, classes = [], []

    with open(label_path) as f:
        lines = f.readlines()
        #print("len lines:",len(lines))
        for line in lines:
            bbox = [int(n) for n in line.strip('\n').split(',')[:-1]]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            bboxes.append([xmin,ymin,xmax,ymax])
            classes.append(line.strip('\n').split(',')[-1])
        f.close()
    # pdb.set_trace()
    # print(bboxes,classes)

    return np.array(image, dtype=np.float), np.array(bboxes, dtype=np.uint8), classes
