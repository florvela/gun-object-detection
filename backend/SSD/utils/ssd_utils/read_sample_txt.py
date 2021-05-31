import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pdb
import pandas as pd


def read_sample_txt(image_path, label_path, name='gun'):
    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)  # read image in bgr format
    bboxes, classes = [], []

    with open(label_path) as f:
        lines = f.readlines()
        for line in lines:
            bboxes.append(line.strip('\n').split(',')[:-1])
            classes.append(line.strip('\n').split(',')[-1])
        f.close()
    # pdb.set_trace()

    return np.array(image, dtype=np.float), np.array(bboxes, dtype=np.float), classes
