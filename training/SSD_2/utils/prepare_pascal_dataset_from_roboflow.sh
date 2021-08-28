import os
import cv2
import argparse
from glob import glob
import json
from xml.dom import minidom
import xml.etree.cElementTree as ET
from data_utils import COCO_Text
import numpy as np
import shutil
import os

parser = argparse.ArgumentParser(
    description='Converts the Pascal VOC 2012 dataset to a format suitable for training tbpp with this repo.')
parser.add_argument('dataset_dir', type=str, help='path to dataset dir.')
parser.add_argument('output_dir', type=str, help='path to output dir.')
args = parser.parse_args()


assert os.path.exists(args.dataset_dir), "dataset_dir does not exist"
out_images_dir = os.path.join(args.output_dir, "images")
out_labels_dir = os.path.join(args.output_dir, "labels")
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)

print(f"-- creating split files")
print(f"---- train.txt")
# with open(os.path.join(args.output_dir, "train.txt"), "w") as train_split_file:
for file in os.path.join(args.dataset_dir, "train"):
    print(file)