import os
import argparse
import shutil
from glob import glob

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
with open(os.path.join(args.output_dir, "train.txt"), "w") as train_split_file:
    for sample in [f.strip(".jpg") for f in os.listdir(os.path.join(args.dataset_dir, "train")) if f.endswith(".jpg")]:
        train_split_file.write(f"{sample}.jpg {sample}.xml\n")

print(f"---- val.txt")
with open(os.path.join(args.output_dir, "val.txt"), "w") as train_split_file:
    for sample in [f.strip(".jpg") for f in os.listdir(os.path.join(args.dataset_dir, "valid")) if f.endswith(".jpg")]:
        train_split_file.write(f"{sample}.jpg {sample}.xml\n")


train_dir = os.path.join(args.dataset_dir, "train")
valid_dir = os.path.join(args.dataset_dir, "valid")
print(f"-- copying images")
for i, sample in enumerate(list(glob(os.path.join(train_dir, "*jpg")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_images_dir, filename)
    )

for i, sample in enumerate(list(glob(os.path.join(valid_dir, "*jpg")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_images_dir, filename)
    )

print(f"-- copying labels")
for i, sample in enumerate(list(glob(os.path.join(train_dir, "*xml")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_labels_dir, filename)
    )

for i, sample in enumerate(list(glob(os.path.join(valid_dir, "*xml")))):
    filename = os.path.basename(sample)
    shutil.copy(
        sample,
        os.path.join(out_labels_dir, filename)
    )

print(f"-- writing label_maps.txt")
with open(os.path.join(args.output_dir, "label_maps.txt"), "w") as label_maps_file:
    labels = ['0','1','Shotgun']
    for classname in labels:
        label_maps_file.write(f"{classname}\n")

