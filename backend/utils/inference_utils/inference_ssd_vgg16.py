import os
import cv2
import numpy as np
from networks import SSD_VGG16
from tensorflow.keras.applications import vgg16
from utils import ssd_utils


def inference_ssd_vgg16(config, args):
    """"""
    # assert args.label_maps is not None, "please specify a label map file"
    # assert os.path.exists(args.label_maps), "label_maps file does not exist"
    # with open(args.label_maps, "r") as file:
    #     label_maps = [line.strip("\n") for line in file.readlines()]
    label_maps = args.label_maps

    model = SSD_VGG16(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)
    process_input_fn = vgg16.preprocess_input

    
    image = cv2.imread(args.input_image)  # read image in bgr format
    image = np.array(image, dtype=np.float)

    return model, label_maps, process_input_fn, np.uint8(image)
