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
        lines = f.readlines()[1:]
        for line in lines:
            bboxes.append(line.strip('\n').split(' '))
            classes.append(name)
        f.close()

    return np.array(image, dtype=np.float), np.array(bboxes, dtype=np.float), classes

# def read_sample_test(image_path):
#     image_path = image_path.strip("\n")
#     assert os.path.exists(image_path), "Image file does not exist."

#     image = cv2.imread(image_path)  # read image in bgr format
#     return np.array(image, dtype=np.float)


# from tensorflow.keras.applications import vgg16
# train = '../../../datasets/fruits/train_zip/train/apple_1'
# x = '../../../datasets/SasankYadati-Guns-Dataset-0eb7329/Labels/1.txt'
# y = '../../../datasets/SasankYadati-Guns-Dataset-0eb7329/Images/1.jpeg'
# #read_sample(train+'.jpg', train+'.xml')

# image, bboxes, classes = read_sample_txt(y,x)

# image = cv2.imread(y) 
# display_image = image.copy()
# image_height, image_width, _ = image.shape
# height_scale, width_scale = 300/image_height, 300/image_width

# image = cv2.resize(image, (300, 300))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = vgg16.preprocess_input(image)

# image = np.expand_dims(image, axis=0)
# # image = process_input_fn(image)

# # image = np.expand_dims(image, axis=0)
# # y_pred = model.predict(image)

# for i, bbox in enumerate(bboxes):
#     xmin = int(bbox[0]) 
#     ymin = int(bbox[1]) 
#     xmax = int(bbox[2]) 
#     ymax = int(bbox[3]) 
#     cv2.putText(
#             display_image,
#             classes[i],
#             (int(xmin), int(ymin)),
#             cv2.FONT_HERSHEY_PLAIN,
#             1,
#             (100, 100, 255),
#             1,
#             1
#         )

#     cv2.rectangle(
#         display_image,
#         (xmin, ymin),
#         (xmax, ymax),
#         (0, 0, 255),
#         1
#     )

# cv2.imshow("image", display_image)
# if cv2.waitKey(0) == ord('q'):
#     cv2.destroyAllWindows()