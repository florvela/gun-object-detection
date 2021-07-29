import cv2
import numpy as np

label_path = './datasets/test/0a4a34cff82148b5_jpg.rf.5c2962aeabcb6d5ea0b6291976ce42c3'

image = cv2.imread(label_path + '.jpg')  # read image in bgr format
bboxes, classes = [], []
with open(label_path + '.txt') as f:
    lines = f.readlines()
    # print("len lines:",len(lines))
    for line in lines:
        bbox = [int(n) for n in line.strip('\n').split(',')[:-1]]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append(line.strip('\n').split(',')[-1])
    f.close()

for bbox in bboxes:
    xmin, ymin, xmax, ymax = bbox

    image = np.array(image, dtype=np.float)
    image = np.uint8(image)
    # cv2.rectangle(image)
    cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 255), thickness=1)

print(np.uint8(bboxes))
cv2.imshow("image", image)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()