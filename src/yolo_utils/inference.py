import cv2
import pdb
import time

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = ["1","2","3","4"]
# with open("classes.txt", "r") as f:
#     class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("demo.mp4")

net = cv2.dnn.readNet("yolov4-custom-flor_best.weights", "yolov4-custom-flor.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


frame = "test1.jpg"
frame = cv2.imread(frame)
print(frame.shape)

classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
print(classes)
print(scores)
print(boxes)

for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = "%s : %f" % (class_names[classid[0]], score)
    print(label)
    cv2.rectangle(frame, box, color, 2)
    # cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


cv2.imshow("detections",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

