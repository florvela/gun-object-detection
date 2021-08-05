# importing the requests library
import requests
import json
import cv2

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

addr = "http://127.0.0.1:5000"
test_url = addr + '/api/v1/yolov4'
ssd_url = addr + '/api/v1/ssd'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

frames = ["test1.jpg","test2.jpg","test3.jpg","test4.jpg"]
for f in frames:
    frame = cv2.imread(f)
    print(frame.shape)
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', frame)
    # send http request with image and receive response
    r = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

    data = r.json()
    if "detections" in data:
        detections = data["detections"]
        classids, classes, scores, boxes = detections["class_ids"], detections["classes"], detections["scores"], detections["boxes"]
        for (classid, class_name, score, box) in zip(classids, classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            color = COLORS[0]
            label = "[YOLO] %s : %f" % (class_name, score)
            print(label)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    r = requests.post(ssd_url, data=img_encoded.tostring(), headers=headers)

    data = r.json()
    if "detections" in data:
        detections = data["detections"]
        classids, classes, scores, boxes = detections["class_ids"], detections["classes"], detections["scores"], detections["boxes"]
        for (classid, class_name, score, box) in zip(classids, classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            color = COLORS[1]
            label = "[SSD] %s : %f" % (class_name, score)
            print(label)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("detections",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


