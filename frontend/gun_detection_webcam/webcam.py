import numpy as np
import cv2
import requests
import json

cap = cv2.VideoCapture(0)
content_type = 'image/jpeg'
headers = {'content-type': content_type}

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    _, img_encoded = cv2.imencode('.jpg', frame)

    response = requests.post('http://127.0.0.1:5000', data=img_encoded.tobytes(), headers=headers)
    response = json.loads(response.text)
    for res in  response['results']:
        cv2.putText(
            frame,
            res['score'],
            (int(res['xmin']), int(res['ymin'])),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (100, 100, 255),
            1, 1)
        cv2.rectangle(
            frame,
            (res['xmin'], res['ymin']),
            (res['xmax'], res['ymax']),
            (0, 255, 255),
            1)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()