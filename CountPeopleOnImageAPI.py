from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import numpy as np
import base64

app = FastAPI()

# Model and classNames initialization
prototxt = "MobileNetSSD_deploy.prototxt"
caffe_model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
classNames = {0: 'background', 15: 'person'}

# Initialize the camera capture
cap = cv2.VideoCapture(0)

@app.get("/get_detection_data/")
def get_detection_data():
    global cap, net, classNames

    frame = None

    while frame is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if frame is not None:
            # size of image
            width = frame.shape[1]
            height = frame.shape[0]
            print(width, height)
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()

            people_count = 0

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.2:
                    class_id = int(detections[0, 0, i, 1])

                    if class_id == 15:
                        people_count += 1
                        # scale to the frame
                        x_top_left = int(detections[0, 0, i, 3] * width) 
                        y_top_left = int(detections[0, 0, i, 4] * height)
                        x_bottom_right   = int(detections[0, 0, i, 5] * width)
                        y_bottom_right   = int(detections[0, 0, i, 6] * height)
                        
                        # draw bounding box around the detected object
                        cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                    (0, 255, 0))
                        
                        if class_id in classNames:
                            # get class label
                            label = classNames[class_id] + ": " + str(confidence)
                            # get width and text of the label string
                            (w, h),t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            y_top_left = max(y_top_left, h)
                            # draw bounding box around the text
                            cv2.rectangle(frame, (x_top_left, y_top_left - h), 
                                            (x_top_left + w, y_top_left + t), (0, 0, 0), cv2.FILLED)
                            cv2.putText(frame, label, (x_top_left, y_top_left),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            

            # Return the data as a dictionary
            response_data = {
                "people_count": people_count,
                "frame": base64.b64encode(frame).decode('utf-8')

            }

            cap.release()
            return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
