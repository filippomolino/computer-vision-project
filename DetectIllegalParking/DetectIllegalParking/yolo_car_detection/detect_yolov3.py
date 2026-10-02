import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov3.weights")
CFG_PATH = os.path.join(os.path.dirname(__file__), "yolov3.cfg")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "coco.names")

net = cv2.dnn.readNet(MODEL_PATH, CFG_PATH)
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

target_classes = [2, 3, 5, 7]  # class IDs from coco.names

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def detect_objects(image):
  
    height, width = image.shape[:2]

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Process each output from YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in target_classes:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    detections = []
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            detections.append((x1, y1, x2, y2, classes[class_ids[i]], confidences[i]))
    return detections

if __name__ == "__main__":
    image_path = "C:/Users/Filippo/Documents/DSDM/Computer Vision/Car_3D_detection/images/leftImg8bit/val/frankfurt/frankfurt_000000_005543_leftImg8bit.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Image at '{image_path}' not found.")
        exit()

    detections = detect_objects(image)

    # Draw detections on the image
    for (x1, y1, x2, y2, label, conf) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv3 Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
