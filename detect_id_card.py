import cv2
from ultralytics import YOLO
import pyautogui
import time

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    classes = results.names
    has_id_card = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = classes[cls_id]

        if label == 'id_card':
            has_id_card = True

    frame_out = results.plot()

    cv2.imshow("ID Detection", frame_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()