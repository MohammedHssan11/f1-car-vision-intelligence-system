from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
img = cv2.imread(r"C:\TERM 7\computer vision\final project\data\CarDD_YOLO\images\train\000009.jpg")

results = model(img, conf=0.25)
results[0].show()
