from ultralytics import YOLO
import cv2

damage_model = YOLO(r"C:\TERM 7\computer vision\final project\data\CarDD_YOLO\runs\detect\train2\weights\best.pt")
img = cv2.imread(r"C:\Users\mh978\Videos\Captures\Media Player 12_17_2025 11_44_01 AM.png")

results = damage_model(img, conf=0.1)
results[0].show()
