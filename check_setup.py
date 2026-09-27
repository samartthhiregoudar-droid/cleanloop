"""Checks that Python packages, YOLO, and the webcam all work."""
import sys

import cv2
import torch
import yaml
from ultralytics import YOLO

print(f"Python:   {sys.version.split()[0]}")
print(f"OpenCV:   {cv2.__version__}")
print(f"PyTorch:  {torch.__version__}")
print(f"GPU (CUDA) available: {torch.cuda.is_available()}")

with open("config.yaml") as f:
    config = yaml.safe_load(f)
print(f"Config loaded: camera = {config['camera']['name']}")

print("\nLoading YOLO11n (downloads ~6 MB the first time)...")
model = YOLO(config["detection"]["model"])
print("YOLO loaded OK")

print("\nTesting webcam...")
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
if ok:
    print(f"Webcam OK, frame size: {frame.shape[1]}x{frame.shape[0]}")
    results = model(frame, verbose=False)
    names = [model.names[int(c)] for c in results[0].boxes.cls]
    print(f"Detected in test frame: {names or 'nothing'}")
else:
    print("Webcam not found (fine if you will use video files instead)")

print("\nSetup check complete!")
