from ultralytics import YOLO
from config import YOLO_DATA_YAML, MODELS_DIR
import os

model = YOLO(os.path.join(MODELS_DIR, "yolo_finetuned/weights/best.pt"))
metrics = model.val(data=YOLO_DATA_YAML, imgsz=640, conf=0.5, iou=0.5)
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")