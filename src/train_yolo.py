from ultralytics import YOLO
from config import YOLO_DATA_YAML, YOLO_EPOCHS, YOLO_BATCH, YOLO_IMG_SIZE, MODELS_DIR

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")   # you can also use yolov8s.pt for better accuracy

# Train
results = model.train(
    data=YOLO_DATA_YAML,
    epochs=YOLO_EPOCHS,
    imgsz=YOLO_IMG_SIZE,
    batch=YOLO_BATCH,
    device=0,                # use GPU if available
    project=MODELS_DIR,
    name="yolo_finetuned",
    exist_ok=True
)

# Save final model
model.export(format="torchscript")  # optional: convert for faster inference
print("YOLO training completed.")