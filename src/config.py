import os

# Paths (adjust to your actual locations)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLASSIFICATION_DIR = os.path.join(DATA_DIR, "classification")
DETECTION_DIR = os.path.join(DATA_DIR, "detection")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Classification parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 26   # from data.yaml (0..25)
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "traffic light", "stop sign", "bench", "bird", "cat", "dog", "horse", "cow",
    "elephant", "bottle", "cup", "bowl", "pizza", "cake", "chair", "couch",
    "potted plant", "bed"
]

# Detection parameters
YOLO_IMG_SIZE = 640
YOLO_EPOCHS = 50
YOLO_BATCH = 16
YOLO_DATA_YAML = os.path.join(DETECTION_DIR, "data.yaml")