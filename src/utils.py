import torch
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from config import CLASS_NAMES, IMG_SIZE

# ------------------ Classification Helpers ------------------
def load_classification_model(model_name, num_classes=26):
    """Load one of the four pre-trained models with custom head."""
    import torchvision.models as models
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def preprocess_image_for_classification(image):
    """Convert PIL or numpy image to tensor (224x224, normalized)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ------------------ YOLO Helpers ------------------
@st.cache_resource
def load_yolo_model(weights_path="yolov8n.pt"):
    """Load YOLO model (cached in Streamlit)."""
    model = YOLO(weights_path)
    return model

def run_yolo_inference(model, image, conf_threshold=0.5):
    """Run detection and return results with boxes, labels, scores."""
    results = model(image, conf=conf_threshold)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]  # from YOLO's internal names
        detections.append(([x1, y1, x2, y2], label, conf))
    return detections