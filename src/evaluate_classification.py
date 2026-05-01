import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from config import CLASSIFICATION_DIR, MODELS_DIR, NUM_CLASSES, CLASS_NAMES
from utils import load_classification_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(os.path.join(CLASSIFICATION_DIR, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for model_name in ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]:
    model = load_classification_model(model_name, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{model_name}_best.pth")))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    print(f"{model_name} Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))