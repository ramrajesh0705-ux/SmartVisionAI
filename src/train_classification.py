import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from config import CLASSIFICATION_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, NUM_CLASSES
from utils import load_classification_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(CLASSIFICATION_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(CLASSIFICATION_DIR, "val"), transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_one_model(model_name):
    print(f"\nTraining {model_name} ...")
    model = load_classification_model(model_name, NUM_CLASSES).to(device)
    for params in model.features.parameters():
        params.requires_grad = False
    
    num_ftrs = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, NUM_CLASSES)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step(val_acc)

        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best.pth"))

    # Save history
    with open(os.path.join(MODELS_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f)
    print(f"Finished {model_name}. Best val accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_one_model("vgg16") #for name in ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]:
