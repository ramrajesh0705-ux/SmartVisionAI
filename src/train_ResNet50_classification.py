import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import copy          
from config import CLASSIFICATION_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, NUM_CLASSES
from utils import load_classification_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (unchanged)
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

# Early Stopping class (unchanged, but now imported copy)
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0.001, path='best_resnet50.pth'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased. Saving model ...')
        self.best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(self.best_model_wts, self.path)

def train_one_model(model_name):
    print(f"\nTraining {model_name} ...")
    model = load_classification_model(model_name, NUM_CLASSES).to(device)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 20 layers (only for ResNet50; adjust if using others)
    params_list = list(model.named_parameters())
    print(f"Total parameter groups: {len(params_list)}")
    # FIX 2: Added missing closing parenthesis
    for name, param in params_list[-20:]:
        param.requires_grad = True
        print(f"Unfrozen: {name}")

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(2048, 512),
        nn.ReLU(True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, NUM_CLASSES)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, verbose=True
    )
    early_stopping = EarlyStopping(patience=8, verbose=True, path=os.path.join(MODELS_DIR, f"{model_name}_best_earlystop.pth"))

    best_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],    # <-- FIX 4: Added val_loss to history
        "val_acc": []
    }

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

        # Validation – compute BOTH loss and accuracy
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0   # <-- FIX 5: Accumulate validation loss
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)          # <-- Compute loss
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_running_loss += loss.item() * images.size(0)   # total loss per batch

        val_acc = correct / total
        val_loss = val_running_loss / len(val_loader.dataset)   # average loss

        # FIX 6: Update scheduler with VALIDATION LOSS (not accuracy)
        scheduler.step(val_loss)

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)      # <-- Store val_loss
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        #  Call early stopping with val_loss
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered! Restoring best weights and stopping training.")
            model.load_state_dict(early_stopping.best_model_wts)
            # Update best_acc to the best saved model's accuracy (optional)
            # We'll use the early stopping's best model later.
            break

        # Save best model based on accuracy (keep as backup)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best_acc.pth"))

    # At the end, load the early stopping best model (loss‑based) if it exists
    if early_stopping.best_model_wts is not None:
        model.load_state_dict(early_stopping.best_model_wts)
        print("Loaded early‑stopping best model (lowest validation loss).")
    else:
        print("No early stopping checkpoint saved (unexpected).")

    # Save history
    with open(os.path.join(MODELS_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f)

    # Compute final validation accuracy of the loaded best model
    # (optional, for logging)
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
    final_acc = correct / total
    print(f"Finished {model_name}. Best val accuracy (early stopping model): {final_acc:.4f}")

if __name__ == "__main__":
    train_one_model("resnet50")   # You can change to loop over multiple models
