import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import copy
import math
from torch.cuda.amp import GradScaler, autocast
from config import CLASSIFICATION_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Advanced Augmentations --------------------
# We use a combination of:
# - RandAugment (torchvision v0.10+) or TrivialAugmentWide
# - RandomErasing (Cutout)
# - Standard normalization
try:
    from torchvision.transforms import RandAugment, TrivialAugmentWide
except ImportError:
    # Fallback for older torchvision versions
    RandAugment = None
    TrivialAugmentWide = None

def get_train_transforms():
    # Base transforms (random crop, flip, color jitter)
    transforms_list = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    ]
    
    # Advanced augmentation (if available)
    if RandAugment is not None:
        transforms_list.append(RandAugment(num_ops=2, magnitude=9))   # RandAugment
    elif TrivialAugmentWide is not None:
        transforms_list.append(TrivialAugmentWide())                  # TrivialAugmentWide
    else:
        # Fallback: simple augmentation
        pass

    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))           # Cutout
    ]
    return transforms.Compose(transforms_list)

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# -------------------- Data Loaders --------------------
train_dataset = datasets.ImageFolder(
    os.path.join(CLASSIFICATION_DIR, "train"),
    transform=get_train_transforms()
)
val_dataset = datasets.ImageFolder(
    os.path.join(CLASSIFICATION_DIR, "val"),
    transform=get_val_transforms()
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

# -------------------- Early Stopping --------------------
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0.001, path='best_model.pth'):
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

# -------------------- Custom Classification Head with BatchNorm --------------------
class EfficientNetHead(nn.Module):
    """Custom head for EfficientNet: adds a BatchNorm layer before the final linear."""
    def __init__(self, in_features, num_classes, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn = nn.BatchNorm1d(in_features)          # BatchNorm on the feature vector
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.bn(x)
        x = self.fc(x)
        return x

# -------------------- Training Function --------------------
def train_efficientnetb0():
    model_name = "efficientnetb0"
    print(f"\n{'='*50}")
    print(f"Training {model_name.upper()} with Mixed Precision & Advanced Augmentations")
    print(f"{'='*50}")

    # 1. Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # 2. Replace classifier head with our custom head (with BatchNorm)
    in_features = model.classifier[1].in_features   # EfficientNet's classifier is (Dropout, Linear)
    model.classifier = EfficientNetHead(in_features, NUM_CLASSES, dropout_rate=0.2)
    model = model.to(device)

    # 3. (Optional) Fine‑tune all layers – we will use a differential learning rate.
    #    Freeze only the first few blocks? We'll just use a lower LR for the base.
    #    We'll train all parameters but with different groups.
    #    Group 1: features (base) – lower LR
    #    Group 2: classifier head – higher LR
    base_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ])

    # 4. Learning Rate Scheduler: CosineAnnealingLR with warmup
    def warmup_cosine_lr(epoch, warmup_epochs=5, total_epochs=EPOCHS, base_lr=1e-4):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            return base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: warmup_cosine_lr(epoch, warmup_epochs=5, total_epochs=EPOCHS, base_lr=1e-4) / 1e-4
    )
    # (Alternatively, use CosineAnnealingLR + Warmup separately)

    # 5. Loss & Early Stopping
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=8, 
        verbose=True, 
        path=os.path.join(MODELS_DIR, f"{model_name}_best_earlystop.pth")
    )

    # 6. Mixed Precision (AMP) setup
    scaler = GradScaler()

    # 7. Training loop
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        # ---- Training Phase ----
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation Phase ----
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_running_loss += loss.item() * images.size(0)

        val_acc = correct / total
        val_loss = val_running_loss / len(val_loader.dataset)

        # Update scheduler (by epoch)
        scheduler.step()

        # Early stopping
        early_stopping(val_loss, model)

        # Log
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.2e}")

        # Save best by accuracy (backup)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best_acc.pth"))

        # Check early stop
        if early_stopping.early_stop:
            print("⏹️ Early stopping triggered!")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    # Load best model (by val loss)
    if early_stopping.best_model_wts is not None:
        model.load_state_dict(early_stopping.best_model_wts)
        print("✅ Loaded early‑stopping best model (lowest val loss).")

    # Save final history
    with open(os.path.join(MODELS_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Final evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    final_acc = correct / total
    print(f"\n✅ Finished {model_name}. Final best val accuracy: {final_acc:.4f}")
    
    return model

if __name__ == "__main__":
    train_efficientnetb0()
