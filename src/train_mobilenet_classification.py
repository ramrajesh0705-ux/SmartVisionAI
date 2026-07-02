import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import copy
from config import CLASSIFICATION_DIR, MODELS_DIR, BATCH_SIZE, EPOCHS, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Data Loaders (Same augmentations as before) ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # MobileNetV2 standard input
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------- Early Stopping (same as before) ----------
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0.001, path='best_mobilenet.pth'):
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

# ---------- MobileNetV2 Training Function ----------
def train_mobilenetv2():
    model_name = "mobilenetv2"
    print(f"\n{'='*40}")
    print(f"Training {model_name.upper()} ...")
    print(f"{'='*40}")

    # 1. Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # 2. FREEZE the entire convolutional base (all feature extractors)
    #    This is the "freeze base" requirement.
    for param in model.features.parameters():
        param.requires_grad = False
    print("✅ Frozen all feature extraction layers.")

    # 3. Add CUSTOM classification head for 25 classes
    #    MobileNetV2's classifier is: Dropout(0.2) -> Linear(1280, 1000)
    #    We replace it with our own lightweight head.
    #    IMPORTANT: Do NOT add huge Dense layers (like 1280->512->256)
    #    because the bottleneck already produces highly discriminative features.
    #    Large intermediate layers kill inference speed without improving accuracy.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),                       # Standard for MobileNetV2
        nn.Linear(model.last_channel, NUM_CLASSES)  # 1280 -> 25
    )
    print(f"✅ Replaced classifier head with: {model.classifier}")

    model = model.to(device)

    # 4. Loss, Optimizer (only trainable params are the new classifier)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, verbose=True
    )
    early_stopping = EarlyStopping(
        patience=8, 
        verbose=True, 
        path=os.path.join(MODELS_DIR, f"{model_name}_best_earlystop.pth")
    )

    # 5. Training loop
    best_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):
        # ---- Training Phase ----
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)  # total loss

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

        # Update scheduler and early stopping (using VALIDATION LOSS)
        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        # Log history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save by accuracy (backup)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best_acc.pth"))

        # Early stopping check
        if early_stopping.early_stop:
            print("⏹️ Early stopping triggered!")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    # Load the best model (based on lowest validation loss)
    if early_stopping.best_model_wts is not None:
        model.load_state_dict(early_stopping.best_model_wts)
        print("✅ Loaded early‑stopping best model (lowest val loss).")

    # Save final training history
    with open(os.path.join(MODELS_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Final evaluation of the best model
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

    # ---- INFERENCE SPEED OPTIMIZATIONS ----
    print("\n⚡ Optimizing model for inference speed ...")
    export_optimized_mobilenet(model, model_name)

    return model

# ---------- Inference Speed Optimization Functions ----------
def export_optimized_mobilenet(model, model_name):
    """
    Exports the trained MobileNetV2 in optimized formats for mobile/edge deployment.
    """
    model.eval().to('cpu')  # Move to CPU for tracing (standard practice)
    
    # 1. TorchScript (JIT) - AOT compilation for faster CPU inference
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, dummy_input)
    jit_path = os.path.join(MODELS_DIR, f"{model_name}_optimized_jit.pt")
    traced_script_module.save(jit_path)
    print(f"✅ TorchScript (JIT) model saved to: {jit_path}")

    # 2. Half-precision (FP16) conversion for GPU inference
    #    PyTorch can run FP16 on CUDA, doubling throughput.
    model.half()  # convert weights to FP16
    dummy_input_fp16 = torch.randn(1, 3, 224, 224).half()
    traced_fp16 = torch.jit.trace(model, dummy_input_fp16)
    fp16_path = os.path.join(MODELS_DIR, f"{model_name}_fp16_jit.pt")
    traced_fp16.save(fp16_path)
    print(f"✅ FP16 TorchScript model saved to: {fp16_path}")
    model.float()  # revert back to FP32 for safety

    # 3. ONNX export (for deployment to mobile via ONNX Runtime / CoreML)
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            os.path.join(MODELS_DIR, f"{model_name}.onnx"),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"✅ ONNX model saved to: {os.path.join(MODELS_DIR, f'{model_name}.onnx')}")
    except Exception as e:
        print(f"⚠️ ONNX export skipped: {e}")

if __name__ == "__main__":
    # Train only MobileNetV2, or loop over multiple
    train_mobilenetv2()
