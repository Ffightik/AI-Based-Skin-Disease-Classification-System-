# train_skin_detector_v2.py
import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm

# ========== CONFIG ==========
DATA_DIR = "skin_detector_dataset"
SAVE_PATH = "skin_detector_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 12
BATCH_SIZE = 32
LR = 3e-4
IMG_SIZE = 224
NUM_WORKERS = 0   #

# ========== TRANSFORMS ==========
# strong phone-style augmentations to simulate real photos
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), shear=6),
    transforms.GaussianBlur(kernel_size=(3,7)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ========== DATASETS & LOADERS ==========
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print("Classes:", train_ds.classes)
print("Train size:", len(train_ds), "Val size:", len(val_ds))

# ========== MODEL ==========
model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ========== TRAIN LOOP ==========
best_val = 0.0

def train_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(train_loader, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def validate():
    model.eval()
    v_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Val"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            v_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return v_loss / total, correct / total

if __name__ == "__main__":
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate()
        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | Val: loss={val_loss:.4f} acc={val_acc:.4f}")
        scheduler.step(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[Saved] {SAVE_PATH} (val_acc={best_val:.4f})")
    print("Done. Best val acc:", best_val)
