import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (confusion_matrix, f1_score,
                             classification_report, roc_curve,
                             auc, label_binarize)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import torch
from torch import nn, optim
import timm
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(THIS_DIR)
DATA_DIR  = os.path.join(ROOT_DIR, 'dataset', 'archive')
IMG_DIR   = os.path.join(DATA_DIR, 'all_images')
CSV_PATH  = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
SAVE_PATH = os.path.join(ROOT_DIR, 'melanoma_best_v3.pth')
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"[INFO] IMG_DIR  : {IMG_DIR}")
print(f"[INFO] SAVE_PATH: {SAVE_PATH}")
print(f"[INFO] PLOTS    : {PLOTS_DIR}")

# ═══════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    torch.zeros(1).to(device)

# ═══════════════════════════════════════════════════════════════
# DATASET — фільтруємо лише наявні файли
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv(CSV_PATH)
label_mapping = {label: idx for idx, label in enumerate(sorted(df['dx'].unique()))}
idx_to_label  = {v: k for k, v in label_mapping.items()}
df['label']   = df['dx'].map(label_mapping)

df['img_path'] = df['image_id'].apply(
    lambda x: os.path.join(IMG_DIR, x + '.jpg'))
before = len(df)
df = df[df['img_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"\nLabel mapping : {label_mapping}")
print(f"Файлів: {len(df)} із {before} "
      f"(відсутніх: {before - len(df)})")
print(f"\nРозподіл:\n{df['label'].value_counts().sort_index().to_string()}\n")

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ═══════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, shear=8),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════
class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row['img_path']).convert('RGB')
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

# ═══════════════════════════════════════════════════════════════
# WEIGHTED SAMPLER
# ═══════════════════════════════════════════════════════════════
counts  = train_df['label'].value_counts().to_dict()
weights = [1.0 / counts[int(lbl)] for lbl in train_df['label']]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_ds = SkinDataset(train_df, train_transform)
val_ds   = SkinDataset(val_df,   val_transform)
test_ds  = SkinDataset(test_df,  val_transform)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False,
                          num_workers=0, pin_memory=True)

# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════
n_classes = len(label_mapping)
model = timm.create_model('tf_efficientnet_b4', pretrained=True,
                          num_classes=n_classes)
model.to(device)

# ═══════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════
cw = compute_class_weight('balanced',
                          classes=np.unique(df['label']),
                          y=df['label'].values)
cw_tensor = torch.tensor(cw, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=cw_tensor)
print(f"Класові ваги: { {idx_to_label[i]: round(cw[i],2) for i in range(n_classes)} }")

# ═══════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# ═══════════════════════════════════════════════════════════════
# EVALUATE HELPER
# ═══════════════════════════════════════════════════════════════
def evaluate(loader, desc="Val"):
    model.eval()
    all_lbl, all_pred, all_prob = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            out   = model(images)
            loss  = criterion(out, labels)
            total_loss += loss.item()
            prob  = torch.softmax(out, dim=1)
            pred  = out.argmax(dim=1)
            all_lbl.extend(labels.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc      = 100.0 * (np.array(all_lbl) == np.array(all_pred)).mean()
    return (np.array(all_lbl), np.array(all_pred),
            np.array(all_prob), avg_loss, acc)

# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    num_epochs   = 15
    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        scheduler.step()

        _, _, _, val_loss, val_acc = evaluate(val_loader,
                                              f"Epoch {epoch+1} [Val]")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] "
              f"| Train: {train_loss:.4f} "
              f"| Val: {val_loss:.4f} "
              f"| Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   💾 Saved! Best: {best_val_acc:.2f}%")

    print(f"\n✅ Done! Best val acc: {best_val_acc:.2f}%")

    # ═══════════════════════════════════════════════════════════
    # TEST SET EVALUATION
    # ═══════════════════════════════════════════════════════════
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=False))
    all_lbl, all_pred, all_prob, test_loss, test_acc = \
        evaluate(test_loader, "Test")

    class_names = [idx_to_label[i] for i in range(n_classes)]
    f1_macro    = f1_score(all_lbl, all_pred, average='macro',    zero_division=0)
    f1_weighted = f1_score(all_lbl, all_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"Test Accuracy : {test_acc:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"F1-macro      : {f1_macro:.4f}")
    print(f"F1-weighted   : {f1_weighted:.4f}")
    print(f"{'='*50}")
    print(classification_report(all_lbl, all_pred,
                                target_names=class_names, zero_division=0))

    # ── Plot 1: Loss & Accuracy ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, num_epochs + 1)

    axes[0].plot(ep, train_losses, 'o-', color='#3266ad',
                 label='Train loss', markersize=4)
    axes[0].plot(ep, val_losses,   's--', color='#e24b4a',
                 label='Val loss',   markersize=4)
    axes[0].set_title('Loss curves')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, val_accs, 'o-', color='#1d9e75', markersize=4)
    axes[1].axhline(best_val_acc, color='#e24b4a', linestyle='--',
                    label=f'Best {best_val_acc:.1f}%')
    axes[1].set_title('Validation accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy %')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_curves.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: {PLOTS_DIR}/loss_curves.png")

    # ── Plot 2: Confusion matrix ─────────────────────────────
    cm = confusion_matrix(all_lbl, all_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Test Set')
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=10,
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: {PLOTS_DIR}/confusion_matrix.png")

    # ── Plot 3: F1 per class ─────────────────────────────────
    f1_cls = f1_score(all_lbl, all_pred, average=None, zero_division=0)
    colors = ['#e24b4a' if v < 0.75 else '#ba7517' if v < 0.85
              else '#1d9e75' for v in f1_cls]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(class_names, f1_cls, color=colors, height=0.6)
    ax.axvline(f1_weighted, color='#3266ad', linestyle='--',
               label=f'Weighted avg {f1_weighted:.3f}')
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('F1-score')
    ax.set_title('F1-score per Class — Test Set')
    for bar, val in zip(bars, f1_cls):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)
    ax.legend(); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'f1_per_class.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: {PLOTS_DIR}/f1_per_class.png")

    # ── Plot 4: ROC-AUC ──────────────────────────────────────
    y_bin = label_binarize(all_lbl, classes=list(range(n_classes)))
    roc_colors = ['#3266ad','#1d9e75','#e24b4a','#ba7517',
                  '#534AB7','#0F6E56','#993C1D']
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=2, color=roc_colors[i],
                label=f'{class_names[i]} AUC={roc_auc:.3f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — One-vs-Rest (Test Set)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: {PLOTS_DIR}/roc_curves.png")

    print(f"\n📁 Всі графіки: {PLOTS_DIR}")
    print(f"\n{'='*50}")
    print(f"ПІДСУМОК")
    print(f"{'='*50}")
    print(f"Test Accuracy : {test_acc:.2f}%")
    print(f"F1-macro      : {f1_macro:.4f}")
    print(f"F1-weighted   : {f1_weighted:.4f}")
    print(f"Macro AUC-ROC : {np.mean(aucs):.4f}")
    print(f"{'='*50}")