# m2.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import tifffile
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ─── Hyperparameters ────────────────────────────────────────────────
DATA_DIR      = "classified"
EXCEL_FILE    = "classified_groundwater.xlsx"
MODEL_OUT     = "model_satellite.pth"
BATCH_SIZE    = 32
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 3
IMG_SIZE      = 224
NUM_WORKERS   = 0
PATIENCE      = 7  # early stopping

# ─── Dataset ────────────────────────────────────────────────────────
class SatDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cls2idx = {"low": 0, "medium": 1, "high": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        path = os.path.join(self.img_dir, row["Image Name"])
        arr = tifffile.imread(path)

        if arr.ndim == 3 and arr.shape[0] == 6:
            arr = np.transpose(arr, (1, 2, 0))

        arr = arr[..., :6].astype(np.float32)

        # Clip values between 0 and 10000
        arr = np.clip(arr, 0, 10000)
        
        # Resize each band
        bands = [cv2.resize(arr[..., b], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) for b in range(6)]
        img = np.stack(bands, axis=2)

        if self.transform:
            img = self.transform(img)

        label = self.cls2idx[row["Depletion Category"]]
        return img, label

# ─── Transforms ─────────────────────────────────────────────────────
class ToTensorNorm:
    def __call__(self, img_np):
        img_np /= 10000.0  # scale to [0, 1]
        t = torch.from_numpy(img_np).permute(2, 0, 1)
        mean = torch.tensor([0.5] * 6).view(-1, 1, 1)
        std = torch.tensor([0.25] * 6).view(-1, 1, 1)
        t = (t - mean) / std
        return t

train_transform = transforms.Compose([
    ToTensorNorm(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

val_transform = transforms.Compose([
    ToTensorNorm()
])

# ─── Model ──────────────────────────────────────────────────────────
def get_model():
    model = models.resnet18(weights=None)  # Train from scratch
    model.conv1 = nn.Conv2d(6, model.conv1.out_channels,
                            kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride,
                            padding=model.conv1.padding,
                            bias=model.conv1.bias is not None)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# ─── Train + Eval ───────────────────────────────────────────────────
def train_and_evaluate():
    df = pd.read_excel(EXCEL_FILE)
    df = df.dropna(subset=["Image Name", "Depletion Category"])
    df["full_path"] = df["Image Name"].apply(lambda x: os.path.join(DATA_DIR, x))
    df = df[df["full_path"].apply(os.path.exists)].drop(columns="full_path")

    print(f"✅ Valid images found: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["Depletion Category"], random_state=42)

    train_loader = DataLoader(SatDataset(train_df, DATA_DIR, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(SatDataset(val_df, DATA_DIR, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_preds = np.concatenate(all_preds)
        train_labels = np.concatenate(all_labels)
        train_acc = (train_preds == train_labels).mean() * 100
        train_f1 = f1_score(train_labels, train_preds, average="macro") * 100

        # ─── Validation ──────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_preds.append(outputs.argmax(1).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc = (val_preds == val_labels).mean() * 100
        val_f1 = f1_score(val_labels, val_preds, average="macro") * 100

        print(f"\nEpoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | F1: {train_f1:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | F1: {val_f1:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"✅ New best model saved: {MODEL_OUT}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print("⚠️ Early stopping triggered.")
            break

    print(f"🏁 Training complete. Best Val Acc: {best_val_acc:.2f}%")

# ─── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_and_evaluate()
