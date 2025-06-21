import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np

# --- Config ---
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOW_THRESHOLD = 2.0
MEDIUM_THRESHOLD = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classify DTWL to classes ---
def classify_by_dtwl(dtwl):
    if dtwl < LOW_THRESHOLD:
        return 'low'
    elif dtwl <= MEDIUM_THRESHOLD:
        return 'medium'
    else:
        return 'high'

# --- Load Dataset ---
df = pd.read_excel('dtwl_classified.xlsx')
df = df.dropna(subset=["Latitude", "Longitude", "DTWL"])
df["class"] = df["DTWL"].apply(classify_by_dtwl)

# Encode class labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["class"])

# Feature & label selection
features = df[["Latitude", "Longitude"]].values
labels = df["label"].values

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# --- Dataset class ---
class LocationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = LocationDataset(X_train, y_train)
test_dataset = LocationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --- Model ---
class LocationClassifier(nn.Module):
    def __init__(self):
        super(LocationClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# --- Train ---
model = LocationClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for X_batch, y_batch in loop:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    scheduler.step()
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    # --- Eval ---
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f} | "
          f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | "
          f"Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")

# --- Save model and tools ---
torch.save({
    "model_state_dict": model.state_dict(),
    "label_encoder": label_encoder,
    "scaler": scaler
}, "model1.pth")
print("✅ Improved model saved as model1.pth")
