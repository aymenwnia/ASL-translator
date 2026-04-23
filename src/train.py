# train.py
# Train a sign language classifier on collected landmark data.
#
# USAGE:
#   python src/train.py
#
# Logs are written to logs/training.log — follow them live with:
#   tail -f logs/training.log
#
# Reads all CSVs from data/raw/, trains an MLP, saves model to models/

import os
import sys
import glob
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---------- CONFIG ----------
DATA_DIR = "data/raw"
MODEL_DIR = "models"
LOG_DIR = "logs"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.3
RANDOM_SEED = 42

# ---------- LOGGING SETUP ----------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "training.log")
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

# File handler (full log)
fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

# Console handler (also print to terminal)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

logger.addHandler(fh)
logger.addHandler(ch)

logger.info("Training log: {}".format(os.path.abspath(log_path)))
logger.info("PyTorch version: {}".format(torch.__version__))
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    logger.info("CUDA device: {}".format(torch.cuda.get_device_name(0)))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info("Using device: {}".format(device))

# ---------- LOAD DATA ----------
logger.info("")
logger.info("=" * 50)
logger.info("  STEP 1: Loading data")
logger.info("=" * 50)

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not csv_files:
    logger.error("No CSV files found in {}".format(DATA_DIR))
    logger.error("Run collect_data.py first to record samples.")
    sys.exit(1)

frames = []
for f in sorted(csv_files):
    df = pd.read_csv(f)
    frames.append(df)
    logger.info("  Loaded {}: {} samples".format(os.path.basename(f), len(df)))

data = pd.concat(frames, ignore_index=True)
logger.info("")
logger.info("Total samples: {}".format(len(data)))
logger.info("Signs found: {}".format(data["label"].unique().tolist()))
for label, count in data["label"].value_counts().items():
    logger.info("  {}: {} samples".format(label, count))

# ---------- PREPARE FEATURES ----------
logger.info("")
logger.info("=" * 50)
logger.info("  STEP 2: Preparing features")
logger.info("=" * 50)

X = data.drop(columns=["label"]).values.astype(np.float32)
y_raw = data["label"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_.tolist()
n_classes = len(class_names)

logger.info("Feature shape: {} (samples x landmarks)".format(X.shape))
logger.info("Classes ({}): {}".format(n_classes, class_names))

# Check for NaN/Inf
nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
if nan_count > 0 or inf_count > 0:
    logger.warning("Data contains {} NaN and {} Inf values! Replacing with 0.".format(nan_count, inf_count))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
)

logger.info("Train: {} | Val: {} | Test: {}".format(len(X_train), len(X_val), len(X_test)))

# Tensors -> device
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_t = torch.tensor(X_val).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)

# ---------- MODEL ----------
logger.info("")
logger.info("=" * 50)
logger.info("  STEP 3: Building model")
logger.info("=" * 50)


class SignMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train.shape[1]
model = SignMLP(input_dim, n_classes).to(device)
logger.info("Model architecture:")
logger.info(str(model))
total_params = sum(p.numel() for p in model.parameters())
logger.info("Total parameters: {}".format(total_params))

# ---------- TRAIN ----------
logger.info("")
logger.info("=" * 50)
logger.info("  STEP 4: Training ({} epochs)".format(EPOCHS))
logger.info("=" * 50)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0
best_epoch = 0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Training
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct += (outputs.argmax(dim=1) == batch_y).sum().item()
        total += batch_y.size(0)

    avg_train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        val_preds = val_outputs.argmax(dim=1)
        val_acc = (val_preds == y_val_t).float().mean().item()

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    epoch_time = time.time() - epoch_start

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "sign_mlp_best.pth"))

    # Log every epoch (useful for tail -f)
    logger.info(
        "Epoch {:3d}/{} | Train Loss: {:.4f} | Train Acc: {:.1f}% | Val Loss: {:.4f} | Val Acc: {:.1f}% | {:.2f}s".format(
            epoch + 1, EPOCHS, avg_train_loss, train_acc * 100,
            val_loss, val_acc * 100, epoch_time
        )
    )

elapsed = time.time() - start_time
logger.info("")
logger.info("Training complete in {:.1f}s".format(elapsed))
logger.info("Best validation accuracy: {:.1f}% (epoch {})".format(best_val_acc * 100, best_epoch))

# ---------- EVALUATE ON TEST SET ----------
logger.info("")
logger.info("=" * 50)
logger.info("  STEP 5: Test evaluation")
logger.info("=" * 50)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "sign_mlp_best.pth"), weights_only=True))
model.eval()

with torch.no_grad():
    test_outputs = model(X_test_t)
    test_preds = test_outputs.argmax(dim=1).cpu().numpy()

y_test_np = y_test
test_acc = (test_preds == y_test_np).mean()
logger.info("")
logger.info("Test accuracy: {:.1f}%".format(test_acc * 100))
logger.info("")
logger.info("Classification Report:")
report = classification_report(y_test_np, test_preds, target_names=class_names)
for line in report.split("\n"):
    logger.info(line)

# ---------- SAVE ARTIFACTS ----------
logger.info("")
logger.info("=" * 50)
logger.info("  STEP 6: Saving artifacts")
logger.info("=" * 50)

# Label mapping
label_map = {"classes": class_names, "input_dim": input_dim}
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)
logger.info("  Saved label_map.json")

# Confusion matrix
cm = confusion_matrix(y_test_np, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set) - Accuracy: {:.1f}%".format(test_acc * 100))
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
logger.info("  Saved confusion_matrix.png")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train")
ax1.plot(val_losses, label="Validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot([a * 100 for a in val_accuracies])
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation Accuracy")
ax2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
logger.info("  Saved training_curves.png")

logger.info("")
logger.info("=" * 50)
logger.info("  DONE!")
logger.info("  Model: {}/sign_mlp_best.pth".format(MODEL_DIR))
logger.info("  Labels: {}/label_map.json".format(MODEL_DIR))
logger.info("  Plots: {}/confusion_matrix.png, training_curves.png".format(MODEL_DIR))
logger.info("  Next: python src/live_demo.py")
logger.info("=" * 50)
