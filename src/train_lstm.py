# train_lstm.py
# Train an LSTM model on the preprocessed Kaggle ASL Signs dataset.
#
# USAGE:
#   python src/train_lstm.py
#
# Reads numpy arrays from data/processed/, trains LSTM, saves model to models/
# Follow logs: tail -f logs/train_lstm.log

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
LOG_DIR = "logs"

EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 10  # early stopping patience

# ---------- LOGGING ----------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger = logging.getLogger("train_lstm")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, "train_lstm.log"), mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(ch)

# ---------- DEVICE ----------
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

# ---------- LOAD DATA ----------
logger.info("Loading preprocessed data...")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

with open(os.path.join(DATA_DIR, "config.json"), "r") as f:
    config = json.load(f)

n_classes = config["n_classes"]
seq_len = config["seq_len"]
n_features = config["n_features"]
idx_to_sign = {int(k): v for k, v in config["idx_to_sign"].items()}

logger.info("Train: {} | Val: {} | Test: {}".format(len(X_train), len(X_val), len(X_test)))
logger.info("Shape: seq_len={}, features={}, classes={}".format(seq_len, n_features, n_classes))

# Tensors
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_t = torch.tensor(X_val).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE,
    shuffle=False
)
# ---------- MODEL ----------
class SignLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden states from both directions
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        h_forward = h_n[-2]  # last layer forward
        h_backward = h_n[-1]  # last layer backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        h_combined = self.dropout(h_combined)
        return self.fc(h_combined)


model = SignLSTM(n_features, HIDDEN_SIZE, NUM_LAYERS, n_classes, DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters())
logger.info("")
logger.info("Model: Bidirectional LSTM")
logger.info("  Hidden: {}, Layers: {}, Dropout: {}".format(HIDDEN_SIZE, NUM_LAYERS, DROPOUT))
logger.info("  Parameters: {:,}".format(total_params))
logger.info("")

# ---------- TRAIN ----------
logger.info("=" * 60)
logger.info("  Training ({} epochs, batch_size={}, lr={})".format(EPOCHS, BATCH_SIZE, LEARNING_RATE))
logger.info("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_losses = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0
best_epoch = 0
no_improve = 0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Train
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        correct += (outputs.argmax(dim=1) == batch_y).sum().item()
        total += batch_y.size(0)

    avg_train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_train_loss)

    # Validate
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item()
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Learning rate schedule
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = time.time() - epoch_start

    # Save best
    improved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "sign_lstm_best.pth"))
        improved = " *"
    else:
        no_improve += 1

    logger.info(
        "Epoch {:3d}/{} | Train: {:.4f} ({:.1f}%) | Val: {:.4f} ({:.1f}%) | lr: {:.6f} | {:.1f}s{}".format(
            epoch + 1, EPOCHS, avg_train_loss, train_acc * 100,
            val_loss, val_acc * 100, current_lr, epoch_time, improved
        )
    )

    # Early stopping
    if no_improve >= PATIENCE:
        logger.info("Early stopping at epoch {} (no improvement for {} epochs)".format(epoch + 1, PATIENCE))
        break

elapsed = time.time() - start_time
logger.info("")
logger.info("Training complete in {:.1f}s".format(elapsed))
logger.info("Best val accuracy: {:.1f}% (epoch {})".format(best_val_acc * 100, best_epoch))

test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------- TEST ----------
logger.info("")
logger.info("=" * 60)
logger.info("  Test Evaluation")
logger.info("=" * 60)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "sign_lstm_best.pth"), weights_only=True))
model.eval()

all_probs = []
all_preds = []

with torch.no_grad():
    for batch_X, _ in test_loader:
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)

test_probs = np.concatenate(all_probs)
test_preds = np.concatenate(all_preds)
y_test_np = y_test
test_acc = (test_preds == y_test_np).mean()

# Top-5 accuracy
top5_acc = top_k_accuracy_score(y_test_np, test_probs, k=5, labels=range(n_classes))

logger.info("")
logger.info("Test Top-1 accuracy: {:.1f}%".format(test_acc * 100))
logger.info("Test Top-5 accuracy: {:.1f}%".format(top5_acc * 100))

# Per-class report (top 10 worst)
report_dict = classification_report(y_test_np, test_preds, output_dict=True, zero_division=0)
logger.info("")
logger.info("Classification Report (summary):")
logger.info("  Macro avg F1: {:.3f}".format(report_dict["macro avg"]["f1-score"]))
logger.info("  Weighted avg F1: {:.3f}".format(report_dict["weighted avg"]["f1-score"]))

# Find worst performing classes
class_f1 = []
for i in range(n_classes):
    key = str(i)
    if key in report_dict:
        class_f1.append((idx_to_sign[i], report_dict[key]["f1-score"], report_dict[key]["support"]))

class_f1.sort(key=lambda x: x[1])
logger.info("")
logger.info("10 worst-performing signs:")
for name, f1, support in class_f1[:10]:
    logger.info("  {}: F1={:.3f} (support={})".format(name, f1, int(support)))

logger.info("")
logger.info("10 best-performing signs:")
for name, f1, support in class_f1[-10:]:
    logger.info("  {}: F1={:.3f} (support={})".format(name, f1, int(support)))

# ---------- SAVE ----------
logger.info("")
logger.info("=" * 60)
logger.info("  Saving artifacts")
logger.info("=" * 60)

# Save label map for inference
label_map = {
    "classes": [idx_to_sign[i] for i in range(n_classes)],
    "input_dim": n_features,
    "seq_len": seq_len,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "n_classes": n_classes,
    "model_type": "lstm",
}
with open(os.path.join(MODEL_DIR, "label_map_lstm.json"), "w") as f:
    json.dump(label_map, f, indent=2)
logger.info("  Saved label_map_lstm.json")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, label="Train")
ax1.plot(val_losses, label="Validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot([a * 100 for a in val_accuracies])
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation Accuracy")
ax2.grid(True, alpha=0.3)
fig.suptitle("LSTM Training — Top-1: {:.1f}% | Top-5: {:.1f}%".format(test_acc * 100, top5_acc * 100))
fig.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "training_curves_lstm.png"), dpi=150)
logger.info("  Saved training_curves_lstm.png")

logger.info("")
logger.info("=" * 60)
logger.info("  DONE!")
logger.info("  Model: {}/sign_lstm_best.pth".format(MODEL_DIR))
logger.info("  Config: {}/label_map_lstm.json".format(MODEL_DIR))
logger.info("  Top-1: {:.1f}% | Top-5: {:.1f}%".format(test_acc * 100, top5_acc * 100))
logger.info("=" * 60)