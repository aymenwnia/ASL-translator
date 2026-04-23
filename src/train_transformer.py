# train_transformer.py
# Train a Transformer model on the preprocessed Kaggle ASL Signs dataset.
#
# USAGE:
#   python src/train_transformer.py
#
# Reads numpy arrays from data/processed/, trains Transformer, saves to models/
# Follow logs: tail -f logs/train_transformer.log

import os
import sys
import json
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, top_k_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
LOG_DIR = "logs"

EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
WARMUP_EPOCHS = 5

# Transformer architecture
D_MODEL = 256        # embedding dimension
N_HEADS = 8          # attention heads
N_LAYERS = 4         # transformer encoder layers
DIM_FF = 512         # feedforward hidden size
DROPOUT = 0.3

PATIENCE = 12        # early stopping

# ---------- LOGGING ----------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger = logging.getLogger("train_transformer")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, "train_transformer.log"), mode="w", encoding="utf-8")
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

# ---------- DATA AUGMENTATION ----------
class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset with online data augmentation for training."""
    def __init__(self, X, y, augment=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # Random noise
            x += torch.randn_like(x) * 0.01

            # Random time shift (roll frames by a small amount)
            shift = torch.randint(-3, 4, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

            # Random frame dropout (zero out some frames)
            if torch.rand(1).item() < 0.2:
                n_drop = torch.randint(1, 5, (1,)).item()
                drop_indices = torch.randint(0, x.shape[0], (n_drop,))
                x[drop_indices] = 0.0

            # Random spatial scaling
            scale = 0.9 + torch.rand(1).item() * 0.2  # 0.9 to 1.1
            x *= scale

        return x, y


train_dataset = AugmentedDataset(X_train, y_train, augment=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# ---------- MODEL ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dim_ff, n_classes, dropout):
        super().__init__()
        # Project input features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head — use both mean and max pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # Create padding mask (frames that are all zeros)
        padding_mask = (x.abs().sum(dim=-1) == 0)  # (batch, seq_len)

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Pool: concatenate mean and max over time
        # Mask padded positions before pooling
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)
        x_masked = x * mask_expanded

        # Mean pooling (avoid dividing by zero)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        x_mean = x_masked.sum(dim=1) / lengths

        # Max pooling
        x_masked[padding_mask] = -1e9
        x_max = x_masked.max(dim=1).values

        x_pooled = torch.cat([x_mean, x_max], dim=1)
        return self.classifier(x_pooled)


model = SignTransformer(
    input_dim=n_features,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dim_ff=DIM_FF,
    n_classes=n_classes,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
logger.info("")
logger.info("Model: Transformer Encoder")
logger.info("  d_model: {}, heads: {}, layers: {}, ff: {}, dropout: {}".format(
    D_MODEL, N_HEADS, N_LAYERS, DIM_FF, DROPOUT))
logger.info("  Parameters: {:,}".format(total_params))
logger.info("  Data augmentation: noise, time-shift, frame-dropout, scaling")
logger.info("")

# ---------- OPTIMIZER WITH WARMUP ----------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Warmup + cosine annealing
def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# ---------- TRAIN ----------
logger.info("=" * 60)
logger.info("  Training ({} epochs, batch={}, lr={})".format(EPOCHS, BATCH_SIZE, LEARNING_RATE))
logger.info("=" * 60)

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
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
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

    # Step scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # Validate
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
    improved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "sign_transformer_best.pth"))
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

# ---------- TEST ----------
logger.info("")
logger.info("=" * 60)
logger.info("  Test Evaluation")
logger.info("=" * 60)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "sign_transformer_best.pth"), weights_only=True))
model.eval()

with torch.no_grad():
    test_outputs = model(X_test_t)
    test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
    test_preds = test_outputs.argmax(dim=1).cpu().numpy()

y_test_np = y_test
test_acc = (test_preds == y_test_np).mean()
top5_acc = top_k_accuracy_score(y_test_np, test_probs, k=5, labels=range(n_classes))

logger.info("")
logger.info("Test Top-1 accuracy: {:.1f}%".format(test_acc * 100))
logger.info("Test Top-5 accuracy: {:.1f}%".format(top5_acc * 100))

report_dict = classification_report(y_test_np, test_preds, output_dict=True, zero_division=0)
logger.info("")
logger.info("Classification Report (summary):")
logger.info("  Macro avg F1: {:.3f}".format(report_dict["macro avg"]["f1-score"]))
logger.info("  Weighted avg F1: {:.3f}".format(report_dict["weighted avg"]["f1-score"]))

# Worst and best signs
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

label_map = {
    "classes": [idx_to_sign[i] for i in range(n_classes)],
    "input_dim": n_features,
    "seq_len": seq_len,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
    "dim_ff": DIM_FF,
    "n_classes": n_classes,
    "model_type": "transformer",
}
with open(os.path.join(MODEL_DIR, "label_map_transformer.json"), "w") as f:
    json.dump(label_map, f, indent=2)
logger.info("  Saved label_map_transformer.json")

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
fig.suptitle("Transformer — Top-1: {:.1f}% | Top-5: {:.1f}%".format(test_acc * 100, top5_acc * 100))
fig.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "training_curves_transformer.png"), dpi=150)
logger.info("  Saved training_curves_transformer.png")

logger.info("")
logger.info("=" * 60)
logger.info("  DONE!")
logger.info("  Model: {}/sign_transformer_best.pth".format(MODEL_DIR))
logger.info("  Config: {}/label_map_transformer.json".format(MODEL_DIR))
logger.info("  Top-1: {:.1f}% | Top-5: {:.1f}%".format(test_acc * 100, top5_acc * 100))
logger.info("=" * 60)