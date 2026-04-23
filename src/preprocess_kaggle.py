# preprocess_kaggle.py
# Preprocess the Kaggle ASL Signs dataset for LSTM training.
#
# USAGE:
#   python src/preprocess_kaggle.py
#
# Reads parquet files, extracts hand landmarks, pads sequences,
# and saves numpy arrays to data/processed/

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SIGN_MAP = os.path.join(DATA_DIR, "sign_to_prediction_index_map.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = "logs"

# Sequence length — pad or truncate all sequences to this
SEQ_LEN = 64

# Which landmarks to use (hand only = 21 per hand x 3 coords)
# Left hand: columns x_left_hand_0..20, y_left_hand_0..20, z_left_hand_0..20
# Right hand: columns x_right_hand_0..20, y_right_hand_0..20, z_right_hand_0..20
# = 42 landmarks x 3 coords = 126 features per frame

# ---------- LOGGING ----------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger("preprocess")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, "preprocess.log"), mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(ch)

# ---------- LOAD METADATA ----------
logger.info("Loading metadata...")
train_df = pd.read_csv(TRAIN_CSV)
logger.info("Total sequences: {}".format(len(train_df)))
logger.info("Unique signs: {}".format(train_df["sign"].nunique()))
logger.info("Unique participants: {}".format(train_df["participant_id"].nunique()))

with open(SIGN_MAP, "r") as f:
    sign_to_idx = json.load(f)
n_classes = len(sign_to_idx)
logger.info("Sign classes: {}".format(n_classes))

# Build column names for landmarks (hands + face + pose)
# Following the Kaggle competition winning approach:
# - 21 left hand + 21 right hand = 42 hand landmarks
# - 76 face landmarks (lips, nose, eyes — most expressive)
# - 12 pose landmarks (6 per arm: shoulders, elbows, wrists)
# Total: 130 landmarks x 3 coords = 390 features per frame

# Face landmark indices that matter for sign language
FACE_INDICES = [
    # Lips outer
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    # Lips inner
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    95, 88, 178, 87, 14, 317, 402, 318, 324,
    # Nose
    1, 2, 98, 327, 4, 5, 195, 197,
    # Left eye
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    # Right eye
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    # Eyebrows
    46, 53, 52, 65, 55, 276, 283, 282, 295, 285,
]

# Pose landmark indices (upper body only)
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

def get_landmark_columns():
    cols = []
    # Hands (all 21 landmarks each)
    for hand in ["left_hand", "right_hand"]:
        for i in range(21):
            for axis in ["x", "y", "z"]:
                cols.append("{}_{}_{}" .format(axis, hand, i))
    # Face (selected landmarks)
    for i in FACE_INDICES:
        for axis in ["x", "y", "z"]:
            cols.append("{}_{}_{}".format(axis, "face", i))
    # Pose (upper body)
    for i in POSE_INDICES:
        for axis in ["x", "y", "z"]:
            cols.append("{}_{}_{}".format(axis, "pose", i))
    return cols

LANDMARK_COLS = get_landmark_columns()
N_FEATURES = len(LANDMARK_COLS)  # 390
logger.info("Features per frame: {} (42 hand + {} face + {} pose landmarks x 3 coords)".format(
    N_FEATURES, len(FACE_INDICES), len(POSE_INDICES)))
logger.info("Sequence length: {} frames".format(SEQ_LEN))

# ---------- BUILD LANDMARK FILTER ----------
# We need to filter and pivot the long-format parquet data
# Each parquet has columns: frame, type, landmark_index, x, y, z

# Build a set of (type, landmark_index) pairs we want to keep
WANTED_LANDMARKS = []
# Hands: all 21 landmarks
for hand in ["left_hand", "right_hand"]:
    for i in range(21):
        WANTED_LANDMARKS.append((hand, i))
# Face: selected landmarks
for i in FACE_INDICES:
    WANTED_LANDMARKS.append(("face", i))
# Pose: selected landmarks
for i in POSE_INDICES:
    WANTED_LANDMARKS.append(("pose", i))

# Create a mapping from (type, landmark_index) to feature position
LANDMARK_TO_POS = {}
for pos, (lm_type, lm_idx) in enumerate(WANTED_LANDMARKS):
    LANDMARK_TO_POS[(lm_type, lm_idx)] = pos

N_LANDMARKS = len(WANTED_LANDMARKS)  # 130
N_FEATURES = N_LANDMARKS * 3  # 390
logger.info("Selected landmarks: {} -> {} features per frame".format(N_LANDMARKS, N_FEATURES))


def process_parquet(parquet_path):
    """Read a long-format parquet and convert to (n_frames, N_FEATURES) array."""
    df = pd.read_parquet(parquet_path)

    # Fast filter: merge with wanted landmarks lookup table
    df["pos"] = df.set_index(["type", "landmark_index"]).index.map(LANDMARK_TO_POS)
    df = df.dropna(subset=["pos"])

    if len(df) == 0:
        return None

    df["pos"] = df["pos"].astype(int)

    # Get unique sorted frames
    frames = sorted(df["frame"].unique())
    n_frames = len(frames)
    if n_frames == 0:
        return None

    frame_to_idx = {f: i for i, f in enumerate(frames)}
    frame_indices = df["frame"].map(frame_to_idx).values
    pos_values = df["pos"].values

    # Build output array with vectorized indexing
    output = np.zeros((n_frames, N_FEATURES), dtype=np.float32)
    x_vals = df["x"].fillna(0).values.astype(np.float32)
    y_vals = df["y"].fillna(0).values.astype(np.float32)
    z_vals = df["z"].fillna(0).values.astype(np.float32)

    output[frame_indices, pos_values * 3] = x_vals
    output[frame_indices, pos_values * 3 + 1] = y_vals
    output[frame_indices, pos_values * 3 + 2] = z_vals

    return output


# ---------- PROCESS ----------
logger.info("")
logger.info("Processing parquet files...")
start_time = time.time()

X_all = []
y_all = []
skipped = 0
errors = 0

for idx, row in train_df.iterrows():
    parquet_path = os.path.join(DATA_DIR, row["path"])
    sign = row["sign"]
    label = sign_to_idx[sign]

    try:
        features = process_parquet(parquet_path)

        if features is None:
            skipped += 1
            continue

        n_frames = features.shape[0]

        # Pad or truncate to SEQ_LEN
        if n_frames >= SEQ_LEN:
            # Take center crop
            start = (n_frames - SEQ_LEN) // 2
            features = features[start:start + SEQ_LEN]
        else:
            # Pad with zeros
            pad = np.zeros((SEQ_LEN - n_frames, N_FEATURES), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)

        X_all.append(features)
        y_all.append(label)

    except Exception as e:
        errors += 1
        if errors <= 5:
            logger.warning("Error processing {}: {}".format(parquet_path, e))

    if (idx + 1) % 5000 == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        eta = (len(train_df) - idx - 1) / rate
        logger.info("  {}/{} processed | {:.0f}/s | ETA: {:.0f}s | skipped: {} | errors: {}".format(
            idx + 1, len(train_df), rate, eta, skipped, errors
        ))

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.int64)

elapsed = time.time() - start_time
logger.info("")
logger.info("Processing complete in {:.1f}s".format(elapsed))
logger.info("Final shape: X={}, y={}".format(X.shape, y.shape))
logger.info("Skipped: {}, Errors: {}".format(skipped, errors))

# ---------- SPLIT ----------
logger.info("")
logger.info("Splitting data...")

from sklearn.model_selection import train_test_split

# Get unique participant IDs for each sample
participant_ids = train_df.iloc[:len(y)]["participant_id"].values

# Split by participant for better generalization
unique_participants = np.unique(participant_ids)
np.random.seed(42)
np.random.shuffle(unique_participants)

n_test = max(2, len(unique_participants) // 5)  # ~20% participants for test
test_participants = set(unique_participants[:n_test])
train_participants = set(unique_participants[n_test:])

train_mask = np.array([p in train_participants for p in participant_ids])
test_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Further split train into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

logger.info("Train: {} | Val: {} | Test: {}".format(len(X_train), len(X_val), len(X_test)))
logger.info("Test participants: {}".format(test_participants))

# ---------- SAVE ----------
logger.info("")
logger.info("Saving to {}...".format(OUTPUT_DIR))

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# Save config for training script
config = {
    "n_classes": n_classes,
    "seq_len": SEQ_LEN,
    "n_features": N_FEATURES,
    "sign_to_idx": sign_to_idx,
    "idx_to_sign": {v: k for k, v in sign_to_idx.items()},
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

logger.info("Saved: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy, config.json")
logger.info("")
logger.info("DONE! Next step: python src/train_lstm.py")