# live_demo_transformer.py
# Real-time sign language recognition using Transformer model (250 signs).
#
# USAGE:
#   python src/live_demo_transformer.py
#
# Requires:
#   - hand_landmarker.task (MediaPipe model)
#   - models/sign_transformer_best.pth
#   - models/label_map_transformer.json
#
# Press 'q' to quit, 'c' to clear sentence, 'space' to add space.

import os
import json
import time
import math
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- CONFIG ----------
MODEL_PATH = "models/sign_transformer_best.pth"
LABEL_MAP_PATH = "models/label_map_transformer.json"
MEDIAPIPE_MODEL = "hand_landmarker.task"

BUFFER_SIZE = 64          # frames to accumulate before predicting (matches seq_len)
HOLD_FRAMES = 25          # stable frames before adding to sentence
CONFIDENCE_THRESHOLD = 0.3  # lower threshold since 250 classes
TOP_K = 5                 # show top-5 predictions

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# Face landmark indices (same as preprocessing)
FACE_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    95, 88, 178, 87, 14, 317, 402, 318, 324,
    1, 2, 98, 327, 4, 5, 195, 197,
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    46, 53, 52, 65, 55, 276, 283, 282, 295, 285,
]

POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

# ---------- LOAD MODEL ----------
print("Loading model...")

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

class_names = label_map["classes"]
n_classes = label_map["n_classes"]
n_features = label_map["input_dim"]
seq_len = label_map["seq_len"]
d_model = label_map["d_model"]
n_heads = label_map["n_heads"]
n_layers = label_map["n_layers"]
dim_ff = label_map["dim_ff"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dim_ff, n_classes, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        padding_mask = (x.abs().sum(dim=-1) == 0)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        x_masked = x * mask_expanded
        lengths = mask_expanded.sum(dim=1).clamp(min=1)
        x_mean = x_masked.sum(dim=1) / lengths
        x_for_max = x_masked.clone()
        x_for_max[padding_mask] = -1e9
        x_max = x_for_max.max(dim=1).values
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        return self.classifier(x_pooled)


model = SignTransformer(n_features, d_model, n_heads, n_layers, dim_ff, n_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("Model loaded: {} classes, Transformer (d={}, h={}, L={})".format(
    n_classes, d_model, n_heads, n_layers))

# ---------- SETUP MEDIAPIPE ----------
print("Loading MediaPipe...")

# Hand landmarker
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# We also need face and pose — use a holistic approach with separate detectors
# For face landmarks, we'll use the FaceLandmarker if available
# For now, we'll extract what we can from hands and fill face/pose with zeros
# unless we set up additional detectors

# Check if face_landmarker.task exists
FACE_MODEL = "face_landmarker.task"
POSE_MODEL = "pose_landmarker.task"

use_face = os.path.exists(FACE_MODEL)
use_pose = os.path.exists(POSE_MODEL)

face_detector = None
pose_detector = None

if use_face:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    face_options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    face_detector = FaceLandmarker.create_from_options(face_options)
    print("Face landmarker loaded.")
else:
    print("WARNING: {} not found. Face landmarks will be zeros.".format(FACE_MODEL))
    print("  Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")

if use_pose:
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    pose_options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    pose_detector = PoseLandmarker.create_from_options(pose_options)
    print("Pose landmarker loaded.")
else:
    print("WARNING: {} not found. Pose landmarks will be zeros.".format(POSE_MODEL))
    print("  Download: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


# ---------- FEATURE EXTRACTION ----------
def extract_frame_features(mp_image):
    """Extract 390 features from one frame using MediaPipe detectors."""
    features = np.zeros(n_features, dtype=np.float32)
    pos = 0

    # Hands: 42 landmarks x 3 = 126 features
    hand_result = hand_detector.detect(mp_image)
    left_hand = None
    right_hand = None

    if hand_result.hand_landmarks and hand_result.handedness:
        for i, handedness in enumerate(hand_result.handedness):
            label = handedness[0].category_name
            if label == "Left" and left_hand is None:
                left_hand = hand_result.hand_landmarks[i]
            elif label == "Right" and right_hand is None:
                right_hand = hand_result.hand_landmarks[i]

    # Left hand (21 landmarks)
    if left_hand:
        for lm in left_hand:
            features[pos] = lm.x
            features[pos + 1] = lm.y
            features[pos + 2] = lm.z
            pos += 3
    else:
        pos += 63

    # Right hand (21 landmarks)
    if right_hand:
        for lm in right_hand:
            features[pos] = lm.x
            features[pos + 1] = lm.y
            features[pos + 2] = lm.z
            pos += 3
    else:
        pos += 63

    # Face: 76 selected landmarks x 3 = 228 features
    if face_detector:
        face_result = face_detector.detect(mp_image)
        if face_result.face_landmarks:
            face_lm = face_result.face_landmarks[0]
            for idx in FACE_INDICES:
                if idx < len(face_lm):
                    features[pos] = face_lm[idx].x
                    features[pos + 1] = face_lm[idx].y
                    features[pos + 2] = face_lm[idx].z
                pos += 3
        else:
            pos += len(FACE_INDICES) * 3
    else:
        pos += len(FACE_INDICES) * 3

    # Pose: 12 selected landmarks x 3 = 36 features
    if pose_detector:
        pose_result = pose_detector.detect(mp_image)
        if pose_result.pose_landmarks:
            pose_lm = pose_result.pose_landmarks[0]
            for idx in POSE_INDICES:
                if idx < len(pose_lm):
                    features[pos] = pose_lm[idx].x
                    features[pos + 1] = pose_lm[idx].y
                    features[pos + 2] = pose_lm[idx].z
                pos += 3
        else:
            pos += len(POSE_INDICES) * 3
    else:
        pos += len(POSE_INDICES) * 3

    return features, (left_hand is not None or right_hand is not None)


# ---------- DRAWING ----------
def draw_hand(image, hand_landmarks, w, h, color=(0, 255, 0)):
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(image, (cx, cy), 3, color, -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, coords[start], coords[end], color, 1)


def draw_ui(frame, top_preds, sentence, fps, recording, frame_count):
    h, w, _ = frame.shape

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 140), (20, 20, 20), -1)

    if top_preds and recording:
        # Top prediction
        sign, conf = top_preds[0]
        color = (0, 255, 0) if conf > CONFIDENCE_THRESHOLD else (0, 150, 255)
        cv2.putText(frame, sign.upper(), (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Confidence bar
        bar_w = int(conf * 200)
        cv2.rectangle(frame, (15, 50), (215, 62), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, 50), (15 + bar_w, 62), color, -1)
        cv2.putText(frame, "{:.0f}%".format(conf * 100), (225, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Top-5
        y_offset = 80
        for i, (s, c) in enumerate(top_preds[:5]):
            bar = int(c * 120)
            cv2.rectangle(frame, (15, y_offset), (15 + bar, y_offset + 10), (80, 80, 80), -1)
            cv2.putText(frame, "{} ({:.0f}%)".format(s, c * 100), (145, y_offset + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
            y_offset += 12
    else:
        status = "Recording... ({}/{})".format(frame_count, BUFFER_SIZE) if recording else "Show your hand..."
        cv2.putText(frame, status, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # Sentence at bottom of top bar
    sentence_display = sentence[-60:] if len(sentence) > 60 else sentence
    cv2.putText(frame, ">> " + sentence_display, (15, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS
    cv2.putText(frame, "{:.0f} FPS".format(fps), (w - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # Frame buffer progress bar
    progress = min(frame_count / BUFFER_SIZE, 1.0)
    cv2.rectangle(frame, (w - 110, 35), (w - 10, 45), (60, 60, 60), -1)
    cv2.rectangle(frame, (w - 110, 35), (w - 110 + int(100 * progress), 45), (0, 200, 0), -1)

    # Controls
    cv2.putText(frame, "q=quit | c=clear | space=add space", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)


# ---------- MAIN LOOP ----------
print("")
print("=" * 55)
print("  LIVE SIGN LANGUAGE TRANSLATOR (Transformer, 250 signs)")
print("  Hold a sign for the buffer to fill, then see prediction.")
print("  Press 'q' to quit, 'c' to clear, space to add space.")
print("=" * 55)
print("")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

frame_buffer = []         # accumulate feature vectors
sentence = ""
last_added = ""
stable_count = 0
last_pred = None
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 30
    prev_time = now

    # Extract features from this frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    features, hand_detected = extract_frame_features(mp_image)

    # Draw hands on frame
    hand_result = hand_detector.detect(mp_image)
    if hand_result.hand_landmarks:
        for i, hand_lm in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name if hand_result.handedness else "?"
            color = (0, 255, 0) if handedness == "Right" else (255, 100, 0)
            draw_hand(frame, hand_lm, img_w, img_h, color)

    top_preds = None
    recording = hand_detected

    if hand_detected:
        frame_buffer.append(features)

        # Keep buffer at max seq_len (sliding window)
        if len(frame_buffer) > BUFFER_SIZE:
            frame_buffer = frame_buffer[-BUFFER_SIZE:]

        # Predict when we have enough frames
        if len(frame_buffer) >= 10:
            # Pad to seq_len if needed
            buf = np.array(frame_buffer, dtype=np.float32)
            n = buf.shape[0]
            if n < seq_len:
                pad = np.zeros((seq_len - n, n_features), dtype=np.float32)
                buf = np.concatenate([buf, pad], axis=0)
            else:
                # Center crop
                start = (n - seq_len) // 2
                buf = buf[start:start + seq_len]

            # Predict
            with torch.no_grad():
                input_tensor = torch.tensor(buf, dtype=torch.float32).unsqueeze(0)
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                top5_probs, top5_indices = probs.topk(TOP_K)
                top_preds = [(class_names[idx], prob.item()) for idx, prob in zip(top5_indices, top5_probs)]

            # Sentence building
            if top_preds:
                best_sign, best_conf = top_preds[0]
                if best_conf > CONFIDENCE_THRESHOLD:
                    if best_sign == last_pred:
                        stable_count += 1
                    else:
                        stable_count = 0
                        last_pred = best_sign

                    if stable_count == HOLD_FRAMES and best_sign != last_added:
                        sentence += best_sign + " "
                        last_added = best_sign
                        print("  Added: {} ({:.0f}%)  |  Sentence: {}".format(
                            best_sign, best_conf * 100, sentence.strip()))
    else:
        # No hand: clear buffer gradually
        if len(frame_buffer) > 0:
            frame_buffer = frame_buffer[max(0, len(frame_buffer) - 5):]
        stable_count = 0
        last_pred = None

    # Draw UI
    draw_ui(frame, top_preds, sentence.strip(), fps, recording, len(frame_buffer))

    cv2.imshow("Sign Language Translator (Transformer)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        sentence = ""
        last_added = ""
        stable_count = 0
        frame_buffer = []
        print("  Sentence cleared.")
    elif key == ord(" "):
        sentence += " "

cap.release()
cv2.destroyAllWindows()
print("")
print("Final sentence: {}".format(sentence.strip()))
print("Done!")