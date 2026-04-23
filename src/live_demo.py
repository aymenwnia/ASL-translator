# live_demo.py
# Real-time sign language recognition from webcam.
#
# USAGE:
#   python src/live_demo.py
#
# Press 'q' to quit, 'c' to clear accumulated text, 'space' to add a space.

import os
import json
import time
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- CONFIG ----------
MODEL_PATH = "models/sign_mlp_best.pth"
LABEL_MAP_PATH = "models/label_map.json"
MEDIAPIPE_MODEL = "hand_landmarker.task"

BUFFER_SIZE = 15          # frames to average prediction over
HOLD_FRAMES = 20          # how many stable frames before adding to sentence
CONFIDENCE_THRESHOLD = 0.7  # minimum confidence to accept a prediction

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ---------- LOAD MODEL ----------
print("Loading model...")

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

class_names = label_map["classes"]
input_dim = label_map["input_dim"]
n_classes = len(class_names)


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


model = SignMLP(input_dim, n_classes)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()
print("Model loaded: {} classes -> {}".format(n_classes, class_names))

# ---------- SETUP MEDIAPIPE ----------
print("Loading MediaPipe...")
base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# ---------- HELPERS ----------
def draw_hand(image, hand_landmarks, w, h):
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, coords[start], coords[end], (0, 200, 0), 2)


def landmarks_to_tensor(hand_landmarks):
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return torch.tensor([features], dtype=torch.float32)


def draw_ui(frame, prediction, confidence, sentence, fps, hand_detected):
    h, w, _ = frame.shape

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 110), (30, 30, 30), -1)

    if hand_detected and prediction:
        # Current prediction
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 180, 255)
        cv2.putText(frame, prediction.upper(), (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        # Confidence bar
        bar_w = int(confidence * 200)
        cv2.rectangle(frame, (15, 55), (215, 70), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, 55), (15 + bar_w, 70), color, -1)
        cv2.putText(frame, "{:.0f}%".format(confidence * 100), (225, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Show your hand...", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # Sentence
    cv2.putText(frame, "Text: " + sentence, (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # FPS
    cv2.putText(frame, "{:.0f} FPS".format(fps), (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Bottom instructions
    cv2.putText(frame, "q=quit | c=clear | space=add space", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


# ---------- MAIN LOOP ----------
print("")
print("=" * 50)
print("  LIVE SIGN LANGUAGE DEMO")
print("  Hold a sign steady to add it to the sentence.")
print("  Press 'q' to quit, 'c' to clear, space to add space.")
print("=" * 50)
print("")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

prediction_buffer = collections.deque(maxlen=BUFFER_SIZE)
sentence = ""
last_added = ""
stable_count = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # FPS calculation
    now = time.time()
    fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 30
    prev_time = now

    # Detect hand
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    hand_detected = len(result.hand_landmarks) > 0 if result.hand_landmarks else False
    current_pred = None
    current_conf = 0.0

    if hand_detected:
        hand_lm = result.hand_landmarks[0]
        draw_hand(frame, hand_lm, img_w, img_h)

        # Predict
        with torch.no_grad():
            input_tensor = landmarks_to_tensor(hand_lm)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = probs.max(dim=1)
            current_pred = class_names[pred_idx.item()]
            current_conf = conf.item()

        prediction_buffer.append(current_pred)

        # Majority vote over buffer
        if len(prediction_buffer) >= BUFFER_SIZE // 2:
            vote_counts = collections.Counter(prediction_buffer)
            stable_pred, count = vote_counts.most_common(1)[0]
            stability = count / len(prediction_buffer)

            if stability > 0.6 and current_conf > CONFIDENCE_THRESHOLD:
                current_pred = stable_pred

                # Add to sentence if held long enough and different from last
                if stable_pred == last_added:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_added = stable_pred

                if stable_count == HOLD_FRAMES:
                    sentence += stable_pred + " "
                    print("  Added: {}  |  Sentence: {}".format(stable_pred, sentence.strip()))
            else:
                current_pred = stable_pred
    else:
        prediction_buffer.clear()
        stable_count = 0

    # Draw UI
    draw_ui(frame, current_pred, current_conf, sentence.strip(), fps, hand_detected)

    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        sentence = ""
        last_added = ""
        stable_count = 0
        print("  Sentence cleared.")
    elif key == ord(" "):
        sentence += " "

cap.release()
cv2.destroyAllWindows()
print("")
print("Final sentence: {}".format(sentence.strip()))
print("Done!")