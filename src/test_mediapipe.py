# test_mediapipe.py
# Verify MediaPipe hand landmarks work with your webcam.
#
# BEFORE RUNNING - download the model file in PowerShell:
#   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
#
# Then run:  python test_mediapipe.py
# Press 'q' to quit.

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Hand bone connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def draw_hand(image, hand_landmarks, w, h):
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, coords[start], coords[end], (0, 200, 0), 2)


# Setup detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# Webcam loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

print("Webcam opened. Show your hand to the camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

    for hand_lm in result.hand_landmarks:
        draw_hand(frame, hand_lm, w, h)

    cv2.putText(frame, "Hands: {}".format(n_hands), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if n_hands > 0:
        cv2.putText(frame, "Landmarks: {}".format(len(result.hand_landmarks[0])),
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("MediaPipe Hand Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done! MediaPipe hand detection is working.")