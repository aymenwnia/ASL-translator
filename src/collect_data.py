# collect_data.py
# Record hand landmark data for sign language recognition.
#
# USAGE:
#   python src/collect_data.py --sign hello --samples 200
#   python src/collect_data.py --sign yes --samples 200
#   python src/collect_data.py --sign no --samples 200
#
# Press 'q' to stop early. Press 'p' to pause/resume.

import argparse
import csv
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
OUTPUT_DIR = "data/raw"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

LANDMARK_NAMES = [
    "WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]


def get_csv_header():
    header = ["label"]
    for name in LANDMARK_NAMES:
        header.extend([name + "_x", name + "_y", name + "_z"])
    return header


def landmarks_to_row(label, hand_landmarks):
    row = [label]
    for lm in hand_landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row


def draw_hand(image, hand_landmarks, w, h):
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, coords[start], coords[end], (0, 200, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Collect sign language landmark data")
    parser.add_argument("--sign", required=True, help="Label for this sign (e.g. hello, yes, no)")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to collect")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, args.sign + ".csv")
    file_exists = os.path.exists(csv_path)

    # Setup MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    collected = 0
    paused = False

    print("")
    print("=" * 50)
    print("  Recording sign: '{}'".format(args.sign))
    print("  Target samples: {}".format(args.samples))
    print("  Saving to: {}".format(csv_path))
    print("=" * 50)
    print("Show your hand to the camera and hold the sign.")
    print("Press 'p' to pause/resume, 'q' to quit early.")
    print("")

    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(get_csv_header())

    try:
        while collected < args.samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            has_hand = len(result.hand_landmarks) > 0 if result.hand_landmarks else False

            if has_hand:
                draw_hand(frame, result.hand_landmarks[0], img_w, img_h)

            # Status display
            status = "PAUSED" if paused else "RECORDING"
            color = (0, 0, 255) if paused else (0, 255, 0)
            cv2.putText(frame, "Sign: {}".format(args.sign), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "{} | {}/{}".format(status, collected, args.samples),
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Progress bar
            progress = int((collected / args.samples) * 300)
            cv2.rectangle(frame, (10, 80), (310, 100), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 80), (10 + progress, 100), color, -1)

            # Record
            if not paused and has_hand:
                row = landmarks_to_row(args.sign, result.hand_landmarks[0])
                writer.writerow(row)
                collected += 1
                if collected % 50 == 0:
                    print("  Collected {}/{} samples...".format(collected, args.samples))

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Stopped early at {} samples.".format(collected))
                break
            elif key == ord("p"):
                paused = not paused
                print("  {} recording.".format("Paused" if paused else "Resumed"))

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()

    print("")
    print("Done! Saved {} samples to {}".format(collected, csv_path))


if __name__ == "__main__":
    main()
