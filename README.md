# Real-Time Sign Language Translator

Real-time American Sign Language (ASL) recognition from a webcam, powered by
MediaPipe landmarks and a PyTorch Transformer classifier trained on the
[Kaggle "Google ASL Signs" dataset](https://www.kaggle.com/competitions/asl-signs)
(250 signs).

The model was **trained remotely** on a GPU machine; only the final checkpoint
is shipped in this repo so you can run real-time inference locally with a
standard webcam.

---

## Features

- 250-class ASL isolated sign recognition
- Real-time webcam inference (CPU-friendly)
- MediaPipe-based landmark extraction (hands + face + pose)
- Transformer encoder classifier (mean+max pooled)
- On-screen top-5 predictions, sentence builder, FPS counter

---

## Project layout

```
sign_lang_trans/
├── src/
│   ├── live_demo_transformer.py   # Real-time demo (Transformer, 250 signs) — main entry point
│   ├── live_demo.py               # Older real-time demo (legacy)
│   ├── train_transformer.py       # Training script for the Transformer model
│   ├── train_lstm.py              # Older LSTM training (legacy baseline)
│   ├── train.py                   # First training script (legacy)
│   ├── preprocess_kaggle.py       # Builds data/processed/*.npy from the Kaggle parquet files
│   ├── collect_data.py            # Capture your own webcam samples
│   └── test_mediapipe.py          # Quick sanity check for MediaPipe install
│
├── models/
│   ├── sign_transformer_best.pth  # Trained Transformer weights (tracked in git)
│   └── label_map_transformer.json # Class names + model hyper-params used at inference
│
├── data/                          # (gitignored — regenerate locally, see below)
│   ├── train.csv                          # Kaggle metadata
│   ├── sign_to_prediction_index_map.json  # Kaggle class map
│   ├── train_landmark_files/              # Raw parquet files from Kaggle
│   └── processed/                         # X_train.npy, y_train.npy, ... produced by preprocess
│
├── logs/                          # (gitignored) training / preprocess logs
│
├── hand_landmarker.task           # MediaPipe hand model
├── face_landmarker.task           # MediaPipe face model
├── pose_landmarker.task           # MediaPipe pose model
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model at a glance

| Item                  | Value                                    |
|-----------------------|------------------------------------------|
| Architecture          | Transformer encoder (mean + max pooled)  |
| Number of classes     | 250                                      |
| Input features / frame| 390 (hands 126 + face 228 + pose 36)     |
| Sequence length       | 64 frames                                |
| `d_model`             | 256                                      |
| Attention heads       | 8                                        |
| Encoder layers        | 4                                        |
| Feed-forward dim      | 512                                      |
| Dropout               | 0.3                                      |
| Train / val / test    | 65 220 / 11 510 / 17 747 sequences       |

Hyper-parameters and class list are persisted in
`models/label_map_transformer.json` and are loaded automatically by the demo,
so you never have to hard-code them at inference time.

---

## Installation

Requires **Python 3.10+** and a webcam.

```bash
# 1. Clone
git clone https://github.com/aymenwnia/ASL-translator sign_lang_trans
cd sign_lang_trans

# 2. Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install mediapipe opencv-python
```

> `requirements.txt` lists the core ML stack (`torch`, `numpy`, `pandas`,
> `scikit-learn`, `matplotlib`, `seaborn`). For the real-time demo you also need
> `mediapipe` and `opencv-python`.

### MediaPipe task files

The three `*.task` files at the project root are required for landmark
extraction. If they are missing, download
them from Google:

- Hands: <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>
- Face:  <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
- Pose:  <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task>

Place them in the repository root.

---

## Running the real-time demo

From the project root:

```bash
python src/live_demo_transformer.py
```

### Controls

| Key      | Action                                  |
|----------|-----------------------------------------|
| `q`      | Quit                                    |
| `c`      | Clear the accumulated sentence / buffer |
| `space`  | Insert a space in the sentence          |

### How it works at runtime

1. Each webcam frame is passed through the MediaPipe hand / face / pose
   detectors, producing a 390-dim feature vector.
2. Feature vectors are accumulated in a sliding window of `BUFFER_SIZE = 64`
   frames.
3. Once the buffer has enough frames, the sequence is fed to the Transformer,
   which returns class probabilities.
4. The top-1 prediction is only appended to the sentence after it stays stable
   for `HOLD_FRAMES = 25` frames and its probability exceeds
   `CONFIDENCE_THRESHOLD = 0.3`. This avoids flicker and "stuttered" words.

---

## Training (done remotely)

Training was performed on a separate GPU machine; the repo only ships the final
checkpoint. To reproduce:

1. **Get the dataset** — download the Kaggle "Google ASL Signs" competition
   data and unpack into `data/`:
   ```
   data/train.csv
   data/sign_to_prediction_index_map.json
   data/train_landmark_files/<participant_id>/<sequence_id>.parquet
   ```

2. **Preprocess** — converts raw parquet into padded numpy arrays
   (`data/processed/X_{train,val,test}.npy`, `y_*.npy`, `config.json`):
   ```bash
   python src/preprocess_kaggle.py
   ```

3. **Train** (run on a GPU box):
   ```bash
   python src/train_transformer.py
   # logs streamed to logs/train_transformer.log
   ```

4. **Copy the artifacts back** to this machine:
   ```
   models/sign_transformer_best.pth
   models/label_map_transformer.json
   ```

The current checkpoint in `models/` was produced this way.

---

## Dataset

- Kaggle "Google Isolated Sign Language Recognition" challenge
- 250 signs, ~94k training sequences across participants
- Each sequence is a sequence of frames with landmark coordinates
  (hands, face, pose)
- Not redistributed here; download directly from Kaggle

---

## Acknowledgements

- [Google MediaPipe](https://developers.google.com/mediapipe) — landmark models
- [Kaggle Google ASL Signs dataset](https://www.kaggle.com/competitions/asl-signs)
- PyTorch
