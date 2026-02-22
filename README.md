# Real-Time Hand Gesture Controller

A real-time hand tracking system using **MediaPipe** and **OpenCV** that maps 21-point hand landmarks to 8+ custom gestures for desktop control (volume, scroll, cursor movement, and more). Includes a lightweight gesture classifier trained on landmark features, running at **30+ FPS**.

## Features

- **Real-time hand detection** using MediaPipe's 21-point hand landmark model
- **10 custom gestures**: fist, open palm, pinch (click), point (cursor), peace (scroll), thumbs up/down (volume), swipe left/right, three fingers (screenshot)
- **Desktop control** via PyAutoGUI — cursor movement, click, scroll, volume, navigation
- **Lightweight classifier** (RandomForest/MLP) trained on landmark-derived features
- **30+ FPS** on CPU with exponential smoothing for jitter-free control
- **Data collection tool** for recording your own gesture samples via webcam

## Supported Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| Fist | Pause | All fingers closed |
| Open Palm | Stop | All fingers extended |
| Pinch | Click | Thumb + index touching |
| Point Up | Cursor Move | Index finger extended |
| Peace | Scroll | Index + middle extended |
| Thumbs Up | Volume Up | Thumb up, fingers closed |
| Thumbs Down | Volume Down | Thumb down, fingers closed |
| Swipe Left | Previous | Open palm moving left |
| Swipe Right | Next | Open palm moving right |
| Three Fingers | Screenshot | Index + middle + ring up |

## Setup

```bash
git clone https://github.com/svn05/hand-gesture-controller.git
cd hand-gesture-controller
pip install -r requirements.txt
```

## Usage

### Run the gesture controller
```bash
python gesture_controller.py
```

### Preview mode (no desktop control)
```bash
python gesture_controller.py --no-control
```

### Debug mode (show landmark indices)
```bash
python gesture_controller.py --debug
```

### Train the gesture classifier
```bash
python gesture_classifier.py
```

### Collect custom gesture data
```bash
python collect_data.py --gesture fist --samples 100
python collect_data.py --gesture pinch --samples 100
```

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Toggle desktop control on/off |
| `SPACE` | Start/stop recording (data collection) |

## Architecture

```
Webcam → MediaPipe Hand Detection → 21 Landmarks → Feature Extraction
    → Gesture Classifier (RandomForest) → Action Mapping → PyAutoGUI
```

**Feature extraction** combines:
- Fingertip-to-wrist distances (5 features)
- Fingertip-to-fingertip distances (4 features)
- Pinch distance (1 feature)
- Finger curl ratios (5 features)
- Binary finger states (5 features)

**Total: 20 features** fed into a RandomForest classifier with 100 trees.

## Project Structure

```
hand-gesture-controller/
├── gesture_controller.py    # Main real-time controller
├── gesture_classifier.py    # Gesture classifier (train + predict)
├── collect_data.py          # Webcam data collection tool
├── utils.py                 # Landmark processing utilities
├── gestures/
│   ├── gesture_map.json     # Gesture-to-action mapping config
│   └── gesture_model.pkl    # Trained classifier (generated)
├── requirements.txt
└── README.md
```

## Tech Stack

- **MediaPipe** — Hand landmark detection (21 keypoints)
- **OpenCV** — Video capture and visualization
- **PyAutoGUI** — Desktop automation (cursor, keyboard, scroll)
- **scikit-learn** — RandomForest/MLP gesture classifier
- **NumPy** — Feature computation and coordinate smoothing
