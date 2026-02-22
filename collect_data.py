"""Collect hand landmark data from webcam for gesture training.

Captures hand landmarks via MediaPipe and saves them with gesture labels
for training the gesture classifier on real data.

Usage:
    python collect_data.py --gesture fist --samples 100
"""

import argparse
import csv
import os
import cv2
import mediapipe as mp
import numpy as np
from utils import normalize_landmarks, compute_distances, get_finger_states
from gesture_classifier import extract_features, GESTURE_LABELS


DATA_DIR = os.path.join(os.path.dirname(__file__), "gestures", "collected_data")


def collect_gesture_data(gesture_name, n_samples=100, output_dir=DATA_DIR):
    """Capture hand landmarks from webcam and save as training data.

    Args:
        gesture_name: Name of the gesture being recorded.
        n_samples: Number of samples to collect.
        output_dir: Directory to save CSV data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Validate gesture name
    valid_gestures = set(GESTURE_LABELS.values())
    if gesture_name not in valid_gestures:
        print(f"Invalid gesture: {gesture_name}")
        print(f"Valid gestures: {', '.join(sorted(valid_gestures))}")
        return

    gesture_id = [k for k, v in GESTURE_LABELS.items() if v == gesture_name][0]

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    csv_path = os.path.join(output_dir, f"{gesture_name}.csv")
    samples_collected = 0
    recording = False

    print(f"\nCollecting data for gesture: '{gesture_name}' (ID: {gesture_id})")
    print(f"Target samples: {n_samples}")
    print("Controls:")
    print("  SPACE - Start/stop recording")
    print("  Q     - Quit")
    print("\nShow the gesture and press SPACE to begin recording.\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened() and samples_collected < n_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    landmarks_array = normalize_landmarks(hand_landmarks, w, h)
                    features = extract_features(landmarks_array)
                    samples_collected += 1

                    # Append to CSV
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([gesture_id] + features.tolist())

            # Display status
            status = "RECORDING" if recording else "PAUSED"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{n_samples}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Gesture Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCollected {samples_collected} samples for '{gesture_name}'")
    print(f"Data saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect gesture training data from webcam")
    parser.add_argument(
        "--gesture", type=str, required=True,
        help=f"Gesture name to collect. Options: {', '.join(GESTURE_LABELS.values())}"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of samples to collect (default: 100)"
    )
    args = parser.parse_args()
    collect_gesture_data(args.gesture, args.samples)


if __name__ == "__main__":
    main()
