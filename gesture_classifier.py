"""Lightweight gesture classifier using hand landmarks.

Trains a small RandomForest/MLP classifier on extracted landmark features
to recognize 10 custom gestures at real-time speed.
"""

import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import compute_distances, get_finger_states


GESTURE_LABELS = {
    0: "fist",
    1: "open_palm",
    2: "pinch",
    3: "point_up",
    4: "peace",
    5: "thumbs_up",
    6: "thumbs_down",
    7: "swipe_left",
    8: "swipe_right",
    9: "three_fingers",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gestures", "gesture_model.pkl")


def extract_features(landmarks_array):
    """Extract feature vector from hand landmarks for classification.

    Combines distance-based features with finger state features.

    Args:
        landmarks_array: np.ndarray of shape (21, 3).

    Returns:
        np.ndarray feature vector.
    """
    distances = compute_distances(landmarks_array)
    finger_states = get_finger_states(landmarks_array)
    state_features = np.array(
        [float(finger_states[f]) for f in ["thumb", "index", "middle", "ring", "pinky"]]
    )
    return np.concatenate([distances, state_features])


def generate_synthetic_data(n_samples_per_class=200):
    """Generate synthetic training data based on known gesture patterns.

    Creates approximate landmark configurations for each gesture class
    with added noise for robustness.

    Returns:
        X: np.ndarray of shape (n_total, n_features).
        y: np.ndarray of shape (n_total,).
    """
    np.random.seed(42)
    X_all, y_all = [], []

    # Base hand configuration (21 landmarks, approximate positions)
    base_hand = np.array([
        [0.0, 0.0, 0.0],      # 0: WRIST
        [0.1, -0.1, 0.0],     # 1: THUMB_CMC
        [0.15, -0.2, 0.0],    # 2: THUMB_MCP
        [0.18, -0.3, 0.0],    # 3: THUMB_IP
        [0.2, -0.4, 0.0],     # 4: THUMB_TIP
        [0.05, -0.35, 0.0],   # 5: INDEX_MCP
        [0.05, -0.5, 0.0],    # 6: INDEX_PIP
        [0.05, -0.6, 0.0],    # 7: INDEX_DIP
        [0.05, -0.7, 0.0],    # 8: INDEX_TIP
        [0.0, -0.35, 0.0],    # 9: MIDDLE_MCP
        [0.0, -0.5, 0.0],     # 10: MIDDLE_PIP
        [0.0, -0.6, 0.0],     # 11: MIDDLE_DIP
        [0.0, -0.7, 0.0],     # 12: MIDDLE_TIP
        [-0.05, -0.33, 0.0],  # 13: RING_MCP
        [-0.05, -0.45, 0.0],  # 14: RING_PIP
        [-0.05, -0.55, 0.0],  # 15: RING_DIP
        [-0.05, -0.65, 0.0],  # 16: RING_TIP
        [-0.1, -0.3, 0.0],    # 17: PINKY_MCP
        [-0.1, -0.4, 0.0],    # 18: PINKY_PIP
        [-0.1, -0.48, 0.0],   # 19: PINKY_DIP
        [-0.1, -0.55, 0.0],   # 20: PINKY_TIP
    ], dtype=np.float32)

    # Gesture-specific modifications
    gesture_configs = {
        0: {"name": "fist", "curl": ["thumb", "index", "middle", "ring", "pinky"]},
        1: {"name": "open_palm", "curl": []},
        2: {"name": "pinch", "curl": ["middle", "ring", "pinky"], "pinch_thumb_index": True},
        3: {"name": "point_up", "curl": ["thumb", "middle", "ring", "pinky"]},
        4: {"name": "peace", "curl": ["thumb", "ring", "pinky"]},
        5: {"name": "thumbs_up", "curl": ["index", "middle", "ring", "pinky"], "thumb_up": True},
        6: {"name": "thumbs_down", "curl": ["index", "middle", "ring", "pinky"], "thumb_down": True},
        7: {"name": "swipe_left", "curl": [], "shift_x": -0.15},
        8: {"name": "swipe_right", "curl": [], "shift_x": 0.15},
        9: {"name": "three_fingers", "curl": ["thumb", "pinky"]},
    }

    finger_tip_indices = {
        "thumb": [3, 4],
        "index": [7, 8],
        "middle": [11, 12],
        "ring": [15, 16],
        "pinky": [19, 20],
    }

    for gesture_id, config in gesture_configs.items():
        for _ in range(n_samples_per_class):
            hand = base_hand.copy()

            # Apply curling (move tips closer to MCP)
            for finger in config.get("curl", []):
                for idx in finger_tip_indices[finger]:
                    hand[idx][1] += np.random.uniform(0.2, 0.35)

            # Special modifications
            if config.get("pinch_thumb_index"):
                hand[4] = hand[8] + np.random.normal(0, 0.02, 3)

            if config.get("thumb_up"):
                hand[4][1] = -0.5 + np.random.normal(0, 0.02)

            if config.get("thumb_down"):
                hand[4][1] = 0.1 + np.random.normal(0, 0.02)

            if "shift_x" in config:
                hand[:, 0] += config["shift_x"]

            # Add noise for robustness
            hand += np.random.normal(0, 0.015, hand.shape)

            features = extract_features(hand)
            X_all.append(features)
            y_all.append(gesture_id)

    return np.array(X_all), np.array(y_all)


def train_classifier(model_type="random_forest"):
    """Train a gesture classifier on synthetic data.

    Args:
        model_type: 'random_forest' or 'mlp'.

    Returns:
        Trained classifier.
    """
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples_per_class=300)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")

    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
        )

    print(f"Training {model_type} classifier...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(GESTURE_LABELS.values())))

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")

    return clf


def load_classifier():
    """Load a trained gesture classifier."""
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Training new classifier...")
        return train_classifier()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_gesture(clf, landmarks_array):
    """Predict gesture from hand landmarks.

    Args:
        clf: Trained classifier.
        landmarks_array: np.ndarray of shape (21, 3).

    Returns:
        (gesture_name, confidence) tuple.
    """
    features = extract_features(landmarks_array).reshape(1, -1)
    proba = clf.predict_proba(features)[0]
    pred_id = np.argmax(proba)
    confidence = proba[pred_id]
    return GESTURE_LABELS[pred_id], confidence


if __name__ == "__main__":
    train_classifier(model_type="random_forest")
