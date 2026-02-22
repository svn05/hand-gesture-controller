"""Utility functions for hand gesture controller."""

import numpy as np
import json
import os


def load_gesture_map(path=None):
    """Load gesture mapping configuration from JSON file."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "gestures", "gesture_map.json")
    with open(path, "r") as f:
        return json.load(f)


def normalize_landmarks(landmarks, frame_width, frame_height):
    """Convert MediaPipe landmarks to normalized numpy array.

    Args:
        landmarks: MediaPipe hand landmarks (21 points).
        frame_width: Width of the video frame.
        frame_height: Height of the video frame.

    Returns:
        np.ndarray of shape (21, 3) with x, y, z coordinates.
    """
    coords = np.array(
        [[lm.x * frame_width, lm.y * frame_height, lm.z] for lm in landmarks.landmark]
    )
    return coords


def compute_distances(landmarks_array):
    """Compute pairwise distances between key landmarks for gesture features.

    Uses fingertip-to-palm and fingertip-to-fingertip distances as features
    for the gesture classifier.

    Args:
        landmarks_array: np.ndarray of shape (21, 3).

    Returns:
        np.ndarray feature vector of distances.
    """
    # Key landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5

    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    palm_center = landmarks_array[WRIST]

    distances = []

    # Fingertip to wrist distances
    for tip in tips:
        dist = np.linalg.norm(landmarks_array[tip] - palm_center)
        distances.append(dist)

    # Fingertip to fingertip distances (thumb-index, index-middle, etc.)
    for i in range(len(tips) - 1):
        dist = np.linalg.norm(landmarks_array[tips[i]] - landmarks_array[tips[i + 1]])
        distances.append(dist)

    # Thumb-index pinch distance (important for click gesture)
    distances.append(np.linalg.norm(landmarks_array[THUMB_TIP] - landmarks_array[INDEX_TIP]))

    # Finger curl ratios (tip to MCP vs wrist to MCP)
    wrist_to_mcp = np.linalg.norm(landmarks_array[WRIST] - landmarks_array[INDEX_MCP])
    for tip in tips:
        tip_to_mcp = np.linalg.norm(landmarks_array[tip] - landmarks_array[INDEX_MCP])
        ratio = tip_to_mcp / (wrist_to_mcp + 1e-6)
        distances.append(ratio)

    return np.array(distances, dtype=np.float32)


def get_finger_states(landmarks_array):
    """Determine which fingers are extended (open) or closed.

    Args:
        landmarks_array: np.ndarray of shape (21, 3).

    Returns:
        dict with finger names as keys and bool (True=extended) as values.
    """
    # Landmark indices for each finger: [MCP, PIP, DIP, TIP]
    fingers = {
        "thumb": [2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    states = {}

    # For thumb, check if tip is to the left/right of IP joint (depends on hand)
    thumb_tip = landmarks_array[4]
    thumb_ip = landmarks_array[3]
    thumb_mcp = landmarks_array[2]
    states["thumb"] = np.linalg.norm(thumb_tip - landmarks_array[0]) > np.linalg.norm(
        thumb_ip - landmarks_array[0]
    )

    # For other fingers, check if tip is above PIP joint (y-axis, lower y = higher)
    for finger in ["index", "middle", "ring", "pinky"]:
        tip_idx = fingers[finger][-1]
        pip_idx = fingers[finger][1]
        states[finger] = landmarks_array[tip_idx][1] < landmarks_array[pip_idx][1]

    return states


def smooth_coordinates(current, previous, alpha=0.3):
    """Apply exponential moving average smoothing to reduce jitter.

    Args:
        current: Current coordinate values.
        previous: Previous smoothed values.
        alpha: Smoothing factor (0 = full smoothing, 1 = no smoothing).

    Returns:
        Smoothed coordinate values.
    """
    if previous is None:
        return current
    return alpha * np.array(current) + (1 - alpha) * np.array(previous)
