"""Real-time hand gesture controller for desktop.

Uses MediaPipe for hand tracking, a trained classifier for gesture recognition,
and PyAutoGUI for desktop control (cursor, volume, scroll, etc.).

Usage:
    python gesture_controller.py
    python gesture_controller.py --no-control   # Preview only, no desktop control
    python gesture_controller.py --debug         # Show landmark coordinates
"""

import argparse
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from utils import normalize_landmarks, smooth_coordinates, load_gesture_map
from gesture_classifier import load_classifier, predict_gesture, extract_features

# Disable PyAutoGUI fail-safe for smoother control (move mouse to corner to stop)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


class GestureController:
    """Maps recognized hand gestures to desktop actions."""

    def __init__(self, enable_control=True, confidence_threshold=0.7, smoothing=0.3):
        self.enable_control = enable_control
        self.confidence_threshold = confidence_threshold
        self.smoothing = smoothing

        # Load gesture classifier
        self.classifier = load_classifier()
        self.gesture_map = load_gesture_map()

        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # State tracking
        self.prev_cursor = None
        self.prev_gesture = None
        self.gesture_start_time = 0
        self.cooldown_ms = self.gesture_map["settings"]["cooldown_ms"]
        self.screen_w, self.screen_h = pyautogui.size()

        # FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def _map_to_screen(self, x_norm, y_norm, frame_w, frame_h):
        """Map normalized hand position to screen coordinates."""
        # Use central region of camera for better control range
        margin = 0.15
        x_clamped = np.clip((x_norm / frame_w - margin) / (1 - 2 * margin), 0, 1)
        y_clamped = np.clip((y_norm / frame_h - margin) / (1 - 2 * margin), 0, 1)
        screen_x = int(x_clamped * self.screen_w)
        screen_y = int(y_clamped * self.screen_h)
        return screen_x, screen_y

    def _execute_action(self, gesture_name, landmarks_array, frame_w, frame_h):
        """Execute desktop action based on recognized gesture."""
        current_time = time.time() * 1000

        if gesture_name == "cursor_move":
            # Move cursor to index fingertip position
            index_tip = landmarks_array[8]
            screen_x, screen_y = self._map_to_screen(
                index_tip[0], index_tip[1], frame_w, frame_h
            )
            smoothed = smooth_coordinates(
                np.array([screen_x, screen_y]), self.prev_cursor, self.smoothing
            )
            self.prev_cursor = smoothed
            if self.enable_control:
                pyautogui.moveTo(int(smoothed[0]), int(smoothed[1]), _pause=False)

        elif gesture_name == "click":
            if current_time - self.gesture_start_time > self.cooldown_ms:
                if self.enable_control:
                    pyautogui.click()
                self.gesture_start_time = current_time

        elif gesture_name == "scroll":
            # Use vertical movement of middle finger for scroll direction
            middle_tip_y = landmarks_array[12][1]
            middle_mcp_y = landmarks_array[9][1]
            scroll_amount = int((middle_mcp_y - middle_tip_y) / frame_h * 10)
            if abs(scroll_amount) > 1 and self.enable_control:
                pyautogui.scroll(scroll_amount)

        elif gesture_name == "volume_up":
            if current_time - self.gesture_start_time > self.cooldown_ms:
                if self.enable_control:
                    pyautogui.hotkey("volumeup" if hasattr(pyautogui, "volumeup") else "up")
                self.gesture_start_time = current_time

        elif gesture_name == "volume_down":
            if current_time - self.gesture_start_time > self.cooldown_ms:
                if self.enable_control:
                    pyautogui.hotkey("volumedown" if hasattr(pyautogui, "volumedown") else "down")
                self.gesture_start_time = current_time

        elif gesture_name == "pause":
            if current_time - self.gesture_start_time > self.cooldown_ms * 2:
                if self.enable_control:
                    pyautogui.hotkey("space")  # Pause/play media
                self.gesture_start_time = current_time

        elif gesture_name == "stop":
            if current_time - self.gesture_start_time > self.cooldown_ms * 2:
                if self.enable_control:
                    pyautogui.hotkey("mediaplaypause") if sys.platform == "win32" else pyautogui.hotkey("space")
                self.gesture_start_time = current_time

        elif gesture_name == "swipe_left":
            if current_time - self.gesture_start_time > self.cooldown_ms * 3:
                if self.enable_control:
                    if sys.platform == "darwin":
                        pyautogui.hotkey("command", "left")
                    else:
                        pyautogui.hotkey("alt", "left")
                self.gesture_start_time = current_time

        elif gesture_name == "swipe_right":
            if current_time - self.gesture_start_time > self.cooldown_ms * 3:
                if self.enable_control:
                    if sys.platform == "darwin":
                        pyautogui.hotkey("command", "right")
                    else:
                        pyautogui.hotkey("alt", "right")
                self.gesture_start_time = current_time

        elif gesture_name == "screenshot":
            if current_time - self.gesture_start_time > self.cooldown_ms * 5:
                if self.enable_control:
                    if sys.platform == "darwin":
                        pyautogui.hotkey("command", "shift", "3")
                    elif sys.platform == "win32":
                        pyautogui.hotkey("win", "shift", "s")
                    else:
                        pyautogui.hotkey("print")
                self.gesture_start_time = current_time

    def _update_fps(self):
        """Calculate current FPS."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

    def run(self, debug=False):
        """Main loop: capture webcam, detect gestures, execute actions.

        Args:
            debug: If True, display landmark coordinates on frame.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Set camera properties for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Hand Gesture Controller Started")
        print(f"Desktop control: {'ENABLED' if self.enable_control else 'DISABLED'}")
        print("Press Q to quit\n")

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        ) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb_frame)

                gesture_text = "No hand detected"
                confidence = 0.0

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )

                    # Normalize landmarks and predict gesture
                    landmarks_array = normalize_landmarks(hand_landmarks, w, h)
                    gesture_name, confidence = predict_gesture(self.classifier, landmarks_array)

                    if confidence >= self.confidence_threshold:
                        gesture_text = f"{gesture_name} ({confidence:.2f})"
                        action = self.gesture_map["gestures"].get(gesture_name, {}).get("action", gesture_name)
                        self._execute_action(action, landmarks_array, w, h)
                        self.prev_gesture = gesture_name
                    else:
                        gesture_text = f"{gesture_name} ({confidence:.2f}) [low conf]"

                    if debug:
                        for i, lm in enumerate(hand_landmarks.landmark):
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.putText(frame, str(i), (cx, cy - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                # Update and display FPS
                self._update_fps()

                # HUD overlay
                cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                control_status = "ON" if self.enable_control else "OFF"
                cv2.putText(frame, f"Control: {control_status}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                cv2.imshow("Hand Gesture Controller", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    self.enable_control = not self.enable_control
                    print(f"Control: {'ON' if self.enable_control else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
        print("Gesture controller stopped.")


def main():
    parser = argparse.ArgumentParser(description="Real-time hand gesture controller")
    parser.add_argument("--no-control", action="store_true",
                        help="Preview mode — gestures recognized but no desktop control")
    parser.add_argument("--debug", action="store_true",
                        help="Show landmark indices on frame")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Gesture confidence threshold (default: 0.7)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="Cursor smoothing factor 0-1 (default: 0.3)")
    args = parser.parse_args()

    controller = GestureController(
        enable_control=not args.no_control,
        confidence_threshold=args.threshold,
        smoothing=args.smoothing,
    )
    controller.run(debug=args.debug)


if __name__ == "__main__":
    main()
