"""
hand_tracker.py — Reusable Hand Tracking Module (MediaPipe Tasks API)
Compatible with mediapipe 0.10.x on Python 3.14
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# ─── Landmark indices (MediaPipe standard) ────────────────────────────────────
WRIST        = 0
THUMB_CMC    = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP    = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP   = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP     = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP    = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

FINGER_TIPS  = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]


class HandTracker:
    """
    Hand tracker using MediaPipe Tasks API (mediapipe 0.10+).
    Requires hand_landmarker.task model file.
    """

    def __init__(self, max_hands: int = 2,
                 detection_conf: float = 0.75,
                 tracking_conf: float = 0.70,
                 model_path: str = "hand_landmarker.task"):

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker  = vision.HandLandmarker.create_from_options(options)
        self._result     = None
        self._timestamp  = 0

    def process(self, frame: np.ndarray) -> bool:
        """Run detection on BGR frame. Returns True if hands found."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 33          # ~30fps increment in ms
        self._result = self.landmarker.detect_for_video(mp_image, self._timestamp)
        return len(self._result.hand_landmarks) > 0

    def hand_count(self) -> int:
        if not self._result:
            return 0
        return len(self._result.hand_landmarks)

    def get_landmarks(self, hand_idx: int = 0):
        """Return list of (x, y, z) normalized coords."""
        if not self._result or hand_idx >= len(self._result.hand_landmarks):
            return []
        return [(lm.x, lm.y, lm.z) for lm in self._result.hand_landmarks[hand_idx]]

    def get_pixel_landmarks(self, hand_idx: int, w: int, h: int):
        """Return list of (px, py) pixel coords."""
        lms = self.get_landmarks(hand_idx)
        return [(int(x * w), int(y * h)) for x, y, z in lms]

    def handedness(self, hand_idx: int = 0) -> str:
        if not self._result or hand_idx >= len(self._result.handedness):
            return "?"
        label = self._result.handedness[hand_idx][0].display_name
        # Mirror for camera
        return "Left" if label == "Right" else "Right"

    def fingers_up(self, hand_idx: int = 0) -> list:
        lms = self.get_landmarks(hand_idx)
        if not lms:
            return [False] * 5

        up = []
        side = self.handedness(hand_idx)
        if side == "Right":
            up.append(lms[THUMB_TIP][0] > lms[THUMB_IP][0])
        else:
            up.append(lms[THUMB_TIP][0] < lms[THUMB_IP][0])

        for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP, RING_PIP),   (PINKY_TIP, PINKY_PIP)]:
            up.append(lms[tip][1] < lms[pip][1])
        return up

    def count_fingers(self, hand_idx: int = 0) -> int:
        return sum(self.fingers_up(hand_idx))

    def gesture_name(self, hand_idx: int = 0) -> str:
        up = self.fingers_up(hand_idx)
        if not any(up):        return "Fist"
        if all(up):            return "Open"
        if up == [False, True,  True,  False, False]: return "Peace"
        if up == [False, True,  False, False, False]: return "Point"
        if up == [True,  False, False, False, True]:  return "Hang Loose"
        if up == [True,  True,  False, False, False]: return "L-Shape"
        return f"{sum(up)} fingers"

    def distance_between(self, hand_idx: int, a: int, b: int,
                         w: int, h: int) -> float:
        pts = self.get_pixel_landmarks(hand_idx, w, h)
        if not pts:
            return 0.0
        return math.hypot(pts[b][0] - pts[a][0], pts[b][1] - pts[a][1])

    def bounding_box(self, hand_idx: int, w: int, h: int,
                     pad: int = 20) -> tuple:
        pts = self.get_pixel_landmarks(hand_idx, w, h)
        if not pts:
            return (0, 0, 0, 0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (max(0, min(xs) - pad), max(0, min(ys) - pad),
                min(w, max(xs) + pad), min(h, max(ys) + pad))
