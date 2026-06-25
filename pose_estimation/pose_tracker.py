"""
pose_tracker.py — 3D Human Pose Estimation Module
Uses MediaPipe Tasks PoseLandmarker API (mediapipe 0.10+)
33 landmarks with x, y, z (depth) coordinates
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions

# ─── MediaPipe Pose landmark indices ─────────────────────────────────────────
NOSE            = 0
LEFT_EYE        = 2;   RIGHT_EYE       = 5
LEFT_EAR        = 7;   RIGHT_EAR       = 8
LEFT_SHOULDER   = 11;  RIGHT_SHOULDER  = 12
LEFT_ELBOW      = 13;  RIGHT_ELBOW     = 14
LEFT_WRIST      = 15;  RIGHT_WRIST     = 16
LEFT_HIP        = 23;  RIGHT_HIP       = 24
LEFT_KNEE       = 25;  RIGHT_KNEE      = 26
LEFT_ANKLE      = 27;  RIGHT_ANKLE     = 28
LEFT_HEEL       = 29;  RIGHT_HEEL      = 30
LEFT_FOOT       = 31;  RIGHT_FOOT      = 32

# ─── Skeleton connections (pairs of landmark indices) ─────────────────────────
POSE_CONNECTIONS = [
    # Face
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    # Torso
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # Left arm
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    # Right arm
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    # Left leg
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT),
    # Right leg
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT),
]

# Body part groups for color coding
TORSO_IDS  = {LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP}
LARM_IDS   = {LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST}
RARM_IDS   = {RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST}
LLEG_IDS   = {LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT}
RLEG_IDS   = {RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT}
FACE_IDS   = {NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR}

# Joint angle triplets: (A, VERTEX, B) — angle measured at VERTEX
JOINT_ANGLES = {
    "L.Elbow":   (LEFT_SHOULDER,  LEFT_ELBOW,   LEFT_WRIST),
    "R.Elbow":   (RIGHT_SHOULDER, RIGHT_ELBOW,  RIGHT_WRIST),
    "L.Shoulder":(LEFT_HIP,       LEFT_SHOULDER, LEFT_ELBOW),
    "R.Shoulder":(RIGHT_HIP,      RIGHT_SHOULDER,RIGHT_ELBOW),
    "L.Knee":    (LEFT_HIP,       LEFT_KNEE,    LEFT_ANKLE),
    "R.Knee":    (RIGHT_HIP,      RIGHT_KNEE,   RIGHT_ANKLE),
    "L.Hip":     (LEFT_SHOULDER,  LEFT_HIP,     LEFT_KNEE),
    "R.Hip":     (RIGHT_SHOULDER, RIGHT_HIP,    RIGHT_KNEE),
}


class PoseTracker:
    """
    Wraps MediaPipe PoseLandmarker for real-time 3D pose estimation.

    Parameters
    ----------
    model_path      : path to pose_landmarker.task
    detection_conf  : min detection confidence
    tracking_conf   : min tracking confidence
    """

    def __init__(self, model_path: str = "pose_landmarker.task",
                 detection_conf: float = 0.6,
                 tracking_conf: float  = 0.5):

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_conf,
            min_pose_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            output_segmentation_masks=False,
        )
        self.landmarker  = PoseLandmarker.create_from_options(options)
        self._result     = None
        self._timestamp  = 0

    # ── Core ─────────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> bool:
        """Run detection on BGR frame. Returns True if pose found."""
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 33
        self._result = self.landmarker.detect_for_video(mp_image, self._timestamp)
        return len(self._result.pose_landmarks) > 0

    def detected(self) -> bool:
        return self._result is not None and len(self._result.pose_landmarks) > 0

    # ── Landmark access ───────────────────────────────────────────────────────

    def get_landmarks(self, pose_idx: int = 0):
        """Return list of (x, y, z, visibility) normalized coords."""
        if not self.detected():
            return []
        return [(lm.x, lm.y, lm.z, lm.visibility)
                for lm in self._result.pose_landmarks[pose_idx]]

    def get_world_landmarks(self, pose_idx: int = 0):
        """Return list of (x, y, z) in metric-scale world coords."""
        if not self.detected() or not self._result.pose_world_landmarks:
            return []
        return [(lm.x, lm.y, lm.z)
                for lm in self._result.pose_world_landmarks[pose_idx]]

    def get_pixel_landmarks(self, pose_idx: int, w: int, h: int):
        """Return list of (px, py, z, vis) — pixel x/y + raw depth + visibility."""
        lms = self.get_landmarks(pose_idx)
        return [(int(x * w), int(y * h), z, vis) for x, y, z, vis in lms]

    # ── Analytics ─────────────────────────────────────────────────────────────

    def joint_angle(self, a: int, vertex: int, b: int,
                    pose_idx: int = 0) -> float:
        """Angle in degrees at `vertex` formed by landmarks a-vertex-b."""
        lms = self.get_landmarks(pose_idx)
        if not lms or len(lms) <= max(a, vertex, b):
            return 0.0
        ax, ay = lms[a][0] - lms[vertex][0], lms[a][1] - lms[vertex][1]
        bx, by = lms[b][0] - lms[vertex][0], lms[b][1] - lms[vertex][1]
        dot    = ax * bx + ay * by
        mag    = math.hypot(ax, ay) * math.hypot(bx, by)
        if mag < 1e-6:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

    def all_joint_angles(self, pose_idx: int = 0) -> dict:
        """Return dict of {joint_name: angle_degrees} for all defined joints."""
        return {
            name: self.joint_angle(a, v, b, pose_idx)
            for name, (a, v, b) in JOINT_ANGLES.items()
        }

    def bounding_box(self, pose_idx: int, w: int, h: int,
                     pad: int = 20) -> tuple:
        pts = self.get_pixel_landmarks(pose_idx, w, h)
        if not pts:
            return (0, 0, 0, 0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (max(0, min(xs) - pad), max(0, min(ys) - pad),
                min(w, max(xs) + pad), min(h, max(ys) + pad))

    def posture_label(self, pose_idx: int = 0) -> str:
        """Rough posture classification based on hip/shoulder/knee heights."""
        lms = self.get_landmarks(pose_idx)
        if not lms:
            return "Unknown"
        hip_y    = (lms[LEFT_HIP][1]      + lms[RIGHT_HIP][1])      / 2
        knee_y   = (lms[LEFT_KNEE][1]     + lms[RIGHT_KNEE][1])     / 2
        shoulder_y = (lms[LEFT_SHOULDER][1] + lms[RIGHT_SHOULDER][1]) / 2

        if abs(hip_y - knee_y) < 0.08:
            return "Sitting"
        if shoulder_y > 0.6:
            return "Lying"
        return "Standing"
