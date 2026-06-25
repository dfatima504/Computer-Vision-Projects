"""
main.py — 3D Human Pose Estimation for Digital Twins
=====================================================
Real-time pose estimation with 3D depth-colored skeleton overlay,
joint angle HUD, posture classification, and switchable input source.

Controls:
  Q / ESC  → Quit
  W        → Switch input source (webcam ↔ video file)
  F        → Toggle fullscreen
  S        → Save screenshot
  H        → Toggle HUD
  B        → Toggle dark background
  A        → Toggle joint angle labels

Run:
  python main.py
  python main.py --video path/to/video.mp4
  python main.py --camera 1
"""

import cv2
import numpy as np
import time
import argparse
import sys
from datetime import datetime
from pathlib import Path

from pose_tracker import PoseTracker
from pose_renderer import (
    draw_pose_skeleton, draw_joint_angles,
    draw_pose_hud, draw_global_hud, apply_dark_blend
)


# ─── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="3D Human Pose Estimation — Digital Twin")
    ap.add_argument("--camera",  type=int,   default=0,
                    help="Webcam index (default 0)")
    ap.add_argument("--video",   type=str,   default=None,
                    help="Path to video file (optional)")
    ap.add_argument("--width",   type=int,   default=1280)
    ap.add_argument("--height",  type=int,   default=720)
    ap.add_argument("--model",   type=str,   default="pose_landmarker.task",
                    help="Path to pose_landmarker.task model file")
    ap.add_argument("--det",     type=float, default=0.6)
    ap.add_argument("--track",   type=float, default=0.5)
    return ap.parse_args()


# ─── FPS counter ───────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ─── Source manager ────────────────────────────────────────────────────────────

class SourceManager:
    """Manages switching between webcam and video file."""

    def __init__(self, camera_idx: int, video_path: str | None,
                 width: int, height: int):
        self.camera_idx  = camera_idx
        self.video_path  = video_path
        self.width       = width
        self.height      = height
        self.using_video = video_path is not None
        self.cap         = None
        self._open()

    def _open(self):
        if self.cap:
            self.cap.release()
        src = self.video_path if self.using_video else self.camera_idx
        self.cap = cv2.VideoCapture(src)
        if not self.using_video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

    def switch(self):
        """Toggle between webcam and video."""
        if self.video_path is None:
            print("[WARN] No video file specified. Use --video path/to/file.mp4")
            return
        self.using_video = not self.using_video
        self._open()
        print(f"[INFO] Source → {'VIDEO' if self.using_video else 'WEBCAM'}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret and self.using_video:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    @property
    def source_label(self) -> str:
        return "VIDEO" if self.using_video else "WEBCAM"

    def release(self):
        if self.cap:
            self.cap.release()


# ─── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Check model file
    if not Path(args.model).exists():
        print(f"""
[ERROR] Model file not found: {args.model}

Download it with:
  Windows PowerShell:
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -OutFile "pose_landmarker.task"

  Linux/Mac:
    wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
""")
        sys.exit(1)

    tracker = PoseTracker(args.model, args.det, args.track)
    source  = SourceManager(args.camera, args.video, args.width, args.height)
    fps_ctr = FPSCounter()

    # Toggles
    show_hud    = True
    dark_bg     = True
    show_angles = True
    fullscr     = False
    mirror      = True

    WIN = "🦾 3D Pose Estimation — Digital Twin  |  Q=quit  W=switch source"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print("=" * 60)
    print("  3D HUMAN POSE ESTIMATION — DIGITAL TWIN")
    print("=" * 60)
    print("  Q/ESC  Quit        W  Switch webcam/video")
    print("  F      Fullscreen  S  Screenshot")
    print("  H      HUD on/off  B  Dark background")
    print("  A      Angle labels on/off  M  Mirror")
    print("=" * 60)

    while True:
        ret, frame = source.read()
        if not ret:
            print("[WARN] Dropped frame.")
            continue

        if mirror and not source.using_video:
            frame = cv2.flip(frame, 1)

        h, w  = frame.shape[:2]
        fps   = fps_ctr.tick()

        # ── Run pose estimation ───────────────────────────────────────────────
        found = tracker.process(frame)

        # ── Canvas ────────────────────────────────────────────────────────────
        canvas = apply_dark_blend(frame, 0.30) if dark_bg else frame.copy()

        # ── Draw skeleton & HUD ───────────────────────────────────────────────
        if found:
            pts     = tracker.get_pixel_landmarks(0, w, h)
            angles  = tracker.all_joint_angles(0)
            posture = tracker.posture_label(0)
            bbox    = tracker.bounding_box(0, w, h)

            draw_pose_skeleton(canvas, pts)

            if show_angles:
                draw_joint_angles(canvas, pts, angles)

            if show_hud:
                draw_pose_hud(canvas, h, w, angles, posture, bbox, fps)

        if show_hud:
            draw_global_hud(canvas, h, w, fps, found, source.source_label)

        # Help bar
        help_txt = "Q=quit  W=switch  F=full  S=shot  H=hud  B=bg  A=angles  M=mirror"
        cv2.putText(canvas, help_txt, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        cv2.imshow(WIN, canvas)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('w'):
            source.switch()
        elif key == ord('f'):
            fullscr = not fullscr
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscr else cv2.WINDOW_NORMAL)
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"pose_screenshot_{ts}.png"
            cv2.imwrite(path, canvas)
            print(f"[SAVED] {path}")
        elif key == ord('h'):
            show_hud = not show_hud
        elif key == ord('b'):
            dark_bg = not dark_bg
        elif key == ord('a'):
            show_angles = not show_angles
        elif key == ord('m'):
            mirror = not mirror

    source.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
