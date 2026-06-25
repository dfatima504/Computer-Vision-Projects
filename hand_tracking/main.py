"""
main.py — Real-Time Robotic Hand Tracking
==========================================
Controls:
  Q / ESC  → Quit
  F        → Toggle fullscreen
  S        → Save screenshot
  M        → Toggle mirror mode
  H        → Toggle HUD panels
  B        → Toggle dark background blend

Run:
  python main.py
  python main.py --camera 1          # use webcam index 1
  python main.py --width 1280 --height 720
"""

import cv2
import numpy as np
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime

from hand_tracker import HandTracker, FINGER_NAMES
from robot_renderer import (
    draw_hand_skeleton, draw_hud, draw_global_hud, draw_pinch_line,
    CYAN, GREEN, ORANGE
)


# ─── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Robotic Hand Tracking")
    ap.add_argument("--camera",  type=int, default=0,    help="Webcam index")
    ap.add_argument("--width",   type=int, default=1280, help="Frame width")
    ap.add_argument("--height",  type=int, default=720,  help="Frame height")
    ap.add_argument("--hands",   type=int, default=2,    help="Max hands to track")
    ap.add_argument("--det",     type=float, default=0.75, help="Detection confidence")
    ap.add_argument("--track",   type=float, default=0.70, help="Tracking confidence")
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


# ─── Dark background blending ──────────────────────────────────────────────────

def apply_dark_blend(frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Reduce the real camera image for a more cyberpunk look."""
    dark = (frame.astype(np.float32) * alpha).astype(np.uint8)
    # slight green channel boost
    dark[:, :, 1] = np.clip(dark[:, :, 1].astype(int) + 8, 0, 255).astype(np.uint8)
    return dark


# ─── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    tracker   = HandTracker(args.hands, args.det, args.track)
    fps_ctr   = FPSCounter()

    # Toggleable state
    mirror    = True
    show_hud  = True
    dark_bg   = True
    fullscr   = False

    WIN       = "🤖 Robotic Hand Tracking  |  Q=quit  F=fullscreen  S=screenshot"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print("=" * 60)
    print("  ROBOTIC HAND TRACKING  |  OpenCV + MediaPipe")
    print("=" * 60)
    print("  Controls:")
    print("   Q / ESC  → Quit")
    print("   F        → Toggle fullscreen")
    print("   S        → Save screenshot")
    print("   M        → Mirror mode on/off")
    print("   H        → Toggle HUD panels")
    print("   B        → Toggle dark background")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame.")
            continue

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        fps  = fps_ctr.tick()

        # ── Run tracking ──────────────────────────────────────────────────────
        found = tracker.process(frame)

        # ── Visual background ─────────────────────────────────────────────────
        if dark_bg:
            canvas = apply_dark_blend(frame, alpha=0.30)
        else:
            canvas = frame.copy()

        # ── Draw all hands ────────────────────────────────────────────────────
        n_hands = tracker.hand_count()
        for hi in range(n_hands):
            pts       = tracker.get_pixel_landmarks(hi, w, h)
            up        = tracker.fingers_up(hi)
            count     = tracker.count_fingers(hi)
            gesture   = tracker.gesture_name(hi)
            label     = tracker.handedness(hi)
            bbox      = tracker.bounding_box(hi, w, h)
            pinch_d   = tracker.distance_between(hi, 4, 8, w, h)

            draw_hand_skeleton(canvas, pts, up)
            draw_pinch_line(canvas, pts)

            if show_hud:
                draw_hud(canvas, h, w, hi, label, count,
                         gesture, up, bbox, pinch_d, fps)

        # ── Global HUD ────────────────────────────────────────────────────────
        if show_hud:
            draw_global_hud(canvas, h, w, fps, n_hands)

        # Help bar at bottom
        help_txt = "Q=quit  F=fullscreen  S=screenshot  M=mirror  H=hud  B=bg"
        cv2.putText(canvas, help_txt, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # ── Show ──────────────────────────────────────────────────────────────
        cv2.imshow(WIN, canvas)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):          # Q or ESC
            break
        elif key == ord('f'):
            fullscr = not fullscr
            prop = cv2.WND_PROP_FULLSCREEN
            cv2.setWindowProperty(WIN, prop,
                cv2.WINDOW_FULLSCREEN if fullscr else cv2.WINDOW_NORMAL)
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{ts}.png"
            cv2.imwrite(path, canvas)
            print(f"[SAVED] {path}")
        elif key == ord('m'):
            mirror = not mirror
            print(f"[INFO] Mirror: {'ON' if mirror else 'OFF'}")
        elif key == ord('h'):
            show_hud = not show_hud
        elif key == ord('b'):
            dark_bg = not dark_bg

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracker stopped.")


if __name__ == "__main__":
    main()
