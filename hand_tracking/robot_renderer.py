"""
robot_renderer.py — Cyberpunk/Robotic Hand Overlay Renderer
Draws the glowing skeleton, joints, and HUD info on top of the camera frame.
"""

import cv2
import numpy as np
import math
import time
from hand_tracker import (
    HandTracker, HAND_CONNECTIONS, FINGER_TIPS, FINGER_NAMES,
    WRIST, INDEX_TIP, THUMB_TIP, MIDDLE_TIP
)


# ─── Color palette (BGR) ──────────────────────────────────────────────────────
CYAN      = (255, 220,  50)   # glowing cyan
ORANGE    = ( 30, 120, 255)   # joint accent
GREEN     = ( 80, 255,  80)   # finger count / status
RED       = ( 30,  30, 255)   # warnings
WHITE     = (255, 255, 255)
DARK_CYAN = (150, 100,  20)
MAGENTA   = (200,  50, 200)


def glow_line(img, p1, p2, color, thickness=2, blur=9):
    """Draw a line with a soft glow effect (layered alpha blend)."""
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, thickness + 4)
    cv2.GaussianBlur(overlay, (blur, blur), 0, dst=overlay)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.line(img, p1, p2, color, thickness)


def glow_circle(img, center, r, color, thickness=1, blur=7):
    """Draw a circle with glow."""
    overlay = img.copy()
    cv2.circle(overlay, center, r + 3, color, thickness + 2)
    cv2.GaussianBlur(overlay, (blur, blur), 0, dst=overlay)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.circle(img, center, r, color, thickness)


def draw_hand_skeleton(img, pts: list, fingers_up: list):
    """
    Draw the robotic skeleton with color-coded fingers.
    pts : list of (px, py) pixel positions, length 21
    """
    if len(pts) < 21:
        return

    finger_colors = [ORANGE, CYAN, GREEN, MAGENTA, WHITE]

    # ── Bone segments ─────────────────────────────────────────────────────────
    from hand_tracker import HAND_CONNECTIONS, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP, WRIST
    finger_segment_map = {
        # (start_lm, end_lm): finger_color_index
        **{(i, i+1): None for i in range(20)}  # placeholder, see below
    }

    for a, b in HAND_CONNECTIONS:
        pa = pts[a]
        pb = pts[b]

        # Colour the bone by which finger it belongs to
        if a in range(1, 5) or b in range(1, 5):      color = ORANGE
        elif a in range(5, 9) or b in range(5, 9):    color = CYAN
        elif a in range(9, 13) or b in range(9, 13):  color = GREEN
        elif a in range(13, 17) or b in range(13, 17):color = MAGENTA
        elif a in range(17, 21) or b in range(17, 21):color = WHITE
        else:                                           color = DARK_CYAN

        glow_line(img, pa, pb, color, thickness=2)

    # ── Joint circles ─────────────────────────────────────────────────────────
    for i, pt in enumerate(pts):
        if i == 0:                         # wrist
            r, col = 7, CYAN
        elif i in [4, 8, 12, 16, 20]:     # fingertips
            r, col = 6, ORANGE
        elif i in [5, 9, 13, 17]:         # MCP knuckles
            r, col = 5, CYAN
        else:
            r, col = 3, DARK_CYAN
        glow_circle(img, pt, r, col, thickness=2)

    # ── Animated pulse at fingertips ──────────────────────────────────────────
    t = time.time()
    for idx, tip_id in enumerate([4, 8, 12, 16, 20]):
        if fingers_up[idx]:
            pulse_r = int(10 + 4 * math.sin(t * 5 + idx))
            cv2.circle(img, pts[tip_id], pulse_r, CYAN, 1)


def draw_hud(img, h: int, w: int, hand_idx: int, label: str,
             finger_count: int, gesture: str, fingers_up: list,
             bbox: tuple, pinch_dist: float, fps: float):
    """
    Render the cyberpunk HUD overlay for one detected hand.
    """
    # ── Bounding box ──────────────────────────────────────────────────────────
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 1)

    # Corner brackets (robotic look)
    L = 18
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                            (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), ORANGE, 2)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), ORANGE, 2)

    # Label above box
    cv2.putText(img, f"HAND {hand_idx + 1}: {label}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)

    # ── Side panel info ───────────────────────────────────────────────────────
    px = 14   # left panel x
    py = 80 + hand_idx * 180

    info_lines = [
        (f"[ HAND {hand_idx + 1} ]",   ORANGE, 0.55, 2),
        (f"Side   : {label}",           WHITE,  0.45, 1),
        (f"Fingers: {finger_count}",    GREEN,  0.45, 1),
        (f"Gesture: {gesture}",         CYAN,   0.45, 1),
        (f"Pinch  : {pinch_dist:.0f}px",MAGENTA,0.45, 1),
    ]

    for i, (text, color, scale, thick) in enumerate(info_lines):
        cv2.putText(img, text, (px, py + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    # Finger status bar
    for fi, fname in enumerate(["T", "I", "M", "R", "P"]):
        col  = GREEN if fingers_up[fi] else RED
        icon = "■" if fingers_up[fi] else "□"
        cv2.putText(img, f"{icon}{fname}",
                    (px + fi * 34, py + 5 * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)


def draw_global_hud(img, h: int, w: int, fps: float, hand_count: int):
    """Top-right FPS + status panel."""
    # Semi-transparent dark bar
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 200, 0), (w, 70), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, f"FPS: {fps:5.1f}",
                (w - 185, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1)
    cv2.putText(img, f"HANDS: {hand_count}",
                (w - 185, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)
    status = "TRACKING" if hand_count else "SCANNING..."
    cv2.putText(img, status,
                (w - 185, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                GREEN if hand_count else ORANGE, 1)

    # Scan-line effect (subtle)
    for y in range(0, h, 6):
        cv2.line(img, (0, y), (w, y), (0, 0, 0), 1)
    # Re-blend scanlines very faintly
    scan = img.copy()
    cv2.addWeighted(scan, 0.92, img, 0.08, 0, img)


def draw_pinch_line(img, pts: list):
    """Draw the thumb-index pinch distance line."""
    if len(pts) < 9:
        return
    thumb = pts[4]
    index = pts[8]
    mid   = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)

    d = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
    color = RED if d < 50 else CYAN

    cv2.line(img, thumb, index, color, 1)
    cv2.circle(img, mid, 5, ORANGE, -1)
    cv2.putText(img, f"{int(d)}px", (mid[0] + 5, mid[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
