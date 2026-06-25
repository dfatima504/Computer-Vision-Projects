"""
pose_renderer.py — 3D Pose Skeleton Renderer
Draws depth-colored skeleton, joint angles, and cyberpunk HUD on camera feed.
"""

import cv2
import numpy as np
import math
import time
from pose_tracker import (
    POSE_CONNECTIONS, JOINT_ANGLES,
    TORSO_IDS, LARM_IDS, RARM_IDS, LLEG_IDS, RLEG_IDS, FACE_IDS,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE, NOSE
)

# ─── Color palette (BGR) ──────────────────────────────────────────────────────
CYAN      = (255, 220,  50)
ORANGE    = ( 30, 120, 255)
GREEN     = ( 80, 255,  80)
MAGENTA   = (200,  50, 200)
WHITE     = (255, 255, 255)
RED       = ( 30,  30, 255)
YELLOW    = (  0, 220, 255)
DARK_CYAN = (120,  80,  20)
BLUE      = (220, 100,  30)

# Per body-part colors
PART_COLORS = {
    "face":   WHITE,
    "torso":  CYAN,
    "l_arm":  GREEN,
    "r_arm":  ORANGE,
    "l_leg":  MAGENTA,
    "r_leg":  YELLOW,
}


def _bone_color(a: int, b: int) -> tuple:
    """Pick color based on which body part the bone belongs to."""
    ids = {a, b}
    if ids & FACE_IDS:   return PART_COLORS["face"]
    if ids & TORSO_IDS and not (ids & LARM_IDS or ids & RARM_IDS
                                or ids & LLEG_IDS or ids & RLEG_IDS):
        return PART_COLORS["torso"]
    if ids & LARM_IDS:   return PART_COLORS["l_arm"]
    if ids & RARM_IDS:   return PART_COLORS["r_arm"]
    if ids & LLEG_IDS:   return PART_COLORS["l_leg"]
    if ids & RLEG_IDS:   return PART_COLORS["r_leg"]
    return DARK_CYAN


def _depth_alpha(z: float) -> float:
    """Map z (depth, negative = closer) to brightness 0.4–1.0."""
    # z typically in [-0.5, 0.5]; closer = more opaque
    return max(0.4, min(1.0, 1.0 - z * 1.5))


def glow_line(img, p1, p2, color, thickness=2, alpha=1.0):
    """Draw a line with soft glow and optional depth fade."""
    faded = tuple(int(c * alpha) for c in color)
    overlay = img.copy()
    cv2.line(overlay, p1, p2, faded, thickness + 4)
    cv2.GaussianBlur(overlay, (7, 7), 0, dst=overlay)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.line(img, p1, p2, faded, thickness)


def glow_circle(img, center, r, color, thickness=2):
    overlay = img.copy()
    cv2.circle(overlay, center, r + 3, color, thickness + 2)
    cv2.GaussianBlur(overlay, (7, 7), 0, dst=overlay)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.circle(img, center, r, color, thickness)


def draw_pose_skeleton(img, pts: list):
    """
    Draw the 3D-depth-colored pose skeleton.
    pts: list of (px, py, z, visibility) — length 33
    """
    if len(pts) < 33:
        return

    # ── Bones ─────────────────────────────────────────────────────────────────
    for a, b in POSE_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        pa_x, pa_y, za, vis_a = pts[a]
        pb_x, pb_y, zb, vis_b = pts[b]

        # Skip low-visibility landmarks
        if vis_a < 0.4 or vis_b < 0.4:
            continue

        z_avg = (za + zb) / 2
        color = _bone_color(a, b)
        alpha = _depth_alpha(z_avg)
        glow_line(img, (pa_x, pa_y), (pb_x, pb_y), color,
                  thickness=2, alpha=alpha)

    # ── Joints ────────────────────────────────────────────────────────────────
    key_joints = {
        LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
        11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
    }
    for i, (px, py, z, vis) in enumerate(pts):
        if vis < 0.4:
            continue
        r     = 6 if i in key_joints else 3
        color = _bone_color(i, i)
        alpha = _depth_alpha(z)
        faded = tuple(int(c * alpha) for c in color)
        glow_circle(img, (px, py), r, faded)

    # ── Animated pulse on wrists & ankles ─────────────────────────────────────
    t = time.time()
    for idx, col in [(LEFT_WRIST, GREEN), (RIGHT_WRIST, ORANGE),
                     (LEFT_ANKLE, MAGENTA), (RIGHT_ANKLE, YELLOW)]:
        px, py, z, vis = pts[idx]
        if vis > 0.5:
            pulse_r = int(10 + 4 * math.sin(t * 4 + idx))
            cv2.circle(img, (px, py), pulse_r, col, 1)


def draw_joint_angles(img, pts: list, angles: dict):
    """
    Draw angle arcs and labels at key joints.
    """
    # Map joint name → vertex landmark index
    vertex_map = {name: v for name, (a, v, b) in JOINT_ANGLES.items()}

    for name, angle in angles.items():
        v_idx = vertex_map[name]
        if v_idx >= len(pts):
            continue
        px, py, z, vis = pts[v_idx]
        if vis < 0.5:
            continue

        color = GREEN if angle > 150 else (YELLOW if angle > 90 else RED)
        cv2.putText(img, f"{int(angle)}",
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        # Small arc indicator
        cv2.ellipse(img, (px, py), (12, 12), 0, 0, int(angle * 0.5),
                    color, 1)


def draw_pose_hud(img, h: int, w: int, angles: dict,
                  posture: str, bbox: tuple, fps: float):
    """Left panel HUD with posture, joint angles, FPS."""

    # ── Bounding box ──────────────────────────────────────────────────────────
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 1)
    L = 20
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                            (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), ORANGE, 2)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), ORANGE, 2)

    # ── Semi-transparent left panel ───────────────────────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (210, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # Title
    cv2.putText(img, "[ DIGITAL TWIN ]", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ORANGE, 1)
    cv2.line(img, (8, 30), (200, 30), DARK_CYAN, 1)

    # Posture
    pcol = GREEN if posture == "Standing" else (YELLOW if posture == "Sitting" else CYAN)
    cv2.putText(img, f"Posture : {posture}", (8, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, pcol, 1)

    # Joint angles
    cv2.putText(img, "Joint Angles (deg)", (8, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)
    cv2.line(img, (8, 80), (200, 80), DARK_CYAN, 1)

    for i, (name, angle) in enumerate(angles.items()):
        color = GREEN if angle > 150 else (YELLOW if angle > 90 else RED)
        bar_w = int((angle / 180.0) * 90)
        y = 96 + i * 24
        cv2.putText(img, f"{name:<12} {int(angle):>3}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
        cv2.rectangle(img, (145, y - 10), (145 + bar_w, y - 2), color, -1)

    # FPS bottom
    cv2.line(img, (8, h - 40), (200, h - 40), DARK_CYAN, 1)
    cv2.putText(img, f"FPS : {fps:5.1f}", (8, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, GREEN, 1)


def draw_global_hud(img, h: int, w: int, fps: float,
                    detected: bool, source: str):
    """Top-right status bar."""
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 220, 0), (w, 75), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, f"FPS    : {fps:5.1f}", (w - 205, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, GREEN, 1)
    cv2.putText(img, f"SOURCE : {source}", (w - 205, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1)
    status = "POSE DETECTED" if detected else "SCANNING..."
    cv2.putText(img, status, (w - 205, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                GREEN if detected else ORANGE, 1)

    # Subtle scanlines
    for y in range(0, h, 8):
        cv2.line(img, (0, y), (w, y), (0, 0, 0), 1)
    scan = img.copy()
    cv2.addWeighted(scan, 0.93, img, 0.07, 0, img)


def apply_dark_blend(frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    dark = (frame.astype(np.float32) * alpha).astype(np.uint8)
    dark[:, :, 1] = np.clip(dark[:, :, 1].astype(int) + 6, 0, 255).astype(np.uint8)
    return dark
