#  3D Human Pose Estimation for Digital Twins

Real-time 3D human pose estimation using **MediaPipe** and **OpenCV**, with a depth-colored skeleton overlay, joint angle analysis, posture classification, and a cyberpunk HUD — designed as a Digital Twin visualization system.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-orange)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)

---

##  Features

-  **33-landmark 3D pose detection** with real-world depth (x, y, z)
-  **Depth-colored skeleton** — bones shift color/brightness based on z-depth
-  **8 joint angles** computed in real time (elbows, shoulders, hips, knees)
-  **Posture classification** — Standing / Sitting / Lying
-  **Switchable input** — live webcam or video file (press `W`)
-  **Cyberpunk HUD** — angle bar chart, bounding box brackets, FPS counter
- Screenshot capture, fullscreen, mirror mode

---

##  Project Structure

```
pose_estimation/
├── main.py                      # Entry point — camera loop & controls
├── pose_tracker.py              # MediaPipe wrapper — landmarks, angles, posture
├── pose_renderer.py             # Drawing code — skeleton, HUD, glow effects
├── pose_landmarker.task         # MediaPipe model file (download separately)
└── README.md
```

---

##  Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/3d-pose-digital-twin.git
cd 3d-pose-digital-twin
```

### 2. Install dependencies
```bash
# Windows
py -m pip install mediapipe opencv-python numpy

# Linux / Mac
pip install mediapipe opencv-python numpy
```

### 3. Download the MediaPipe Pose model

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task" -OutFile "pose_landmarker.task"
```
```bash
# Linux / Mac
wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

### 4. Run

```bash
# Webcam (default)
py main.py

# Video file
py main.py --video path/to/video.mp4

# Custom camera index
py main.py --camera 1
```

---

##  Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `W` | Switch between webcam and video |
| `F` | Toggle fullscreen |
| `S` | Save screenshot |
| `H` | Toggle HUD panels |
| `B` | Toggle dark background |
| `A` | Toggle joint angle labels |
| `M` | Toggle mirror mode |

---

## How It Works

### Landmark Detection
MediaPipe detects **33 body landmarks** per frame, each with:
- `x`, `y` — normalized screen coordinates
- `z` — depth relative to the hip midpoint (negative = closer to camera)
- `visibility` — confidence score (0–1)

### Depth Coloring
Bone brightness is mapped from the `z` coordinate — closer limbs appear brighter, creating a pseudo-3D depth effect on the 2D camera feed.

### Joint Angle Calculation
Angles are computed using the dot product formula at each joint vertex:

```
angle = arccos( (A-V) · (B-V) / |A-V| |B-V| )
```

Tracked joints: Left/Right Elbow, Shoulder, Hip, Knee.

### Posture Classification
Simple rule-based classification using relative y-positions of hip, knee, and shoulder landmarks.

| Posture | Condition |
|---------|-----------|
| Standing | Hip significantly above knee, shoulders high |
| Sitting | Hip and knee at similar y-level |
| Lying | Shoulders below y=0.6 threshold |

---

##  Requirements

- Python 3.10+ (tested on 3.14)
- Webcam or video file
- mediapipe >= 0.10.14
- opencv-python >= 4.8
- numpy >= 1.24

---

##  Extension Ideas

-  Log joint angles to CSV for biomechanics analysis
-  Build a rep counter for exercise tracking (e.g. squat counter using knee angle)
-  Control a 3D avatar in Unity/Blender via OSC/socket stream
-  Feed joint angles as state space into an RL agent
-  Fall detection using posture + acceleration heuristics

---

##  Author

**Dua Fatima** — MS Robotics & AI Engineering, NUST Islamabad  
GitHub: [@dfatimas504](https://github.com/dfatimas504)

