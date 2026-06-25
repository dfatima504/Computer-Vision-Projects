# 🤖 Real-Time Robotic Hand Tracking

A cyberpunk-style real-time hand tracking system built with **OpenCV** and **MediaPipe**, featuring a glowing robotic skeleton overlay, gesture recognition, and a live HUD display.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-orange)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)

---

## ✨ Features

- 🖐 Tracks up to **2 hands simultaneously** in real time
- 🦾 **Robotic skeleton overlay** with color-coded fingers and glowing joints
- 👌 **Gesture recognition** — Fist, Peace, Point, Hang Loose, Open, and more
- 📐 **Pinch distance** measurement between thumb and index finger
- 🖥 **Cyberpunk HUD** with bounding box, finger status bar, FPS counter
- 📸 Screenshot capture with one keypress
- 🔄 Mirror mode, dark background blend, fullscreen toggle

---

## 📁 Project Structure

```
hand_tracking/
├── main.py                  # Entry point — camera loop & controls
├── hand_tracker.py          # MediaPipe wrapper — detection & gesture logic
├── robot_renderer.py        # All drawing code — glow effects, HUD, UI
├── hand_landmarker.task     # MediaPipe model file (download separately)
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/robotic-hand-tracking.git
cd robotic-hand-tracking
```

### 2. Install dependencies
```bash
# Windows
py -m pip install mediapipe opencv-python numpy

# Linux / Mac
pip install mediapipe opencv-python numpy
```

### 3. Download the MediaPipe model
```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```
```bash
# Linux / Mac
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 4. Run
```bash
# Windows
py main.py

# Linux / Mac
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `F` | Toggle fullscreen |
| `S` | Save screenshot |
| `M` | Toggle mirror mode |
| `H` | Toggle HUD panels |
| `B` | Toggle dark background |

---

## 🚀 Optional Flags

```bash
py main.py --camera 1            # use webcam index 1
py main.py --width 1280 --height 720
py main.py --hands 2             # max hands to track
py main.py --det 0.8 --track 0.7 # detection & tracking confidence
```

---

## 🧠 How It Works

| Module | Role |
|--------|------|
| `hand_tracker.py` | Wraps MediaPipe Tasks API — detects 21 landmarks per hand, classifies gestures, measures pinch distance |
| `robot_renderer.py` | Draws the glowing skeleton, animated fingertip pulses, bounding box brackets, and side HUD panels |
| `main.py` | Opens webcam, runs the tracking loop, handles keyboard input |

MediaPipe detects **21 landmarks** per hand. The tracker uses y-coordinate comparison (tip vs PIP joint) to determine which fingers are extended, enabling gesture classification without any ML training.

---

## 🔧 Requirements

- Python 3.10+ (tested on 3.14)
- Webcam
- mediapipe >= 0.10.14
- opencv-python >= 4.8
- numpy >= 1.24

---

## 💡 Extension Ideas

- 🔊 Map pinch distance to system volume (`pycaw`)
- 🖱 Use index fingertip as mouse cursor (`pyautogui`)
- 🤟 Train a classifier on the 21 landmarks for sign language recognition
- 🎮 Build gesture-controlled games

---

## 👩‍💻 Author

**Dua Fatima** — MS Robotics & AI Engineering, NUST Islamabad  
GitHub: [@dfatimas504](https://github.com/dfatimas504)

---

## 📄 License

MIT License — free to use, modify, and distribute.
