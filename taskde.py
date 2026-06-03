"""
Phase 3 Motor Task Analyzer
============================
Voice-guided precision motor assessment with ESP32-CAM pose analysis.
Tasks: Raise Hands, Touch Head, Clap, Tap Button, Say Name (voice)

ESP32-CAM Setup:
    1. Flash ESP32-CAM with CameraWebServer example (Arduino IDE)
    2. Set your WiFi credentials in the sketch
    3. Note the IP shown in Serial Monitor
    4. Set ESP32_CAM_URL below to: http://<your-ip>/stream

Requirements:
    pip install mediapipe opencv-python numpy pyttsx3 SpeechRecognition pyaudio requests

Run:
    python phase3_motor_espcam.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
import queue
import math
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque

# ─── ESP32-CAM URL ────────────────────────────────────────────────────────────
ESP32_CAM_URL = "http://192.168.137.69/stream"   # ← change to your ESP32-CAM IP

# ─── Optional voice/audio imports ────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ─── MediaPipe setup ──────────────────────────────────────────────────────────
mp_pose   = mp.solutions.pose
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ─── Constants ────────────────────────────────────────────────────────────────
TASKS = {
    1: "Raise Both Hands",
    2: "Touch Head with Left Hand",
    3: "Clap Your Hands",
    4: "Tap Finger on Screen Button",
    5: "Say Your Name (Voice)",
}

TASK_VOICE_PROMPTS = {
    1: "Task one. Please raise both hands above your head.",
    2: "Task two. Touch your head with your left hand.",
    3: "Task three. Clap your hands together.",
    4: "Task four. Tap the button shown on screen with your finger.",
    5: "Task five. Say your name clearly when prompted.",
}

TASK_INSTRUCTIONS = {
    1: "Raise BOTH hands above head level",
    2: "Touch head with LEFT hand (right wrist near nose)",
    3: "Bring both hands together (clap)",
    4: "Point finger at the RED button on screen",
    5: "Press [V] to speak your name",
}

COLORS = {
    "bg":       (15, 15, 25),
    "accent":   (0, 220, 180),
    "warn":     (0, 140, 255),
    "danger":   (50, 50, 255),
    "success":  (50, 220, 100),
    "text":     (230, 230, 240),
    "muted":    (130, 130, 150),
    "gold":     (30, 200, 255),
    "panel":    (25, 28, 45),
}

# ─── Dataclasses ─────────────────────────────────────────────────────────────
@dataclass
class TaskResult:
    task_id: int
    task_name: str
    score: float        = 0.0   # 0-100
    attempts: int       = 0
    best_confidence: float = 0.0
    reaction_time: float   = 0.0  # seconds
    hold_duration: float   = 0.0  # seconds held correctly
    completed: bool        = False
    timestamp: float       = 0.0
    detail_scores: Dict    = field(default_factory=dict)

@dataclass
class AnalyticsWindow:
    """Rolling window for smoothing confidence values."""
    size: int = 15
    values: deque = field(default_factory=lambda: deque(maxlen=15))

    def push(self, v: float):
        self.values.append(v)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def trend(self) -> float:
        """Positive = improving, negative = worsening."""
        if len(self.values) < 4:
            return 0.0
        first_half = list(self.values)[:len(self.values)//2]
        second_half = list(self.values)[len(self.values)//2:]
        return (sum(second_half)/len(second_half)) - (sum(first_half)/len(first_half))


# ─── ESP32-CAM Stream Reader ──────────────────────────────────────────────────
class ESP32CamCapture:
    """
    Drop-in replacement for cv2.VideoCapture.
    Reads MJPEG stream from ESP32-CAM /stream endpoint in a background thread.
    Call read() exactly like cap.read() in the original code.
    """
    def __init__(self, url: str):
        self.url      = url
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def open(self):
        self._running = True
        self._thread  = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        # Wait up to 8 s for first frame
        deadline = time.time() + 8
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.1)
        print("[ESP32-CAM] Warning: no frame received yet, continuing anyway.")
        return True

    def isOpened(self):
        return self._running

    def read(self):
        with self._lock:
            if self._frame is None:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                return False, blank
            return True, self._frame.copy()

    def release(self):
        self._running = False

    def set(self, prop, val):
        pass  # resolution is set on the device

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 1280
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 720
        return 0

    def _stream_loop(self):
        SOI = b'\xff\xd8'
        EOI = b'\xff\xd9'
        while self._running:
            try:
                print(f"[ESP32-CAM] Connecting to {self.url} ...")
                resp = requests.get(self.url, stream=True, timeout=(8, None))
                resp.raise_for_status()
                print("[ESP32-CAM] Stream connected ✓")
                buf = b""
                for chunk in resp.iter_content(chunk_size=8192):
                    if not self._running:
                        break
                    buf += chunk
                    while True:
                        start = buf.find(SOI)
                        if start == -1:
                            buf = b""
                            break
                        end = buf.find(EOI, start)
                        if end == -1:
                            buf = buf[start:]
                            break
                        jpeg = buf[start:end + 2]
                        buf  = buf[end + 2:]
                        arr  = np.frombuffer(jpeg, dtype=np.uint8)
                        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            with self._lock:
                                self._frame = img
            except Exception as e:
                print(f"[ESP32-CAM] Error: {e}. Retrying in 2s...")
                time.sleep(2)


# ─── Voice Engine ─────────────────────────────────────────────────────────────
class VoiceEngine:
    def __init__(self):
        self.tts = None
        self.recognizer = None
        self.microphone = None
        self._queue = queue.Queue()
        self._thread = None
        self._init_tts()
        self._init_sr()

    def _init_tts(self):
        if TTS_AVAILABLE:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty('rate', 160)
                self.tts.setProperty('volume', 0.9)
                print("[Voice] TTS engine ready.")
            except Exception as e:
                print(f"[Voice] TTS init failed: {e}")
        else:
            print("[Voice] pyttsx3 not available. Install: pip install pyttsx3")

    def _init_sr(self):
        if SR_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone  = sr.Microphone()
                print("[Voice] Speech recognition ready.")
            except Exception as e:
                print(f"[Voice] SR init failed: {e}")
        else:
            print("[Voice] SpeechRecognition not available. Install: pip install SpeechRecognition pyaudio")

    def speak(self, text: str, blocking=False):
        if self.tts:
            if blocking:
                self.tts.say(text)
                self.tts.runAndWait()
            else:
                t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
                t.start()

    def _speak_worker(self, text: str):
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception:
            pass

    def listen(self, timeout=5, phrase_limit=4) -> Optional[str]:
        """Record from mic and return recognized text or None."""
        if not (self.recognizer and self.microphone):
            return None
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            text = self.recognizer.recognize_google(audio)
            return text
        except Exception:
            return None

    def listen_from_file(self, filepath: str) -> Optional[str]:
        """Recognize speech from a local mp3/wav file."""
        if not self.recognizer:
            return None
        try:
            with sr.AudioFile(filepath) as source:
                audio = self.recognizer.record(source)
            return self.recognizer.recognize_google(audio)
        except Exception as e:
            print(f"[Voice] File recognition error: {e}")
            return None


# ─── Pose Analyzer ────────────────────────────────────────────────────────────
class PoseAnalyzer:
    """Extracts normalized body landmarks and computes motor metrics."""

    # Landmark shortcuts
    NOSE      = mp_pose.PoseLandmark.NOSE
    L_WRIST   = mp_pose.PoseLandmark.LEFT_WRIST
    R_WRIST   = mp_pose.PoseLandmark.RIGHT_WRIST
    L_ELBOW   = mp_pose.PoseLandmark.LEFT_ELBOW
    R_ELBOW   = mp_pose.PoseLandmark.RIGHT_ELBOW
    L_SHOULDER= mp_pose.PoseLandmark.LEFT_SHOULDER
    R_SHOULDER= mp_pose.PoseLandmark.RIGHT_SHOULDER
    L_HIP     = mp_pose.PoseLandmark.LEFT_HIP
    R_HIP     = mp_pose.PoseLandmark.RIGHT_HIP
    L_INDEX   = mp_pose.PoseLandmark.LEFT_INDEX
    R_INDEX   = mp_pose.PoseLandmark.RIGHT_INDEX

    def __init__(self):
        self.pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, rgb_frame):
        pose_result  = self.pose.process(rgb_frame)
        hands_result = self.hands.process(rgb_frame)
        return pose_result, hands_result

    @staticmethod
    def lm_to_xy(lm_obj) -> np.ndarray:
        return np.array([lm_obj.x, lm_obj.y])

    @staticmethod
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def extract_keypoints(self, pose_result) -> Optional[dict]:
        if not pose_result.pose_landmarks:
            return None
        lm = pose_result.pose_landmarks.landmark
        def xy(idx): return self.lm_to_xy(lm[idx])
        def v(idx): return lm[idx].visibility

        kp = {
            "nose":       xy(self.NOSE),
            "l_wrist":    xy(self.L_WRIST),
            "r_wrist":    xy(self.R_WRIST),
            "l_elbow":    xy(self.L_ELBOW),
            "r_elbow":    xy(self.R_ELBOW),
            "l_shoulder": xy(self.L_SHOULDER),
            "r_shoulder": xy(self.R_SHOULDER),
            "l_hip":      xy(self.L_HIP),
            "r_hip":      xy(self.R_HIP),
            "l_index":    xy(self.L_INDEX),
            "r_index":    xy(self.R_INDEX),
            "vis_l_wrist":  v(self.L_WRIST),
            "vis_r_wrist":  v(self.R_WRIST),
            "vis_nose":     v(self.NOSE),
        }

        # Derived body scale (shoulder width + torso height weighted)
        shoulder_w = self.dist(kp["l_shoulder"], kp["r_shoulder"])
        torso_h    = self.dist(
            (kp["l_shoulder"] + kp["r_shoulder"]) / 2,
            (kp["l_hip"]      + kp["r_hip"])      / 2
        )
        kp["body_scale"] = (shoulder_w * 0.4 + torso_h * 0.6) + 1e-6
        kp["shoulder_w"] = shoulder_w
        kp["torso_h"]    = torso_h
        return kp

    # ── Task-specific confidence functions ───────────────────────────────────

    def score_raise_hands(self, kp: dict) -> dict:
        """Both wrists must be above nose."""
        nose   = kp["nose"]
        lw, rw = kp["l_wrist"], kp["r_wrist"]
        bs     = kp["body_scale"]

        # Height above nose (positive = above)
        l_above = (nose[1] - lw[1]) / bs
        r_above = (nose[1] - rw[1]) / bs

        # Symmetry bonus
        symmetry = 1.0 - abs(l_above - r_above) / (abs(l_above) + abs(r_above) + 1e-6)

        # Arm extension (elbows less bent)
        l_ext = self.dist(kp["l_shoulder"], lw) / (bs + 1e-6)
        r_ext = self.dist(kp["r_shoulder"], rw) / (bs + 1e-6)
        extension = min(1.0, (l_ext + r_ext) / 3.0)

        raw = (l_above + r_above) / 2.0
        conf = np.clip(raw, 0, 1)
        detected = lw[1] < nose[1] and rw[1] < nose[1]

        score = 0.0
        if detected:
            score = conf * 0.6 + symmetry * 0.2 + extension * 0.2

        return {
            "confidence": float(np.clip(score, 0, 1)),
            "detected": detected,
            "l_height": float(l_above),
            "r_height": float(r_above),
            "symmetry": float(symmetry),
            "extension": float(extension),
            "sub_scores": {
                "height": conf,
                "symmetry": symmetry,
                "extension": extension,
            }
        }

    def score_touch_head(self, kp: dict) -> dict:
        """Left wrist (or right_wrist) near nose."""
        nose = kp["nose"]
        lw   = kp["l_wrist"]
        rw   = kp["r_wrist"]
        bs   = kp["body_scale"]

        d_left  = self.dist(lw, nose) / bs
        d_right = self.dist(rw, nose) / bs
        best_d  = min(d_left, d_right)
        which   = "left" if d_left < d_right else "right"

        threshold = 0.55
        conf = float(np.clip(1.0 - best_d / threshold, 0, 1))
        detected = best_d < threshold

        # Precision bonus: closer = better
        precision = float(np.clip(1.0 - best_d / (threshold * 0.5), 0, 1))

        score = conf * 0.7 + precision * 0.3 if detected else conf * 0.5

        return {
            "confidence": float(np.clip(score, 0, 1)),
            "detected": detected,
            "dist_norm": float(best_d),
            "which_hand": which,
            "precision": float(precision),
            "sub_scores": {
                "proximity": conf,
                "precision": precision,
            }
        }

    def score_clap(self, kp: dict) -> dict:
        """Both wrists close together and near chest level."""
        lw, rw = kp["l_wrist"], kp["r_wrist"]
        bs     = kp["body_scale"]
        mid_y  = (kp["l_shoulder"][1] + kp["r_shoulder"][1]) / 2

        hand_dist = self.dist(lw, rw) / bs
        threshold = 0.4
        conf = float(np.clip(1.0 - hand_dist / threshold, 0, 1))
        detected = hand_dist < threshold

        # Chest-level bonus (hands near torso midline)
        center_x = (kp["l_shoulder"][0] + kp["r_shoulder"][0]) / 2
        clap_center_x = (lw[0] + rw[0]) / 2
        centering = float(np.clip(1.0 - abs(clap_center_x - center_x) / (bs * 0.5), 0, 1))

        score = conf * 0.75 + centering * 0.25 if detected else conf * 0.5

        return {
            "confidence": float(np.clip(score, 0, 1)),
            "detected": detected,
            "dist_norm": float(hand_dist),
            "centering": float(centering),
            "sub_scores": {
                "proximity": conf,
                "centering": centering,
            }
        }

    def score_tap_button(self, kp: dict, button_rect, frame_size) -> dict:
        """Finger (index) points toward on-screen button region."""
        fw, fh = frame_size
        bx1, by1, bx2, by2 = button_rect

        # Normalize button to [0,1]
        nbx1, nbx2 = bx1/fw, bx2/fw
        nby1, nby2 = by1/fh, by2/fh
        bc = np.array([(nbx1+nbx2)/2, (nby1+nby2)/2])

        # Use index finger tip (closer to actual tap)
        ri = kp["r_index"]
        li = kp["l_index"]

        d_right = self.dist(ri, bc)
        d_left  = self.dist(li, bc)
        best_d  = min(d_right, d_left)
        which   = "right" if d_right < d_left else "left"

        bs = kp["body_scale"]
        threshold = 0.35
        conf = float(np.clip(1.0 - best_d / threshold, 0, 1))
        detected = best_d < threshold

        # Tip straightness (index extended toward button)
        # Use wrist→index vector alignment with button direction
        if which == "right":
            wrist = kp["r_wrist"]
            tip   = ri
        else:
            wrist = kp["l_wrist"]
            tip   = li

        vec_arm = tip - wrist
        vec_btn = bc  - wrist
        norm_arm = np.linalg.norm(vec_arm) + 1e-6
        norm_btn = np.linalg.norm(vec_btn) + 1e-6
        alignment = float(np.clip(np.dot(vec_arm, vec_btn) / (norm_arm * norm_btn), 0, 1))

        score = conf * 0.6 + alignment * 0.4 if detected else conf * 0.4

        return {
            "confidence": float(np.clip(score, 0, 1)),
            "detected": detected,
            "dist_norm": float(best_d),
            "alignment": float(alignment),
            "which_hand": which,
            "tip_pos": tip.tolist(),
            "sub_scores": {
                "proximity": conf,
                "alignment": alignment,
            }
        }


# ─── Score Calculator ─────────────────────────────────────────────────────────
class ScoreCalculator:
    """
    Converts raw detection events → task score (0–100).
    Factors: confidence, reaction time, hold duration, precision consistency.
    """
    REACTION_MAX = 8.0    # seconds; faster = better
    HOLD_TARGET  = 2.0    # seconds to hold for full hold score
    ATTEMPT_PENALTY = 5   # deducted per extra attempt beyond 1

    def compute(self, result: TaskResult) -> float:
        if not result.completed:
            # Partial score based on best confidence reached
            return round(result.best_confidence * 40, 1)  # max 40 if not completed

        base = 70.0

        # +Confidence score (up to 20 pts)
        conf_pts = result.best_confidence * 20

        # +Reaction time score (up to 5 pts)
        rt_pts = max(0, 5 * (1 - result.reaction_time / self.REACTION_MAX))

        # +Hold duration score (up to 5 pts)
        hold_pts = min(5, 5 * result.hold_duration / self.HOLD_TARGET)

        # -Attempt penalty
        attempt_pen = max(0, (result.attempts - 1) * self.ATTEMPT_PENALTY)

        total = base + conf_pts + rt_pts + hold_pts - attempt_pen
        return round(float(np.clip(total, 0, 100)), 1)


# ─── Main Application ─────────────────────────────────────────────────────────
class MotorTaskApp:
    def __init__(self):
        self.voice   = VoiceEngine()
        self.analyzer= PoseAnalyzer()
        self.scorer  = ScoreCalculator()

        # ── ESP32-CAM replaces cv2.VideoCapture(0) ────────────────────────
        self.cap = ESP32CamCapture(ESP32_CAM_URL)
        if not self.cap.open():
            raise RuntimeError("Cannot open ESP32-CAM stream.")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # App state
        self.phase: str = "intro"   # intro | task | review | done
        self.current_task: int = 0  # 1..5
        self.results: Dict[int, TaskResult] = {}
        self.task_start_time: float = 0
        self.task_timeout: float = 15.0

        # Per-task tracking
        self.conf_window  = AnalyticsWindow(size=20)
        self.hold_start   = 0.0
        self.holding      = False
        self.detected_once= False

        # On-screen tap button (task 4)
        bw, bh = 140, 70
        self.button_rect = (
            self.frame_w//2 - bw//2,
            self.frame_h//2 - bh//2,
            self.frame_w//2 + bw//2,
            self.frame_h//2 + bh//2,
        )

        # Voice state (task 5)
        self.voice_result: Optional[str] = None
        self.voice_listening = False

        # Overlay panels
        self.overlay_alpha = 0.72

        # FPS
        self.fps_counter = deque(maxlen=30)
        self.last_frame_t = time.time()

        print("\n=== Phase 3 Motor Task Analyzer [ESP32-CAM] ===")
        print("Press [SPACE] to begin | [Q] to quit")
        if not TTS_AVAILABLE:
            print("TIP: pip install pyttsx3   → enables voice prompts")
        if not SR_AVAILABLE:
            print("TIP: pip install SpeechRecognition pyaudio  → enables voice task")

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def draw_rounded_rect(self, img, x1, y1, x2, y2, r, color, alpha=1.0, thickness=-1):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, thickness)
        for cx, cy in [(x1+r, y1+r),(x2-r, y1+r),(x1+r, y2-r),(x2-r, y2-r)]:
            cv2.circle(overlay, (cx, cy), r, color, thickness)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        else:
            img[:] = overlay[:]

    def draw_panel(self, img, x1, y1, x2, y2, alpha=0.75):
        sub = img[y1:y2, x1:x2]
        bg  = np.zeros_like(sub)
        bg[:] = COLORS["panel"]
        cv2.addWeighted(bg, alpha, sub, 1-alpha, 0, sub)
        img[y1:y2, x1:x2] = sub
        cv2.rectangle(img, (x1,y1), (x2,y2), COLORS["accent"], 1)

    def put_text(self, img, text, pos, scale=0.7, color=None, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
        color = color or COLORS["text"]
        cv2.putText(img, text, pos, font, scale, (0,0,0), thickness+2)
        cv2.putText(img, text, pos, font, scale, color, thickness)

    def draw_progress_bar(self, img, x, y, w, h, value, color=None, bg_color=(40,40,60)):
        color = color or COLORS["accent"]
        cv2.rectangle(img, (x, y), (x+w, y+h), bg_color, -1)
        fill = int(w * np.clip(value, 0, 1))
        if fill > 0:
            cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), COLORS["muted"], 1)

    def sub_score_bars(self, img, sub_scores: dict, x, y, bar_w=120):
        for i, (label, val) in enumerate(sub_scores.items()):
            yy = y + i * 28
            self.put_text(img, f"{label}:", (x, yy+14), scale=0.42, color=COLORS["muted"])
            self.draw_progress_bar(img, x+80, yy, bar_w, 16, val,
                                   color=COLORS["success"] if val > 0.6 else COLORS["warn"])
            self.put_text(img, f"{val:.0%}", (x+80+bar_w+5, yy+13), scale=0.4, color=COLORS["text"])

    def color_for_conf(self, c):
        if c > 0.7: return COLORS["success"]
        if c > 0.4: return COLORS["warn"]
        return COLORS["danger"]

    # ── Phase Screens ─────────────────────────────────────────────────────────

    def draw_intro(self, img):
        h, w = img.shape[:2]
        overlay = img.copy()
        overlay[:] = (12, 14, 22)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cx = w // 2
        self.put_text(img, "PHASE 3  MOTOR TASK ANALYZER", (cx-320, 100),
                      scale=1.1, color=COLORS["accent"], thickness=3)
        self.put_text(img, "Precision Neuromotor Assessment System", (cx-230, 140),
                      scale=0.7, color=COLORS["muted"])

        self.draw_panel(img, cx-340, 170, cx+340, 470, alpha=0.82)

        lines = [
            ("5 Tasks Will Be Presented",   COLORS["text"],   0.75),
            ("Voice prompt guides each task", COLORS["muted"],  0.6),
            ("",                             None,             0),
            ("TASK 1 — Raise Both Hands",   COLORS["accent"], 0.65),
            ("TASK 2 — Touch Your Head",    COLORS["accent"], 0.65),
            ("TASK 3 — Clap Hands",         COLORS["accent"], 0.65),
            ("TASK 4 — Tap Screen Button",  COLORS["accent"], 0.65),
            ("TASK 5 — Say Your Name",      COLORS["accent"], 0.65),
            ("",                             None,             0),
            ("Each task scored 0–100",      COLORS["gold"],   0.65),
            ("Overall score aggregated",    COLORS["gold"],   0.65),
        ]
        for i, (txt, col, sc) in enumerate(lines):
            if txt:
                self.put_text(img, txt, (cx-300, 205 + i*28), scale=sc, color=col)

        t = time.time()
        blink = int(t * 2) % 2
        if blink:
            self.put_text(img, ">> PRESS [SPACE] TO BEGIN <<", (cx-185, 510),
                          scale=0.85, color=COLORS["gold"], thickness=2)

        if not TTS_AVAILABLE:
            self.put_text(img, "! Voice: pip install pyttsx3", (20, h-60),
                          scale=0.5, color=COLORS["warn"])
        if not SR_AVAILABLE:
            self.put_text(img, "! Speech Rec: pip install SpeechRecognition pyaudio",
                          (20, h-40), scale=0.5, color=COLORS["warn"])

    def draw_task_hud(self, img, task_id, det_result, elapsed, timeout):
        h, w = img.shape[:2]
        task_name = TASKS[task_id]
        instruction = TASK_INSTRUCTIONS[task_id]

        # ── Top bar ──
        self.draw_panel(img, 0, 0, w, 90, alpha=0.85)
        self.put_text(img, f"TASK {task_id}/5  ·  {task_name.upper()}", (20, 38),
                      scale=0.9, color=COLORS["accent"], thickness=2)
        self.put_text(img, instruction, (20, 68), scale=0.58, color=COLORS["text"])

        # Timer
        remaining = max(0, timeout - elapsed)
        timer_color = COLORS["danger"] if remaining < 3 else COLORS["warn"] if remaining < 6 else COLORS["muted"]
        self.put_text(img, f"{remaining:.1f}s", (w-120, 55),
                      scale=1.1, color=timer_color, thickness=2)
        self.draw_progress_bar(img, w-180, 68, 160, 10,
                               remaining/timeout, color=timer_color)

        # ── Right panel ──
        px, py = w-250, 100
        self.draw_panel(img, px-10, py, w-10, py+280, alpha=0.82)

        if det_result:
            conf  = det_result.get("confidence", 0)
            det   = det_result.get("detected", False)

            # Big confidence ring
            ring_cx, ring_cy = px + 105, py + 70
            ring_r = 55
            # Background circle
            cv2.circle(img, (ring_cx, ring_cy), ring_r, (40,40,60), -1)
            cv2.circle(img, (ring_cx, ring_cy), ring_r, COLORS["muted"], 2)
            # Arc fill
            angle = int(360 * conf)
            for i in range(0, angle, 3):
                rad = math.radians(i - 90)
                x_a = int(ring_cx + ring_r * math.cos(rad))
                y_a = int(ring_cy + ring_r * math.sin(rad))
                cv2.circle(img, (x_a, y_a), 3, self.color_for_conf(conf), -1)

            # Confidence text inside
            self.put_text(img, f"{conf:.0%}", (ring_cx-28, ring_cy+8),
                          scale=0.85, color=self.color_for_conf(conf), thickness=2)
            self.put_text(img, "CONF", (ring_cx-22, ring_cy+28),
                          scale=0.4, color=COLORS["muted"])

            # Status badge
            if det:
                badge_c = COLORS["success"]
                badge_t = "DETECTED"
            else:
                badge_c = COLORS["danger"]
                badge_t = "NOT YET"
            cv2.rectangle(img, (px, py+140), (px+210, py+170), badge_c, -1)
            self.put_text(img, badge_t, (px+30, py+162), scale=0.65,
                          color=(10,10,10), thickness=2)

            # Sub-scores
            sub = det_result.get("sub_scores", {})
            if sub:
                self.put_text(img, "SUB-SCORES:", (px, py+185),
                              scale=0.5, color=COLORS["muted"])
                self.sub_score_bars(img, sub, px, py+195, bar_w=100)

        # ── Smoothed confidence trend bar ──
        sm_conf = self.conf_window.mean()
        self.put_text(img, "SMOOTHED:", (20, h-90), scale=0.5, color=COLORS["muted"])
        self.draw_progress_bar(img, 110, h-104, 300, 18, sm_conf,
                               color=self.color_for_conf(sm_conf))
        self.put_text(img, f"{sm_conf:.0%}", (420, h-90), scale=0.55, color=COLORS["text"])

        trend = self.conf_window.trend()
        trend_sym = "▲" if trend > 0.02 else ("▼" if trend < -0.02 else "─")
        trend_col = COLORS["success"] if trend > 0.02 else (COLORS["danger"] if trend < -0.02 else COLORS["muted"])
        self.put_text(img, trend_sym, (465, h-88), scale=0.7, color=trend_col)

        # ── Hold indicator ──
        if self.holding:
            elapsed_hold = time.time() - self.hold_start
            hold_pct = min(1.0, elapsed_hold / self.scorer.HOLD_TARGET)
            self.put_text(img, "HOLD:", (20, h-60), scale=0.5, color=COLORS["muted"])
            self.draw_progress_bar(img, 80, h-74, 200, 18, hold_pct,
                                   color=COLORS["success"])
            self.put_text(img, f"{elapsed_hold:.1f}s", (290, h-60),
                          scale=0.5, color=COLORS["text"])

        # ── Task number bubbles ──
        for t_id in range(1, 6):
            cx_ = 30 + (t_id-1) * 42
            cy_ = h - 25
            done = t_id in self.results and self.results[t_id].completed
            cur  = t_id == task_id
            col = COLORS["success"] if done else (COLORS["accent"] if cur else COLORS["muted"])
            cv2.circle(img, (cx_, cy_), 15, col, -1 if (done or cur) else 2)
            self.put_text(img, str(t_id), (cx_-5, cy_+5), scale=0.5,
                          color=(10,10,10) if (done or cur) else COLORS["muted"])

    def draw_tap_button(self, img):
        x1, y1, x2, y2 = self.button_rect
        # Pulsing glow
        t = time.time()
        pulse = 0.6 + 0.4 * math.sin(t * 4)
        r = int(pulse * 255)
        cv2.rectangle(img, (x1-4, y1-4), (x2+4, y2+4), (0, 0, r), 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), -1)
        self.put_text(img, "TAP HERE", (x1+10, (y1+y2)//2+8),
                      scale=0.75, color=(255,255,255), thickness=2)

    def draw_voice_task(self, img):
        h, w = img.shape[:2]
        self.draw_panel(img, w//2-250, h//2-80, w//2+250, h//2+100, alpha=0.9)

        if self.voice_listening:
            t   = time.time()
            dot = "●" * (1 + int(t * 2) % 3)
            self.put_text(img, f"LISTENING {dot}", (w//2-100, h//2-30),
                          scale=0.9, color=COLORS["danger"], thickness=2)
            self.put_text(img, "Say your name clearly...", (w//2-140, h//2+10),
                          scale=0.65, color=COLORS["text"])
        elif self.voice_result:
            self.put_text(img, "Recognized:", (w//2-100, h//2-40),
                          scale=0.65, color=COLORS["muted"])
            self.put_text(img, f'"{self.voice_result}"', (w//2-120, h//2+5),
                          scale=0.85, color=COLORS["success"], thickness=2)
        else:
            self.put_text(img, "Press [V] to speak your name", (w//2-185, h//2-20),
                          scale=0.7, color=COLORS["text"])
            if not SR_AVAILABLE:
                self.put_text(img, "SpeechRecognition not installed!", (w//2-185, h//2+20),
                              scale=0.6, color=COLORS["danger"])
                self.put_text(img, "pip install SpeechRecognition pyaudio", (w//2-185, h//2+48),
                              scale=0.5, color=COLORS["muted"])

    def draw_review(self, img):
        h, w = img.shape[:2]
        overlay = img.copy()
        overlay[:] = (8, 10, 20)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        self.put_text(img, "SESSION RESULTS", (w//2-180, 60),
                      scale=1.2, color=COLORS["gold"], thickness=3)

        scores = [r.score for r in self.results.values()]
        overall = sum(scores) / max(len(scores), 1)

        # Overall score ring
        oc = int(w * 0.75)
        oy = 220
        ring_r = 80
        cv2.circle(img, (oc, oy), ring_r+6, (30,30,50), -1)
        cv2.circle(img, (oc, oy), ring_r+6, COLORS["muted"], 2)
        angle = int(360 * overall / 100)
        ring_col = COLORS["success"] if overall >= 70 else COLORS["warn"] if overall >= 40 else COLORS["danger"]
        for i in range(0, angle, 2):
            rad = math.radians(i - 90)
            xp  = int(oc + ring_r * math.cos(rad))
            yp  = int(oy + ring_r * math.sin(rad))
            cv2.circle(img, (xp, yp), 5, ring_col, -1)
        self.put_text(img, f"{overall:.0f}", (oc-28, oy+12),
                      scale=1.5, color=ring_col, thickness=3)
        self.put_text(img, "OVERALL", (oc-42, oy+45),
                      scale=0.55, color=COLORS["muted"])

        # Grade
        grade = "A" if overall>=90 else "B" if overall>=80 else "C" if overall>=70 else "D" if overall>=55 else "F"
        self.put_text(img, grade, (oc-18, oy-20), scale=1.8, color=ring_col, thickness=4)

        # Per-task breakdown
        self.draw_panel(img, 40, 100, w//2+60, 500, alpha=0.82)
        self.put_text(img, "TASK BREAKDOWN", (60, 135), scale=0.75, color=COLORS["accent"])

        for i, (t_id, res) in enumerate(sorted(self.results.items())):
            yy = 165 + i * 62
            done_col = COLORS["success"] if res.completed else COLORS["warn"]
            cv2.circle(img, (75, yy+10), 14, done_col, -1)
            self.put_text(img, str(t_id), (70, yy+15), scale=0.55,
                          color=(10,10,10), thickness=2)
            self.put_text(img, TASKS[t_id], (100, yy+15),
                          scale=0.6, color=COLORS["text"])

            # Score bar
            self.draw_progress_bar(img, 100, yy+24, 280, 16, res.score/100,
                                   color=self.color_for_conf(res.score/100))
            self.put_text(img, f"{res.score:.0f}/100", (395, yy+37),
                          scale=0.6, color=ring_col if res.score >= 70 else COLORS["warn"])

            # Reaction time
            self.put_text(img, f"RT:{res.reaction_time:.1f}s  Hold:{res.hold_duration:.1f}s",
                          (100, yy+48), scale=0.42, color=COLORS["muted"])

        # Controls
        self.put_text(img, "[R] Retry Session   [Q] Quit", (w//2-170, h-40),
                      scale=0.7, color=COLORS["muted"])

    def draw_fps(self, img):
        now = time.time()
        self.fps_counter.append(now - self.last_frame_t)
        self.last_frame_t = now
        avg_dt = sum(self.fps_counter) / max(len(self.fps_counter), 1)
        fps = 1.0 / avg_dt if avg_dt > 0 else 0
        self.put_text(img, f"FPS:{fps:.0f}", (self.frame_w-90, self.frame_h-10),
                      scale=0.45, color=COLORS["muted"], thickness=1)

    # ── Task lifecycle ────────────────────────────────────────────────────────

    def start_task(self, task_id: int):
        self.current_task = task_id
        self.phase = "task"
        self.task_start_time = time.time()
        self.conf_window = AnalyticsWindow(size=20)
        self.hold_start   = 0.0
        self.holding      = False
        self.detected_once= False
        self.voice_result = None
        self.voice_listening = False

        prompt = TASK_VOICE_PROMPTS.get(task_id, f"Task {task_id}")
        print(f"\n[Task {task_id}] {TASKS[task_id]}")
        print(f"[Voice] {prompt}")
        self.voice.speak(prompt)

        if task_id not in self.results:
            self.results[task_id] = TaskResult(
                task_id=task_id,
                task_name=TASKS[task_id],
                timestamp=time.time(),
            )

    def finish_task(self, task_id: int, completed: bool):
        res = self.results[task_id]
        res.completed = completed
        elapsed = time.time() - self.task_start_time
        res.reaction_time = min(elapsed, self.scorer.REACTION_MAX)
        if self.holding:
            res.hold_duration = time.time() - self.hold_start
        res.score = self.scorer.compute(res)

        msg = f"Task {task_id} complete. Score: {res.score:.0f}" if completed else \
              f"Task {task_id} timed out. Partial score: {res.score:.0f}"
        print(f"[Result] {msg}")
        self.voice.speak(msg)

        if task_id < 5:
            time.sleep(0.5)
            self.start_task(task_id + 1)
        else:
            self.phase = "review"
            scores = [r.score for r in self.results.values()]
            overall = sum(scores) / max(len(scores), 1)
            self.voice.speak(f"Session complete. Your overall score is {overall:.0f} out of 100.")

    def trigger_voice_listen(self):
        if self.voice_listening:
            return
        self.voice_listening = True
        self.voice.speak("Say your name now.")

        def listen_worker():
            result = self.voice.listen(timeout=5, phrase_limit=4)
            self.voice_result = result if result else "[Not recognized]"
            self.voice_listening = False
            res = self.results.get(5)
            if res:
                res.attempts += 1
                if result:
                    res.best_confidence = 1.0
                    res.completed = True
                    print(f"[Voice] Recognized: {result}")
                    self.finish_task(5, True)
                else:
                    res.best_confidence = max(res.best_confidence, 0.3)

        threading.Thread(target=listen_worker, daemon=True).start()

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("\nESP32-CAM ready. Press [SPACE] to start.")
        while True:
            ret, frame = self.cap.read()          # ← reads from ESP32-CAM /stream
            if not ret:
                continue                           # wait for next frame, don't break

            # NOTE: no cv2.flip — ESP32-CAM is not mirrored like a webcam.
            # Uncomment below if your setup appears mirrored:
            # frame = cv2.flip(frame, 1)

            if self.phase == "intro":
                self.draw_intro(frame)

            elif self.phase == "task":
                task_id = self.current_task
                elapsed = time.time() - self.task_start_time
                res     = self.results.get(task_id)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_r, hands_r = self.analyzer.process(rgb)

                # Draw pose skeleton
                if pose_r.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame, pose_r.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,220,180), thickness=2, circle_radius=3),
                        connection_drawing_spec=mp_draw.DrawingSpec(color=(100,200,255), thickness=2),
                    )

                kp = self.analyzer.extract_keypoints(pose_r)
                det_result = None

                if kp and task_id != 5:
                    if task_id == 1:
                        det_result = self.analyzer.score_raise_hands(kp)
                    elif task_id == 2:
                        det_result = self.analyzer.score_touch_head(kp)
                    elif task_id == 3:
                        det_result = self.analyzer.score_clap(kp)
                    elif task_id == 4:
                        self.draw_tap_button(frame)
                        det_result = self.analyzer.score_tap_button(
                            kp, self.button_rect, (self.frame_w, self.frame_h))

                        # Visualize finger tip
                        tip = det_result.get("tip_pos")
                        if tip:
                            tip_px = (int(tip[0]*self.frame_w), int(tip[1]*self.frame_h))
                            cv2.drawMarker(frame, tip_px, COLORS["gold"], cv2.MARKER_CROSS, 20, 2)

                # Task 5: voice
                elif task_id == 5:
                    self.draw_voice_task(frame)
                    det_result = {
                        "confidence": 1.0 if self.voice_result and "[" not in self.voice_result else 0.0,
                        "detected":   bool(self.voice_result and "[" not in self.voice_result),
                        "sub_scores": {"recognized": 1.0 if (self.voice_result and "[" not in self.voice_result) else 0.0},
                    }

                # Update tracking
                if det_result and res:
                    conf = det_result["confidence"]
                    det  = det_result["detected"]
                    self.conf_window.push(conf)

                    if conf > res.best_confidence:
                        res.best_confidence = conf

                    if det and not self.detected_once:
                        self.detected_once = True
                        res.attempts += 1
                        res.reaction_time = elapsed

                    # Hold tracking
                    if det:
                        if not self.holding:
                            self.holding   = True
                            self.hold_start= time.time()
                        hold_dur = time.time() - self.hold_start
                        res.hold_duration = hold_dur
                        if hold_dur >= self.scorer.HOLD_TARGET:
                            res.completed = True
                            self.finish_task(task_id, True)
                            continue
                    else:
                        if self.holding:
                            self.holding = False

                # Timeout
                if elapsed >= self.task_timeout and res and not res.completed:
                    self.finish_task(task_id, False)
                    continue

                if self.phase == "task":
                    self.draw_task_hud(frame, task_id, det_result, elapsed, self.task_timeout)

            elif self.phase == "review":
                self.draw_review(frame)

            self.draw_fps(frame)
            cv2.imshow("Phase 3 Motor Task Analyzer", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') and self.phase == "intro":
                self.start_task(1)
            elif key == ord('v') and self.phase == "task" and self.current_task == 5:
                self.trigger_voice_listen()
            elif key == ord('r') and self.phase == "review":
                self.results.clear()
                self.phase = "intro"
            elif key == ord('s') and self.phase == "task":
                # Manual skip (debug)
                self.finish_task(self.current_task, False)

        self.cap.release()
        cv2.destroyAllWindows()
        self.analyzer.pose.close()
        self.analyzer.hands.close()
        print("\nSession ended.")
        if self.results:
            scores = [r.score for r in self.results.values()]
            print(f"Overall Score: {sum(scores)/len(scores):.1f}/100")
            for t_id, r in sorted(self.results.items()):
                status = "✓" if r.completed else "✗"
                print(f"  Task {t_id} [{status}] {r.task_name}: {r.score:.1f}  "
                      f"(RT:{r.reaction_time:.1f}s, Hold:{r.hold_duration:.1f}s)")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MotorTaskApp()
    app.run()