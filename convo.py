"""
Phase 3 — Conversation Monitor
================================
Attention & Communication Assessment via Webcam + Microphone
Designed with care for autistic children.

Features:
  • Name-call attention detection (eye contact + head orientation)
  • Greeting response with patience-aware retry logic
  • 5 structured questions with voice + visual response analysis
  • Camera: gaze direction, face presence, head turn, facial engagement
  • Voice: response latency, volume, clarity, word detection
  • Positive, non-judgmental feedback at every step
  • Scores: attention, response speed, verbal clarity, engagement

Requirements:
    pip install mediapipe opencv-python numpy pyttsx3 SpeechRecognition pyaudio

Run:
    python conversation_monitor.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import deque

# ─── Optional voice imports ───────────────────────────────────────────────────
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

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose      = mp.solutions.pose
mp_draw      = mp.solutions.drawing_utils

# ─── Conversation Questions ───────────────────────────────────────────────────
CHILD_NAME = "friend"   # ← personalise: replace with child's name

GREETING_SCRIPT = f"Hello, {CHILD_NAME}! Can you say hello to me?"

QUESTIONS = {
    1: {
        "ask":    f"What is your name?",
        "prompt": "Tell me your name.",
        "hint":   "You can say your name.",
        "expect": ["name", "i am", "my name", "im"],
    },
    2: {
        "ask":    "How are you feeling today?",
        "prompt": "Are you happy? Or tired?",
        "hint":   "You can say happy, good, or tired.",
        "expect": ["happy", "good", "fine", "okay", "tired", "sad", "great"],
    },
    3: {
        "ask":    "Can you tell me one thing you like?",
        "prompt": "What do you like? A toy? A food? A colour?",
        "hint":   "Say one thing you really like.",
        "expect": ["like", "love", "favourite", "play", "eat", "color", "colour"],
    },
    4: {
        "ask":    "Can you look at me and wave hello?",
        "prompt": "Look at the camera and wave your hand.",
        "hint":   "Wave your hand up in the air.",
        "expect": ["wave", "hi", "hello", "hey"],
        "visual_task": "wave",  # camera checks for wrist raised
    },
    5: {
        "ask":    "You did so well! Can you give me a big smile?",
        "prompt": "Show me your best smile!",
        "hint":   "Can you smile for me?",
        "expect": ["yes", "smile", "okay", "sure", "ha", "haha"],
        "visual_task": "smile",
    },
}

# Timing (tuned for autistic children — generous patience)
GREETING_PATIENCE    = 8.0   # seconds to wait for greeting response
QUESTION_PATIENCE    = 12.0  # seconds per question before gentle hint
HINT_PATIENCE        = 10.0  # seconds after hint before moving on
MAX_RETRY_GREET      = 2     # re-greet this many times before proceeding
HOLD_ATTENTION_SEC   = 1.5   # seconds of face presence = "attending"

POSITIVE_AFFIRMATIONS = [
    "Great job!",
    "Well done!",
    "That's wonderful!",
    "You're doing amazing!",
    "Fantastic!",
    "I loved that!",
    "Super!",
    "Brilliant!",
]

GENTLE_ENCOURAGEMENTS = [
    "Take your time, no rush.",
    "It's okay, you're doing great.",
    "Whenever you're ready.",
    "That's okay! Let's try together.",
]

COLORS = {
    "accent":  (0, 220, 180),
    "warm":    (60, 160, 255),
    "success": (60, 210, 100),
    "warn":    (40, 170, 255),
    "danger":  (80, 80, 255),
    "text":    (235, 235, 245),
    "muted":   (140, 140, 160),
    "gold":    (30, 210, 255),
    "panel":   (20, 24, 38),
    "bg":      (10, 12, 20),
    "blush":   (140, 100, 220),
}

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ConvResult:
    q_id: int
    question: str
    score: float            = 0.0
    attention_score: float  = 0.0   # face present + looking
    response_speed: float   = 0.0   # 0–100, faster = higher
    verbal_score: float     = 0.0   # voice detected + keyword match
    visual_score: float     = 0.0   # wave/smile detected
    response_latency: float = 0.0   # seconds to first response
    responded: bool         = False
    voice_text: str         = ""
    needed_hint: bool       = False
    needed_retry: bool      = False

@dataclass
class SmoothVal:
    size: int = 20
    _vals: deque = field(default_factory=lambda: deque(maxlen=20))

    def push(self, v): self._vals.append(float(v))
    def get(self) -> float:
        return float(np.mean(self._vals)) if self._vals else 0.0
    def trend(self) -> float:
        if len(self._vals) < 6: return 0.0
        half = len(self._vals) // 2
        a = list(self._vals)
        return np.mean(a[half:]) - np.mean(a[:half])


# ─── Voice Engine ─────────────────────────────────────────────────────────────

class VoiceEngine:
    """TTS + mic-based speech recognition with noise filtering."""

    def __init__(self):
        self.tts = None
        self.rec = None
        self.mic = None
        self._speaking = False
        self._init_tts()
        self._init_sr()

    def _init_tts(self):
        if not TTS_AVAILABLE:
            print("[Voice] pyttsx3 not installed — no audio prompts.")
            return
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 145)   # slightly slower for clarity
            self.tts.setProperty('volume', 0.95)
            # Prefer female voice if available (friendlier tone)
            voices = self.tts.getProperty('voices')
            for v in voices:
                if 'female' in v.name.lower() or 'zira' in v.name.lower() \
                        or 'samantha' in v.name.lower():
                    self.tts.setProperty('voice', v.id)
                    break
            print("[Voice] TTS ready.")
        except Exception as e:
            print(f"[Voice] TTS error: {e}")

    def _init_sr(self):
        if not SR_AVAILABLE:
            print("[Voice] SpeechRecognition not installed — voice tasks disabled.")
            return
        try:
            self.rec = sr.Recognizer()
            # Noise filtering settings
            self.rec.energy_threshold      = 350    # lower = more sensitive
            self.rec.dynamic_energy_threshold = True
            self.rec.pause_threshold       = 1.0    # wait 1s of silence before done
            self.rec.phrase_threshold      = 0.3
            self.rec.non_speaking_duration = 0.5
            self.mic = sr.Microphone()
            # Calibrate ambient noise once at startup
            with self.mic as src:
                print("[Voice] Calibrating mic noise... (2s)")
                self.rec.adjust_for_ambient_noise(src, duration=2)
            print(f"[Voice] Mic ready. Energy threshold: {self.rec.energy_threshold:.0f}")
        except Exception as e:
            print(f"[Voice] Mic error: {e}")

    def speak(self, text: str, blocking=False):
        if not self.tts:
            print(f"[TTS] {text}")
            return
        self._speaking = True
        if blocking:
            self.tts.say(text)
            self.tts.runAndWait()
            self._speaking = False
        else:
            def _worker():
                try:
                    self.tts.say(text)
                    self.tts.runAndWait()
                except: pass
                finally:
                    self._speaking = False
            threading.Thread(target=_worker, daemon=True).start()

    def listen(self, timeout=8, phrase_limit=6) -> dict:
        """
        Listen via mic. Returns:
          { text, volume, duration, detected }
        Noise-robust: uses calibrated threshold + phrase detection.
        """
        result = {"text": "", "volume": 0.0, "duration": 0.0, "detected": False}
        if not (self.rec and self.mic):
            return result
        try:
            with self.mic as source:
                # Short re-calibration to handle noise drift
                self.rec.adjust_for_ambient_noise(source, duration=0.3)
                t0 = time.time()
                audio = self.rec.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )
                result["duration"] = time.time() - t0

            # Volume estimate from audio energy
            raw = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
            result["volume"] = float(np.sqrt(np.mean(raw**2))) / 32768.0

            text = self.rec.recognize_google(audio, language="en-US")
            result["text"]     = text.lower().strip()
            result["detected"] = bool(text.strip())
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            result["detected"] = False
        except Exception as e:
            print(f"[Voice] Listen error: {e}")
        return result


# ─── Face & Attention Analyzer ───────────────────────────────────────────────

class AttentionAnalyzer:
    """
    Uses MediaPipe FaceMesh + Pose to measure:
    - Face presence / visibility
    - Gaze direction (iris position relative to eye corners)
    - Head yaw (facing camera vs turned away)
    - Wave detection (wrist raised above shoulder)
    """

    # FaceMesh landmark indices
    LEFT_EYE_INNER  = 133
    LEFT_EYE_OUTER  = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_IRIS_CENTER  = 468
    RIGHT_IRIS_CENTER = 473
    NOSE_TIP   = 1
    LEFT_CHEEK = 234
    RIGHT_CHEEK= 454
    MOUTH_LEFT = 61
    MOUTH_RIGHT= 291
    UPPER_LIP  = 13
    LOWER_LIP  = 14

    def __init__(self):
        self.face = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,       # enables iris landmarks (468,473)
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, rgb):
        face_r = self.face.process(rgb)
        pose_r = self.pose.process(rgb)
        return face_r, pose_r

    def _lm(self, face_result, idx) -> Optional[np.ndarray]:
        if not face_result.multi_face_landmarks:
            return None
        lm = face_result.multi_face_landmarks[0].landmark[idx]
        return np.array([lm.x, lm.y, lm.z])

    def analyze_attention(self, face_r, pose_r) -> dict:
        """
        Returns dict with:
          face_present, gaze_score, head_yaw_score, attention_score,
          wave_detected, smile_score
        """
        out = {
            "face_present":    False,
            "gaze_score":      0.0,
            "head_yaw_score":  0.0,
            "attention_score": 0.0,
            "wave_detected":   False,
            "smile_score":     0.0,
        }

        # ── Face presence ──
        if not face_r.multi_face_landmarks:
            return out
        out["face_present"] = True

        # ── Head yaw (left cheek vs right cheek x-distance to nose) ──
        nose      = self._lm(face_r, self.NOSE_TIP)
        l_cheek   = self._lm(face_r, self.LEFT_CHEEK)
        r_cheek   = self._lm(face_r, self.RIGHT_CHEEK)
        if nose is not None and l_cheek is not None and r_cheek is not None:
            # Symmetric when facing camera: |nose.x - l_cheek.x| ≈ |nose.x - r_cheek.x|
            d_left  = abs(nose[0] - l_cheek[0])
            d_right = abs(nose[0] - r_cheek[0])
            total   = d_left + d_right + 1e-6
            ratio   = min(d_left, d_right) / total  # 0.5 = perfectly centered
            # Score: higher when ratio closer to 0.5 (facing forward)
            yaw_score = float(np.clip(ratio / 0.5, 0, 1))
            out["head_yaw_score"] = yaw_score

        # ── Gaze (iris position within eye) ──
        try:
            li = self._lm(face_r, self.LEFT_IRIS_CENTER)
            ri = self._lm(face_r, self.RIGHT_IRIS_CENTER)
            le_in  = self._lm(face_r, self.LEFT_EYE_INNER)
            le_out = self._lm(face_r, self.LEFT_EYE_OUTER)
            re_in  = self._lm(face_r, self.RIGHT_EYE_INNER)
            re_out = self._lm(face_r, self.RIGHT_EYE_OUTER)

            if all(x is not None for x in [li, ri, le_in, le_out, re_in, re_out]):
                # Iris position ratio within eye width (0=outer edge, 1=inner edge)
                l_eye_w = abs(le_in[0] - le_out[0]) + 1e-6
                r_eye_w = abs(re_in[0] - re_out[0]) + 1e-6
                l_ratio = (li[0] - le_out[0]) / l_eye_w
                r_ratio = (ri[0] - re_out[0]) / r_eye_w
                # 0.5 = looking straight ahead
                l_gaze = 1.0 - 2.0 * abs(l_ratio - 0.5)
                r_gaze = 1.0 - 2.0 * abs(r_ratio - 0.5)
                gaze   = float(np.clip((l_gaze + r_gaze) / 2, 0, 1))
                out["gaze_score"] = gaze
        except Exception:
            out["gaze_score"] = out["head_yaw_score"]  # fallback

        # ── Smile (mouth open ratio) ──
        ul = self._lm(face_r, self.UPPER_LIP)
        ll = self._lm(face_r, self.LOWER_LIP)
        ml = self._lm(face_r, self.MOUTH_LEFT)
        mr = self._lm(face_r, self.MOUTH_RIGHT)
        if all(x is not None for x in [ul, ll, ml, mr]):
            mouth_h = abs(ul[1] - ll[1])
            mouth_w = abs(ml[0] - mr[0]) + 1e-6
            smile   = float(np.clip(mouth_h / mouth_w * 4, 0, 1))
            out["smile_score"] = smile

        # ── Wave detection via pose ──
        if pose_r.pose_landmarks:
            lm = pose_r.pose_landmarks.landmark
            l_shoulder_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            r_shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            l_wrist_y    = lm[mp_pose.PoseLandmark.LEFT_WRIST].y
            r_wrist_y    = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
            # Wrist above shoulder = waving
            out["wave_detected"] = (l_wrist_y < l_shoulder_y - 0.05 or
                                     r_wrist_y < r_shoulder_y - 0.05)

        # ── Overall attention ──
        out["attention_score"] = float(np.clip(
            out["head_yaw_score"] * 0.5 + out["gaze_score"] * 0.5, 0, 1
        ))
        return out


# ─── Score Calculator ─────────────────────────────────────────────────────────

class ConvScorer:
    MAX_LATENCY = 10.0  # seconds for 0-speed score

    def compute(self, res: ConvResult) -> float:
        """Weighted composite score 0–100."""
        # Attention component (30%)
        att = res.attention_score * 30

        # Response speed (20%) — generous, non-punishing curve
        if res.response_latency > 0:
            speed = max(0, 1.0 - (res.response_latency / self.MAX_LATENCY) ** 0.6)
        else:
            speed = 0.0
        sp_pts = speed * 20

        # Verbal (30%)
        vb_pts = res.verbal_score * 30

        # Visual (20%)
        vis_pts = res.visual_score * 20

        # Hint/retry gentle deductions (max -10)
        deduct = 0
        if res.needed_hint:  deduct += 3
        if res.needed_retry: deduct += 5

        total = att + sp_pts + vb_pts + vis_pts - deduct
        return float(np.clip(round(total, 1), 0, 100))


# ─── Main App ─────────────────────────────────────────────────────────────────

class ConversationMonitor:

    def __init__(self):
        self.voice    = VoiceEngine()
        self.analyzer = AttentionAnalyzer()
        self.scorer   = ConvScorer()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam not found.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # State machine
        self.phase      = "intro"    # intro | greeting | question | result
        self.q_id       = 0
        self.results: Dict[int, ConvResult] = {}

        # Timing & patience
        self.phase_start    = 0.0
        self.greet_retries  = 0
        self.hint_given     = False
        self.hint_time      = 0.0
        self.listening      = False
        self.listen_thread  = None
        self.listen_result: Optional[dict] = None
        self.voice_done     = False

        # Attention smoothing
        self.att_smooth    = SmoothVal(size=25)
        self.gaze_smooth   = SmoothVal(size=25)
        self.yaw_smooth    = SmoothVal(size=25)
        self.smile_smooth  = SmoothVal(size=15)

        # Current frame attention snapshot
        self.cur_att: dict = {}

        # Affirmation index
        self._aff_idx = 0

        # FPS
        self._fps_dq  = deque(maxlen=30)
        self._last_ft = time.time()

        # Conversation log for display
        self.conv_log: List[tuple] = []  # (speaker, text, color)

        print(f"\n=== Conversation Monitor ===")
        print(f"Child name: '{CHILD_NAME}'")
        print(f"TTS: {'✓' if TTS_AVAILABLE else '✗'}   SR: {'✓' if SR_AVAILABLE else '✗'}")
        print("Press [SPACE] to begin | [Q] to quit\n")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _affirm(self) -> str:
        t = POSITIVE_AFFIRMATIONS[self._aff_idx % len(POSITIVE_AFFIRMATIONS)]
        self._aff_idx += 1
        return t

    def _encourage(self) -> str:
        import random
        return random.choice(GENTLE_ENCOURAGEMENTS)

    def say(self, text: str, speaker="System", color=None, blocking=False):
        color = color or COLORS["warm"]
        self.conv_log.append((speaker, text, color))
        if len(self.conv_log) > 8:
            self.conv_log.pop(0)
        print(f"[{speaker}] {text}")
        self.voice.speak(text, blocking=blocking)

    def log_child(self, text: str):
        self.conv_log.append(("Child", text, COLORS["success"]))
        if len(self.conv_log) > 8:
            self.conv_log.pop(0)
        print(f"[Child] {text}")

    # ─── Drawing ─────────────────────────────────────────────────────────────

    def put_text(self, img, text, pos, scale=0.65, color=None, thickness=2):
        color = color or COLORS["text"]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, pos, font, scale, (0,0,0), thickness+2)
        cv2.putText(img, text, pos, font, scale, color, thickness)

    def panel(self, img, x1, y1, x2, y2, alpha=0.78):
        h, w = img.shape[:2]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)
        if x2<=x1 or y2<=y1: return
        sub = img[y1:y2, x1:x2]
        bg  = np.full_like(sub, COLORS["panel"])
        cv2.addWeighted(bg, alpha, sub, 1-alpha, 0, sub)
        img[y1:y2, x1:x2] = sub
        cv2.rectangle(img, (x1,y1), (x2,y2), COLORS["accent"], 1)

    def bar(self, img, x, y, w, h, val, col=None, bg=(35,35,55)):
        col = col or COLORS["accent"]
        cv2.rectangle(img, (x,y), (x+w,y+h), bg, -1)
        fill = int(w * np.clip(val,0,1))
        if fill>0: cv2.rectangle(img, (x,y), (x+fill,y+h), col, -1)
        cv2.rectangle(img, (x,y), (x+w,y+h), COLORS["muted"], 1)

    def color_val(self, v):
        if v > 0.65: return COLORS["success"]
        if v > 0.35: return COLORS["warn"]
        return COLORS["danger"]

    def draw_face_overlay(self, img, face_r, pose_r):
        """Draw subtle face mesh + skeleton."""
        if face_r.multi_face_landmarks:
            for fl in face_r.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    img, fl,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(0, 160, 140), thickness=1, circle_radius=1),
                )
        if pose_r and pose_r.pose_landmarks:
            mp_draw.draw_landmarks(
                img, pose_r.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(0,200,160), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(80,180,200), thickness=2),
            )

    def draw_attention_hud(self, img):
        """Right-side real-time attention panel."""
        att = self.cur_att
        px, py = self.frame_w - 240, 100
        pw, ph = 230, 260
        self.panel(img, px, py, px+pw, py+ph, alpha=0.82)

        self.put_text(img, "ATTENTION", (px+10, py+24), scale=0.6,
                      color=COLORS["accent"])

        metrics = [
            ("Face",     att.get("face_present", False)),
            ("Looking",  self.gaze_smooth.get()),
            ("Facing",   self.yaw_smooth.get()),
            ("Attention",self.att_smooth.get()),
        ]

        for i, (label, val) in enumerate(metrics):
            yy = py + 50 + i * 48
            self.put_text(img, label+":", (px+10, yy+14), scale=0.48,
                          color=COLORS["muted"])
            if isinstance(val, bool):
                dot_c = COLORS["success"] if val else COLORS["danger"]
                cv2.circle(img, (px+110, yy+8), 10, dot_c, -1)
                self.put_text(img, "YES" if val else "NO", (px+126, yy+14),
                              scale=0.48, color=dot_c)
            else:
                self.bar(img, px+10, yy+20, pw-20, 14, val,
                         col=self.color_val(val))
                self.put_text(img, f"{val:.0%}", (px+pw-42, yy+32),
                              scale=0.42, color=COLORS["text"])

        # Wave / Smile indicator
        if att.get("wave_detected"):
            self.put_text(img, "WAVE ✓", (px+10, py+ph-30),
                          scale=0.6, color=COLORS["success"])
        smile = self.smile_smooth.get()
        if smile > 0.3:
            self.put_text(img, f"SMILE {smile:.0%}", (px+10, py+ph-10),
                          scale=0.55, color=COLORS["gold"])

    def draw_conv_log(self, img):
        """Bottom conversation transcript strip."""
        h = self.frame_h
        log_y = h - 180
        self.panel(img, 10, log_y, 600, h-10, alpha=0.80)
        self.put_text(img, "CONVERSATION", (20, log_y+20), scale=0.5,
                      color=COLORS["accent"])
        for i, (spk, txt, col) in enumerate(self.conv_log[-6:]):
            yy = log_y + 40 + i * 22
            # Truncate long text
            display = f"{spk}: {txt}"
            if len(display) > 58: display = display[:55] + "..."
            self.put_text(img, display, (20, yy), scale=0.45, color=col, thickness=1)

    def draw_timer_bar(self, img, elapsed, total, label=""):
        """Full-width countdown bar at very top."""
        remaining = max(0, total - elapsed)
        frac      = remaining / total
        col = COLORS["danger"] if frac < 0.25 else COLORS["warn"] if frac < 0.5 else COLORS["accent"]
        self.bar(img, 0, 0, self.frame_w, 8, frac, col=col, bg=(20,20,35))
        if label:
            self.put_text(img, f"{label} — {remaining:.0f}s", (10, 26),
                          scale=0.5, color=col)

    def draw_question_hud(self, img):
        """Top panel during question phase."""
        q  = QUESTIONS.get(self.q_id, {})
        qt = q.get("ask", "")
        self.panel(img, 0, 0, self.frame_w, 85, alpha=0.88)
        self.put_text(img, f"Q{self.q_id}/5  ·  {qt}", (18, 36),
                      scale=0.82, color=COLORS["warm"], thickness=2)
        listen_state = " 🎙 LISTENING..." if self.listening else ""
        self.put_text(img, TASK_INSTRUCTIONS_CONV.get(self.q_id, "") + listen_state,
                      (18, 66), scale=0.52, color=COLORS["text"])

    def draw_progress_dots(self, img):
        """Bottom-right question progress dots."""
        for i in range(1, 6):
            cx = self.frame_w - 30 - (5-i)*32
            cy = self.frame_h - 25
            done = i in self.results and self.results[i].responded
            cur  = i == self.q_id
            col  = COLORS["success"] if done else (COLORS["warm"] if cur else COLORS["muted"])
            cv2.circle(img, (cx, cy), 12, col, -1 if (done or cur) else 2)
            self.put_text(img, str(i), (cx-5, cy+5), scale=0.42,
                          color=(10,10,10) if (done or cur) else COLORS["muted"])

    def draw_intro(self, img):
        ov = img.copy(); ov[:] = (8,10,18)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
        cx = self.frame_w // 2
        self.put_text(img, "CONVERSATION MONITOR", (cx-250, 90),
                      scale=1.15, color=COLORS["accent"], thickness=3)
        self.put_text(img, "Attention & Communication Assessment", (cx-220, 128),
                      scale=0.7, color=COLORS["muted"])

        self.panel(img, cx-340, 155, cx+340, 450, alpha=0.85)
        lines = [
            ("Greeting → 5 Questions → Score Report", COLORS["text"], 0.68),
            ("", None, 0),
            ("Camera measures:  face, gaze, head direction, wave, smile", COLORS["muted"], 0.55),
            ("Microphone measures:  voice, words, response speed",        COLORS["muted"], 0.55),
            ("", None, 0),
            ("Gentle reminders given if no response",  COLORS["warn"],  0.6),
            ("Positive feedback throughout",           COLORS["success"],0.6),
            ("Scoring is supportive, not punishing",   COLORS["gold"],  0.6),
            ("", None, 0),
            (f"Child: {CHILD_NAME}  |  Edit CHILD_NAME at top of file", COLORS["blush"], 0.55),
        ]
        for i, (t, c, s) in enumerate(lines):
            if t: self.put_text(img, t, (cx-310, 190 + i*28), scale=s, color=c)

        blink = int(time.time()*2)%2
        if blink:
            self.put_text(img, ">> PRESS [SPACE] TO BEGIN <<", (cx-185, 478),
                          scale=0.85, color=COLORS["gold"], thickness=2)
        if not TTS_AVAILABLE:
            self.put_text(img, "pip install pyttsx3  (voice prompts)", (20, self.frame_h-50),
                          scale=0.48, color=COLORS["warn"])
        if not SR_AVAILABLE:
            self.put_text(img, "pip install SpeechRecognition pyaudio  (voice detection)",
                          (20, self.frame_h-28), scale=0.48, color=COLORS["warn"])

    def draw_result(self, img):
        ov = img.copy(); ov[:] = (6,8,16)
        cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)

        cx, cy = self.frame_w//2, self.frame_h//2
        self.put_text(img, "SESSION COMPLETE!", (cx-200, 55),
                      scale=1.1, color=COLORS["gold"], thickness=3)

        scores = [r.score for r in self.results.values()]
        overall = sum(scores)/max(len(scores),1)

        # Big ring
        rr = 85
        oc, oy = int(self.frame_w*0.76), 230
        cv2.circle(img, (oc,oy), rr+8, (25,28,50), -1)
        ring_col = self.color_val(overall/100)
        angle = int(360*overall/100)
        for i in range(0, angle, 2):
            rad = math.radians(i-90)
            cv2.circle(img, (int(oc+rr*math.cos(rad)), int(oy+rr*math.sin(rad))),
                       5, ring_col, -1)
        grade = "A"if overall>=90 else "B"if overall>=80 else "C"if overall>=70 else "D"if overall>=55 else "Keep Trying!"
        self.put_text(img, f"{overall:.0f}", (oc-30,oy+12),
                      scale=1.6, color=ring_col, thickness=3)
        self.put_text(img, "OVERALL", (oc-38,oy+46), scale=0.5, color=COLORS["muted"])
        self.put_text(img, grade, (oc-20,oy-25), scale=1.5, color=ring_col, thickness=3)

        # Per-question breakdown
        self.panel(img, 30, 95, cx+20, 510, alpha=0.85)
        self.put_text(img, "QUESTION BREAKDOWN", (50,128),
                      scale=0.7, color=COLORS["accent"])

        for idx, (qid, res) in enumerate(sorted(self.results.items())):
            yy = 158 + idx*66
            dot_c = COLORS["success"] if res.responded else COLORS["warn"]
            cv2.circle(img, (62, yy+8), 14, dot_c, -1)
            self.put_text(img, str(qid), (57,yy+13), scale=0.5,
                          color=(8,8,8), thickness=2)
            q_txt = QUESTIONS.get(qid,{}).get("ask","?")
            if len(q_txt)>42: q_txt = q_txt[:40]+"..."
            self.put_text(img, q_txt, (85,yy+14), scale=0.52, color=COLORS["text"])
            self.bar(img, 85, yy+22, 260, 14, res.score/100, col=self.color_val(res.score/100))
            self.put_text(img, f"{res.score:.0f}/100", (355,yy+34),
                          scale=0.55, color=ring_col if res.score>=60 else COLORS["warn"])

            # Sub
            sub_txt = (f"Att:{res.attention_score:.0%}  "
                       f"Speed:{res.response_speed:.0%}  "
                       f"Voice:{res.verbal_score:.0%}  "
                       f"Visual:{res.visual_score:.0%}")
            self.put_text(img, sub_txt, (85,yy+50), scale=0.38, color=COLORS["muted"])

        # Encouragement message
        enc_map = {
            (90,100): f"Outstanding! {CHILD_NAME} is a superstar!",
            (70, 90): f"Really well done, {CHILD_NAME}!",
            (50, 70): f"Good effort! Keep it up, {CHILD_NAME}.",
            (0,  50): f"Every step counts! You're doing great, {CHILD_NAME}.",
        }
        for (lo,hi), msg in enc_map.items():
            if lo <= overall < hi or (lo==90 and overall>=90):
                self.put_text(img, msg, (cx-220, self.frame_h-70),
                              scale=0.7, color=COLORS["blush"], thickness=2)
                break

        self.put_text(img, "[R] Restart   [Q] Quit", (cx-120, self.frame_h-35),
                      scale=0.6, color=COLORS["muted"])

    # ─── Listening Thread ────────────────────────────────────────────────────

    def _start_listen(self, timeout=10):
        if self.listening: return
        self.listening     = True
        self.listen_result = None
        self.voice_done    = False

        def worker():
            r = self.voice.listen(timeout=timeout, phrase_limit=7)
            self.listen_result = r
            self.listening     = False
            self.voice_done    = True

        self.listen_thread = threading.Thread(target=worker, daemon=True)
        self.listen_thread.start()

    # ─── Phase Transitions ───────────────────────────────────────────────────

    def start_greeting(self):
        self.phase        = "greeting"
        self.phase_start  = time.time()
        self.greet_retries= 0
        self.hint_given   = False
        time.sleep(0.5)
        self.say(GREETING_SCRIPT, speaker="Monitor", color=COLORS["warm"])
        time.sleep(0.8)
        self._start_listen(timeout=GREETING_PATIENCE)

    def _next_question(self):
        self.q_id += 1
        if self.q_id > 5:
            self.phase = "result"
            scores = [r.score for r in self.results.values()]
            ov = sum(scores)/max(len(scores),1)
            self.say(f"Wonderful! You did so well today, {CHILD_NAME}! "
                     f"Your score is {ov:.0f} out of 100.", speaker="Monitor")
            return
        self.phase       = "question"
        self.phase_start = time.time()
        self.hint_given  = False
        self.hint_time   = 0.0
        self.voice_done  = False

        q = QUESTIONS[self.q_id]
        ask_text = q["ask"]

        # Fresh result
        self.results[self.q_id] = ConvResult(
            q_id=self.q_id, question=ask_text)

        self.say(ask_text, speaker="Monitor", color=COLORS["warm"])
        time.sleep(0.6)
        self._start_listen(timeout=QUESTION_PATIENCE)

    def _process_voice_result(self, vr: dict, q_id: int) -> float:
        """Score 0–1 for verbal response quality."""
        if not vr or not vr.get("detected"):
            return 0.0
        text = vr.get("text","").lower()
        vol  = vr.get("volume", 0)
        q = QUESTIONS.get(q_id, {})
        keywords = q.get("expect", [])

        # Keyword match
        kw_match = any(kw in text for kw in keywords) if keywords else bool(text)
        kw_score = 0.7 if kw_match else (0.3 if text else 0.0)

        # Volume (0=silent, 1=loud)
        vol_score = float(np.clip(vol * 4, 0, 1))

        return float(np.clip(kw_score * 0.7 + vol_score * 0.3, 0, 1))

    def _attention_snapshot(self) -> dict:
        """Average of smoothed attention values."""
        return {
            "face_present":    self.cur_att.get("face_present", False),
            "attention_score": self.att_smooth.get(),
            "gaze_score":      self.gaze_smooth.get(),
            "wave_detected":   self.cur_att.get("wave_detected", False),
            "smile_score":     self.smile_smooth.get(),
        }

    # ─── Main Loop ───────────────────────────────────────────────────────────

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)

            # ── Pose + Face analysis (every frame) ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_r, pose_r = self.analyzer.process(rgb)
            att = self.analyzer.analyze_attention(face_r, pose_r)
            self.cur_att = att
            self.att_smooth.push(att["attention_score"])
            self.gaze_smooth.push(att["gaze_score"])
            self.yaw_smooth.push(att["head_yaw_score"])
            self.smile_smooth.push(att["smile_score"])
            self.draw_face_overlay(frame, face_r, pose_r)

            # ─────────────────────────────────────────
            if self.phase == "intro":
                self.draw_intro(frame)

            # ─── Greeting phase ──────────────────────
            elif self.phase == "greeting":
                elapsed = time.time() - self.phase_start
                self.draw_timer_bar(frame, elapsed, GREETING_PATIENCE, "Greeting")
                self.draw_attention_hud(frame)
                self.draw_conv_log(frame)

                self.panel(frame, 0, 0, self.frame_w, 80, alpha=0.88)
                self.put_text(frame, f"Waiting for greeting response from {CHILD_NAME}...",
                              (18,42), scale=0.8, color=COLORS["warm"], thickness=2)

                if self.voice_done and self.listen_result:
                    vr = self.listen_result
                    self.listen_result = None
                    self.voice_done    = False

                    if vr.get("detected") or att.get("face_present"):
                        txt = vr.get("text","") or "(visual response)"
                        self.log_child(txt or "hello")
                        aff = self._affirm()
                        self.say(f"{aff} Hello, {CHILD_NAME}!",
                                 speaker="Monitor", color=COLORS["success"])
                        time.sleep(1.0)
                        self._next_question()
                    else:
                        # No response
                        self.greet_retries += 1
                        if self.greet_retries <= MAX_RETRY_GREET:
                            enc = self._encourage()
                            self.say(f"{enc} {GREETING_SCRIPT}",
                                     speaker="Monitor", color=COLORS["warn"])
                            self.phase_start = time.time()
                            time.sleep(0.6)
                            self._start_listen(timeout=GREETING_PATIENCE)
                        else:
                            # Proceed gently
                            self.say(f"That's okay, {CHILD_NAME}! Let's get started.",
                                     speaker="Monitor")
                            time.sleep(1.0)
                            self._next_question()

            # ─── Question phase ───────────────────────
            elif self.phase == "question":
                elapsed = time.time() - self.phase_start
                q = QUESTIONS.get(self.q_id, {})
                patience = QUESTION_PATIENCE if not self.hint_given else HINT_PATIENCE
                self.draw_timer_bar(frame, elapsed, patience, f"Q{self.q_id}")
                self.draw_question_hud(frame)
                self.draw_attention_hud(frame)
                self.draw_conv_log(frame)
                self.draw_progress_dots(frame)

                # Listening indicator
                if self.listening:
                    t = time.time()
                    dots = "●" * (1 + int(t*2)%3)
                    self.put_text(frame, f"Listening {dots}",
                                  (self.frame_w//2-80, self.frame_h//2),
                                  scale=0.9, color=COLORS["danger"], thickness=2)

                res = self.results.get(self.q_id)

                # ── Voice result arrived ──
                if self.voice_done and self.listen_result and res:
                    vr = self.listen_result
                    self.listen_result = None
                    self.voice_done    = False

                    has_voice  = vr.get("detected", False)
                    vt = vr.get("text","")
                    if vt: self.log_child(vt)

                    # Attention snapshot at response time
                    snap = self._attention_snapshot()
                    elapsed_at_response = time.time() - self.phase_start

                    # Visual task check
                    vis_score = 0.0
                    vt_type = q.get("visual_task","")
                    if vt_type == "wave":
                        vis_score = 1.0 if snap["wave_detected"] else 0.3
                    elif vt_type == "smile":
                        vis_score = float(np.clip(snap["smile_score"]*2, 0, 1))
                    else:
                        vis_score = 0.5 if snap["face_present"] else 0.1

                    # Update result
                    res.responded         = has_voice or snap["face_present"]
                    res.voice_text        = vt
                    res.response_latency  = elapsed_at_response
                    res.response_speed    = float(np.clip(
                        1.0 - (elapsed_at_response/QUESTION_PATIENCE)**0.5, 0, 1))
                    res.attention_score   = snap["attention_score"]
                    res.verbal_score      = self._process_voice_result(vr, self.q_id)
                    res.visual_score      = vis_score
                    res.needed_hint       = self.hint_given
                    res.score             = self.scorer.compute(res)

                    if res.responded:
                        aff = self._affirm()
                        self.say(f"{aff}", speaker="Monitor", color=COLORS["success"])
                        time.sleep(1.2)
                        self._next_question()
                    else:
                        # No response — give hint if not yet given
                        if not self.hint_given:
                            self.hint_given  = True
                            self.hint_time   = time.time()
                            res.needed_hint  = True
                            hint_txt = q.get("hint", q.get("prompt",""))
                            self.say(f"{self._encourage()} {hint_txt}",
                                     speaker="Monitor", color=COLORS["warn"])
                            self.phase_start = time.time()
                            time.sleep(0.5)
                            self._start_listen(timeout=HINT_PATIENCE)
                        else:
                            # Still no response — move on warmly
                            self.say(f"That's okay! Let's try the next one.",
                                     speaker="Monitor")
                            time.sleep(1.0)
                            # Partial score with whatever attention we observed
                            res.attention_score = self.att_smooth.get()
                            res.score           = self.scorer.compute(res)
                            self._next_question()

            # ─── Result phase ────────────────────────
            elif self.phase == "result":
                self.draw_result(frame)

            # ─── FPS ────────────────────────────────
            now = time.time()
            self._fps_dq.append(now - self._last_ft)
            self._last_ft = now
            fps = 1.0/np.mean(self._fps_dq) if self._fps_dq else 0
            self.put_text(frame, f"FPS:{fps:.0f}", (self.frame_w-80, self.frame_h-8),
                          scale=0.4, color=COLORS["muted"], thickness=1)

            cv2.imshow("Conversation Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') and self.phase == "intro":
                threading.Thread(target=self.start_greeting, daemon=True).start()
            elif key == ord('r') and self.phase == "result":
                self.results.clear()
                self.q_id  = 0
                self.phase = "intro"
                self.conv_log.clear()
                self._aff_idx = 0

        self.cap.release()
        cv2.destroyAllWindows()
        self.analyzer.face.close()
        self.analyzer.pose.close()

        # Terminal summary
        print("\n=== FINAL REPORT ===")
        if self.results:
            scores = [r.score for r in self.results.values()]
            print(f"Overall: {sum(scores)/len(scores):.1f}/100")
            for qid, r in sorted(self.results.items()):
                print(f"  Q{qid}: {r.question[:50]}")
                print(f"       Score:{r.score:.0f}  "
                      f"Att:{r.attention_score:.0%}  "
                      f"Speed:{r.response_speed:.0%}  "
                      f"Voice:{r.verbal_score:.0%}  "
                      f"Visual:{r.visual_score:.0%}  "
                      f"Latency:{r.response_latency:.1f}s")


# ─── Supplementary constant ───────────────────────────────────────────────────
TASK_INSTRUCTIONS_CONV = {
    1: "Listen and say your name",
    2: "Tell me how you feel",
    3: "Say one thing you like",
    4: "Look at camera and wave",
    5: "Show a big smile!",
}


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ConversationMonitor()
    app.run()