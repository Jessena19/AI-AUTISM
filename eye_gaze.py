"""
Attention & Distraction — 2-Minute Session Scorer
══════════════════════════════════════════════════
No ML model needed. Uses the v4 rule-based scoring engine.

How it works:
  1. 5-second personal calibration (look at screen naturally)
  2. 2-minute live tracking session (auto-stops at exactly 2:00)
  3. Final result screen shows averaged scores + breakdown

Install:
    pip install opencv-python mediapipe numpy scipy

Run:
    python attention_2min_session.py
    python attention_2min_session.py --source 1   # second camera
"""

import cv2
import mediapipe as mp
import numpy as np
import time, argparse
from collections import deque
from scipy.spatial import distance as dist

# ──────────────────────────────────────────────
# SESSION CONFIG
# ──────────────────────────────────────────────
SESSION_SECONDS = 120      # 2 minutes exactly
CAL_SECONDS     = 8        # extended: 8s gives more stable personal baseline
WINDOW_NAME     = "Attention Scorer — 2 Min Session"

# ── Scoring parameters (tightened for accurate detection) ──
EAR_BLINK_RATIO = 0.72
EAR_CONSEC      = 3

# Gaze: dead zone shrunk — even small look-away starts penalty
GAZE_DEAD = 0.05    # was 0.12 — now triggers after tiny deviation
GAZE_FULL = 0.18    # was 0.30 — full penalty reached much sooner

# Head: tighter dead zones — small nod/turn counts as distracted
YAW_DEAD   =  6;  YAW_FULL  = 25   # was 15/45
PITCH_DEAD =  5;  PITCH_FULL= 18   # was 12/32

# Weights: gaze now dominates (most reliable signal)
W_HEAD  = 0.30    # was 0.45
W_GAZE  = 0.45    # was 0.30 — increased, gaze is most direct measure
W_EAR   = 0.15
W_BLINK = 0.10

# Hysteresis: harder to get ATTENTIVE, easier to flip DISTRACTED
HYST_LOW  = 50    # was 42 — must drop below 50 to be DISTRACTED
HYST_HIGH = 65    # was 58 — must reach 65 to be ATTENTIVE

# EMA: faster response to changes (was 0.06 = ~4s lag)
EMA_ALPHA = 0.12  # ~2s half-life, reacts in about 2-3 seconds

SIG_MED      = 15   # slightly smaller median window = less lag
NOFACE_LIMIT = 30   # flag no-face sooner

# ──────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

LEFT_EAR_IDX   = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_IDX  = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDX   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_IDX  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS_IDX  = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]
HEAD_2D_IDX    = [1, 152, 263, 33, 287, 57]
HEAD_3D_PTS    = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)


# ──────────────────────────────────────────────
# FEATURE EXTRACTORS
# ──────────────────────────────────────────────

def calc_ear(lms, idx, w, h):
    pts = np.array([(lms[i].x * w, lms[i].y * h) for i in idx])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3]) + 1e-6
    return (A + B) / (2.0 * C)


def calc_iris_ratio(lms, eye_idx, iris_idx, w, h):
    eye  = np.array([(lms[i].x * w, lms[i].y * h) for i in eye_idx])
    iris = np.array([(lms[i].x * w, lms[i].y * h) for i in iris_idx])
    cx, cy = np.mean(iris, axis=0)
    xmin, ymin = eye.min(axis=0)
    xmax, ymax = eye.max(axis=0)
    hr = (cx - xmin) / (xmax - xmin + 1e-6)
    vr = (cy - ymin) / (ymax - ymin + 1e-6)
    return float(np.clip(hr, 0, 1)), float(np.clip(vr, 0, 1))


def calc_head_pose(lms, w, h, cam_mat):
    pts2d = np.array([[lms[i].x * w, lms[i].y * h]
                      for i in HEAD_2D_IDX], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        HEAD_3D_PTS, pts2d, cam_mat, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = float(np.degrees(np.arctan2(-rmat[2,0],  sy)))
        yaw   = float(np.degrees(np.arctan2( rmat[1,0],  rmat[0,0])))
        roll  = float(np.degrees(np.arctan2( rmat[2,1],  rmat[2,2])))
    else:
        pitch = float(np.degrees(np.arctan2(-rmat[2,0],  sy)))
        yaw   = 0.0
        roll  = float(np.degrees(np.arctan2(-rmat[1,2],  rmat[1,1])))
    return yaw, pitch, roll


# ──────────────────────────────────────────────
# MEDIAN BUFFER
# ──────────────────────────────────────────────

class MedBuf:
    def __init__(self, n=SIG_MED):
        self._d = deque(maxlen=n)
    def push(self, v):
        self._d.append(v)
        return float(np.median(self._d))
    def clear(self):
        self._d.clear()


# ──────────────────────────────────────────────
# BASELINE
# ──────────────────────────────────────────────

class Baseline:
    def __init__(self):
        self.ready   = False
        self.ear     = 0.28
        self.blink_t = 0.20
        self.gh = 0.50; self.gv = 0.50
        self.yaw = 0.0; self.pitch = 0.0
        self._b = {k: [] for k in ('ear','gh','gv','yaw','pit')}

    def feed(self, ear, gh, gv, yaw, pit):
        for k, v in zip(('ear','gh','gv','yaw','pit'), (ear,gh,gv,yaw,pit)):
            self._b[k].append(v)

    def finalise(self):
        b = self._b
        if len(b['ear']) < 10:
            return

        def trimmed_median(arr):
            """Median of middle 60% — rejects head movements during calibration."""
            a   = np.array(arr, dtype=float)
            lo  = np.percentile(a, 20)
            hi  = np.percentile(a, 80)
            mid = a[(a >= lo) & (a <= hi)]
            return float(np.median(mid)) if len(mid) else float(np.median(a))

        self.ear     = trimmed_median(b['ear'])
        self.blink_t = self.ear * EAR_BLINK_RATIO
        # Use trimmed median for gaze — rejects any look-away during calibration
        self.gh      = trimmed_median(b['gh'])
        self.gv      = trimmed_median(b['gv'])
        self.yaw     = trimmed_median(b['yaw'])
        self.pitch   = trimmed_median(b['pit'])
        self.ready   = True
        print(f"[CAL] EAR:{self.ear:.3f}  blink<{self.blink_t:.3f}  "
              f"gaze=({self.gh:.3f},{self.gv:.3f})  "
              f"yaw:{self.yaw:.1f}°  pitch:{self.pitch:.1f}°")


# ──────────────────────────────────────────────
# SCORING ENGINE
# ──────────────────────────────────────────────

def _ramp(dev, dead, full):
    if dev <= dead: return 100.0
    if dev >= full: return   0.0
    return 100.0 * (1.0 - (dev - dead) / (full - dead))


def score_frame(ear_s, blink_rate, gh_s, gv_s, yaw_s, pit_s,
                bl: Baseline, elapsed: float):
    # EAR: start penalising after just 10% drop, reach zero at 70% of blink gap
    ear_drop  = max(0.0, bl.ear - ear_s)
    ear_dead  = max(bl.ear - bl.blink_t, 0.01) * 0.10
    ear_range = max(bl.ear - bl.blink_t, 0.01) * 0.70
    s_ear     = _ramp(ear_drop, ear_dead, ear_range)

    if elapsed < 60:
        s_blink = 85.0
    elif blink_rate < 2:
        s_blink = 50.0
    elif blink_rate <= 28:
        s_blink = 100.0
    elif blink_rate <= 50:
        s_blink = 72.0
    else:
        s_blink = 45.0

    gaze_dev = float(np.hypot(gh_s - bl.gh, gv_s - bl.gv))
    s_gaze   = _ramp(gaze_dev, GAZE_DEAD, GAZE_FULL)

    yaw_dev  = abs(yaw_s  - bl.yaw)
    pit_dev  = abs(pit_s  - bl.pitch)
    s_head   = (_ramp(yaw_dev, YAW_DEAD, YAW_FULL) +
                _ramp(pit_dev, PITCH_DEAD, PITCH_FULL)) / 2.0

    att = float(np.clip(
        W_HEAD*s_head + W_GAZE*s_gaze + W_EAR*s_ear + W_BLINK*s_blink,
        0, 100))
    return att, dict(ear=s_ear, blink=s_blink, gaze=s_gaze, head=s_head,
                     gaze_dev=gaze_dev, yaw_dev=yaw_dev, pit_dev=pit_dev)


# ──────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────

def draw_bar(frame, x, y, w, h, value, color, bg=(40,40,40)):
    cv2.rectangle(frame, (x, y), (x+w, y+h), bg, -1)
    fill = int(w * value / 100.0)
    cv2.rectangle(frame, (x, y), (x+fill, y+h), color, -1)

def txt(frame, s, x, y, col=(210,210,210), sc=0.50, bold=False):
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                sc, col, 2 if bold else 1, cv2.LINE_AA)

def color_for(score):
    """Green→Yellow→Red based on attention score."""
    r = int(np.clip(255 * (1 - score/100) * 2, 0, 255))
    g = int(np.clip(255 * (score/100) * 2,     0, 255))
    return (0, g, r)


# ──────────────────────────────────────────────
# LIVE UI
# ──────────────────────────────────────────────

def draw_live_ui(frame, att_ema, dist_ema, label, bl,
                 sub, remaining_sec, blinks, blink_rate,
                 noface_cnt, fps, frame_id):
    FH, FW = frame.shape[:2]

    # Sidebar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (340, FH), (12,12,12), -1)
    cv2.addWeighted(ov, 0.62, frame, 0.38, 0, frame)

    # Timer ring (top of sidebar)
    frac     = remaining_sec / SESSION_SECONDS
    cx, cy   = 170, 60
    r_out, r_in = 48, 34
    # Background ring
    cv2.circle(frame, (cx, cy), r_out, (45,45,45), -1)
    cv2.circle(frame, (cx, cy), r_in,  (12,12,12), -1)
    # Countdown arc
    start_ang = -90
    end_ang   = start_ang + int(360 * frac)
    t_col = (0,200,255) if frac > 0.25 else (0,100,255)
    cv2.ellipse(frame, (cx,cy), (r_out,r_out), 0,
                start_ang, end_ang, t_col, r_out - r_in)
    mins = int(remaining_sec) // 60
    secs = int(remaining_sec) % 60
    t_str = f"{mins}:{secs:02d}"
    # Centre time text
    (tw,th), _ = cv2.getTextSize(t_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, t_str,
                (cx - tw//2, cy + th//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    txt(frame, "2-MIN SESSION", 100, 120, (120,120,120), 0.42)

    # ── Attention bar ──
    a_col = color_for(att_ema)
    draw_bar(frame, 8, 135, 322, 26, att_ema, a_col)
    txt(frame, f"ATTENTION   {att_ema:.1f}%", 10, 155, a_col, 0.56, True)

    # ── Distraction bar ──
    d_col = color_for(100 - dist_ema)  # inverted: high distraction = red
    draw_bar(frame, 8, 165, 322, 26, dist_ema, d_col)
    txt(frame, f"DISTRACTION {dist_ema:.1f}%", 10, 185, d_col, 0.56, True)

    # ── Label badge ──
    bcol = (20,140,50) if label == "ATTENTIVE" else (30,30,190)
    cv2.rectangle(frame, (8,194), (330,218), bcol, -1)
    txt(frame, f"  {label}", 10, 212, (255,255,255), 0.58, True)

    # ── Sub-scores ──
    txt(frame, "Sub-scores:", 10, 240, (100,100,100), 0.42)
    txt(frame,
        f"  Head:{sub['head']:.0f}  Gaze:{sub['gaze']:.0f}  "
        f"EAR:{sub['ear']:.0f}  Blink:{sub['blink']:.0f}",
        10, 257, (130,130,130), 0.42)

    # ── Eye info ──
    txt(frame, f"Blinks: {blinks}   Rate: {blink_rate:.1f}/min",
        10, 280, (160,160,160), 0.44)
    txt(frame, f"Gaze dev: {sub['gaze_dev']:.3f}   "
               f"Yaw dev: {sub['yaw_dev']:.1f}°",
        10, 298, (130,130,130), 0.40)

    # ── Gaze compass ──
    gcx, gcy, gr = FW-68, 68, 48
    cv2.circle(frame, (gcx,gcy), gr, (40,40,40), -1)
    cv2.circle(frame, (gcx,gcy), gr, (75,75,75),  1)
    dz = max(3, int(gr * GAZE_DEAD / GAZE_FULL))
    cv2.circle(frame, (gcx,gcy), dz, (0,160,50),  1)
    gx = int(np.clip((sub.get('_gh',bl.gh)-bl.gh)/GAZE_FULL*gr, -gr, gr))
    gy = int(np.clip((sub.get('_gv',bl.gv)-bl.gv)/GAZE_FULL*gr, -gr, gr))
    dc = (0,210,255) if sub['gaze_dev'] < GAZE_DEAD else (0,100,255)
    cv2.circle(frame, (gcx+gx, gcy+gy), 7, dc, -1)
    txt(frame, "GAZE", gcx-15, gcy+gr+14, (100,100,100), 0.34)

    # ── No-face warning ──
    if noface_cnt > NOFACE_LIMIT:
        cv2.putText(frame, "NO FACE DETECTED",
                    (FW//2-130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0,50,255), 2, cv2.LINE_AA)

    txt(frame, f"FPS:{fps:.0f}  Frame:{frame_id}",
        10, FH-10, (60,60,60), 0.36)

    # ── Calibration hint (first 10 s) ──
    # (Shown via caller if needed)


def draw_calibrating(frame, pct):
    FH, FW = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (FW,FH), (0,0,0), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    bar_w = int(FW * pct)
    cv2.rectangle(frame, (0, FH//2-28), (bar_w, FH//2+28),
                  (0,160,230), -1)
    cv2.putText(frame,
        f"  CALIBRATING {int(pct*100)}%  —  look directly at screen, sit still",
        (30, FH//2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame,
        "Keep your eyes on the camera. Do NOT look away during calibration.",
        (30, FH//2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (180,220,255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# RESULT SCREEN
# ──────────────────────────────────────────────

def draw_result_screen(frame, results: dict):
    """
    Full-screen result display shown after 2 minutes.
    Shows averaged attention/distraction + breakdown.
    """
    FH, FW = frame.shape[:2]

    # Dark background
    bg = np.zeros((FH, FW, 3), dtype=np.uint8)
    bg[:] = (14, 17, 22)

    def t(s, x, y, col=(210,210,210), sc=0.55, bold=False):
        cv2.putText(bg, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, 2 if bold else 1, cv2.LINE_AA)

    def bar(x, y, w, h, val, col, bg_col=(35,35,35)):
        cv2.rectangle(bg, (x,y), (x+w, y+h), bg_col, -1)
        cv2.rectangle(bg, (x,y), (x+int(w*val/100), y+h), col, -1)
        cv2.rectangle(bg, (x,y), (x+w, y+h), (55,55,55), 1)

    # ── Title ──
    t("2-MINUTE SESSION COMPLETE", FW//2 - 230, 55,
      (80,210,255), 0.80, True)
    cv2.line(bg, (40, 70), (FW-40, 70), (40,40,40), 1)

    att  = results['attention_avg']
    dist = results['distraction_avg']
    a_col = color_for(att)
    d_col = color_for(100 - dist)

    # ── Big score circles ──
    # Attention
    cv2.circle(bg, (FW//4, 180), 90, (30,30,30), -1)
    cv2.circle(bg, (FW//4, 180), 90, a_col, 3)
    a_txt = f"{att:.1f}%"
    (tw,th),_ = cv2.getTextSize(a_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(bg, a_txt, (FW//4 - tw//2, 180 + th//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, a_col, 2, cv2.LINE_AA)
    t("ATTENTION", FW//4 - 60, 290, a_col, 0.58, True)

    # Distraction
    cv2.circle(bg, (3*FW//4, 180), 90, (30,30,30), -1)
    cv2.circle(bg, (3*FW//4, 180), 90, d_col, 3)
    d_txt = f"{dist:.1f}%"
    (tw,th),_ = cv2.getTextSize(d_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(bg, d_txt, (3*FW//4 - tw//2, 180 + th//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, d_col, 2, cv2.LINE_AA)
    t("DISTRACTION", 3*FW//4 - 70, 290, d_col, 0.58, True)

    cv2.line(bg, (40,310), (FW-40,310), (40,40,40), 1)

    # ── Sub-score bars ──
    t("SCORE BREAKDOWN  (averaged over 2 min)", 50, 345,
      (160,160,160), 0.50)

    sub_items = [
        ("Head Pose",    results['head_avg'],  (100,200,255)),
        ("Gaze Focus",   results['gaze_avg'],  (100,255,200)),
        ("Eye Openness", results['ear_avg_s'], (255,220,100)),
        ("Blink Rate",   results['blink_avg'], (180,150,255)),
    ]

    bx, by, bw, bh = 50, 370, FW - 100, 28
    gap = 46

    for i, (label, val, col) in enumerate(sub_items):
        y_ = by + i * gap
        t(f"{label}", bx, y_ + 20, (180,180,180), 0.46)
        bar(bx + 160, y_, bw - 160, bh, val, col)
        t(f"{val:.1f}%", bx + bw - 160 + int((bw-160)*val/100) + 6,
          y_ + 20, col, 0.46, True)

    cv2.line(bg, (40, by + len(sub_items)*gap + 10),
             (FW-40, by + len(sub_items)*gap + 10), (40,40,40), 1)

    # ── Stats row ──
    sy = by + len(sub_items)*gap + 40
    stats = [
        ("Total Blinks",   f"{results['total_blinks']}"),
        ("Avg Blink Rate", f"{results['blink_rate_avg']:.1f}/min"),
        ("Frames Tracked", f"{results['frames_tracked']}"),
        ("Attentive Time", f"{results['pct_attentive']:.1f}%"),
        ("Distracted Time",f"{results['pct_distracted']:.1f}%"),
    ]
    col_w = (FW - 80) // len(stats)
    for i, (lbl, val) in enumerate(stats):
        x_ = 40 + i * col_w
        t(lbl, x_, sy,      (120,120,120), 0.38)
        t(val,  x_, sy + 26, (220,220,220), 0.52, True)

    # ── Overall verdict ──
    cv2.line(bg, (40, sy+55), (FW-40, sy+55), (40,40,40), 1)
    verdict, vcol = _verdict(att, results['pct_attentive'])
    (vw,_),_ = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    cv2.putText(bg, verdict,
                (FW//2 - vw//2, sy + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, vcol, 2, cv2.LINE_AA)

    t("Press  Q  to close", FW//2 - 80, FH - 20,
      (70,70,70), 0.42)

    return bg


def _verdict(att_avg, pct_att):
    """Verdict based on averaged attention — calibrated to tighter scoring."""
    if att_avg >= 70 and pct_att >= 60:
        return "  HIGHLY ATTENTIVE SESSION", (0, 220, 100)
    elif att_avg >= 50 and pct_att >= 45:
        return "~  MODERATELY ATTENTIVE", (0, 200, 255)
    elif att_avg >= 35:
        return "!  LOW ATTENTION — FREQUENT DISTRACTION", (0, 165, 255)
    else:
        return "X  HIGHLY DISTRACTED SESSION", (0, 60, 255)


# ──────────────────────────────────────────────
# MAIN SESSION CLASS
# ──────────────────────────────────────────────

class TwoMinSession:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.cap.set(cv2.CAP_PROP_FPS,            30)

        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.70,
        )

        self.bl          = Baseline()
        self.calibrating = True
        self.cal_start   = None

        self.mb_ear = MedBuf(); self.mb_gh = MedBuf()
        self.mb_gv  = MedBuf(); self.mb_yaw= MedBuf()
        self.mb_pit = MedBuf()

        self.ema_att = 75.0
        self.label   = "ATTENTIVE"

        # Session accumulators
        self.att_history   = []   # per-frame EMA attention
        self.head_history  = []
        self.gaze_history  = []
        self.ear_history   = []
        self.blink_history = []

        self.frame_id    = 0
        self.blinks      = 0
        self.blink_frames= 0
        self.noface_cnt  = 0
        self.sess_start  = None   # set after calibration
        self._fps_t      = time.time()
        self._fps_cnt    = 0
        self.fps         = 0.0

        self.session_done = False

    def _cam_mat(self, w, h):
        f = float(w)
        return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)

    def _tick_fps(self):
        self._fps_cnt += 1
        dt = time.time() - self._fps_t
        if dt >= 1.0:
            self.fps      = self._fps_cnt / dt
            self._fps_cnt = 0
            self._fps_t   = time.time()

    def _compute_results(self):
        """Average everything collected over the 2-minute session."""
        if not self.att_history:
            return None

        att_arr   = np.array(self.att_history)
        head_arr  = np.array(self.head_history)
        gaze_arr  = np.array(self.gaze_history)
        ear_arr   = np.array(self.ear_history)
        blink_arr = np.array(self.blink_history)

        att_avg  = float(np.mean(att_arr))
        dist_avg = 100.0 - att_avg

        pct_att  = float(np.mean(att_arr >= HYST_HIGH) * 100)
        pct_dis  = float(np.mean(att_arr <= HYST_LOW)  * 100)

        blink_rate_avg = float(np.mean(blink_arr)) if len(blink_arr) else 0.0

        return dict(
            attention_avg   = round(att_avg,  2),
            distraction_avg = round(dist_avg, 2),
            head_avg        = round(float(np.mean(head_arr)),  2),
            gaze_avg        = round(float(np.mean(gaze_arr)),  2),
            ear_avg_s       = round(float(np.mean(ear_arr)),   2),
            blink_avg       = round(float(np.mean(blink_arr)), 2),
            total_blinks    = self.blinks,
            blink_rate_avg  = round(blink_rate_avg, 2),
            frames_tracked  = len(self.att_history),
            pct_attentive   = round(pct_att, 1),
            pct_distracted  = round(pct_dis, 1),
        )

    def run(self):
        print("[INFO]  Q=quit early   (session auto-stops at 2:00)")

        # ── Calibration phase ──
        self.cal_start = time.time()
        print(f"[CAL]  Calibrating for {CAL_SECONDS}s — look at screen naturally...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            FH, FW  = frame.shape[:2]
            cam_mat = self._cam_mat(FW, FH)
            now     = time.time()

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                ear_l = calc_ear(lms, LEFT_EAR_IDX,  FW, FH)
                ear_r = calc_ear(lms, RIGHT_EAR_IDX, FW, FH)
                ear   = (ear_l + ear_r) / 2.0
                gh_l,gv_l = calc_iris_ratio(lms, LEFT_EYE_IDX,  LEFT_IRIS_IDX,  FW, FH)
                gh_r,gv_r = calc_iris_ratio(lms, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, FW, FH)
                gh = (gh_l+gh_r)/2; gv = (gv_l+gv_r)/2
                yaw, pitch, _ = calc_head_pose(lms, FW, FH, cam_mat)

                ear_s = self.mb_ear.push(ear)
                gh_s  = self.mb_gh.push(gh)
                gv_s  = self.mb_gv.push(gv)
                yaw_s = self.mb_yaw.push(yaw)
                pit_s = self.mb_pit.push(pitch)

                self.bl.feed(ear_s, gh_s, gv_s, yaw_s, pit_s)

            cal_elapsed = now - self.cal_start
            cal_pct     = min(cal_elapsed / CAL_SECONDS, 1.0)
            draw_calibrating(frame, cal_pct)
            cv2.imshow(WINDOW_NAME, frame)

            if cal_elapsed >= CAL_SECONDS:
                self.bl.finalise()
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._cleanup()
                return

        print("[INFO]  Session started — tracking for 2 minutes...")
        self.sess_start = time.time()

        # ── 2-Minute tracking phase ──
        last_subs = dict(ear=100, blink=85, gaze=100, head=100,
                         gaze_dev=0.0, yaw_dev=0.0, pit_dev=0.0,
                         _gh=self.bl.gh, _gv=self.bl.gv)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            FH, FW  = frame.shape[:2]
            cam_mat = self._cam_mat(FW, FH)
            now     = time.time()
            elapsed = now - self.sess_start

            # ── Auto-stop at 2 minutes ──
            remaining = max(0.0, SESSION_SECONDS - elapsed)
            if elapsed >= SESSION_SECONDS:
                print("[INFO]  2 minutes reached — computing results...")
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = self.mesh.process(rgb)

            if results_mp.multi_face_landmarks:
                self.noface_cnt = 0
                lms = results_mp.multi_face_landmarks[0].landmark

                ear_l = calc_ear(lms, LEFT_EAR_IDX,  FW, FH)
                ear_r = calc_ear(lms, RIGHT_EAR_IDX, FW, FH)
                ear   = (ear_l + ear_r) / 2.0
                gh_l,gv_l = calc_iris_ratio(lms,LEFT_EYE_IDX, LEFT_IRIS_IDX, FW, FH)
                gh_r,gv_r = calc_iris_ratio(lms,RIGHT_EYE_IDX,RIGHT_IRIS_IDX,FW, FH)
                gh = (gh_l+gh_r)/2; gv = (gv_l+gv_r)/2
                yaw, pitch, _ = calc_head_pose(lms, FW, FH, cam_mat)

                ear_s = self.mb_ear.push(ear)
                gh_s  = self.mb_gh.push(gh)
                gv_s  = self.mb_gv.push(gv)
                yaw_s = self.mb_yaw.push(yaw)
                pit_s = self.mb_pit.push(pitch)

                # Blink
                if ear_s < self.bl.blink_t:
                    self.blink_frames += 1
                else:
                    if self.blink_frames >= EAR_CONSEC:
                        self.blinks += 1
                    self.blink_frames = 0

                blink_rate = self.blinks / max(elapsed / 60.0, 1e-6)

                # Score
                att_raw, subs = score_frame(
                    ear_s, blink_rate, gh_s, gv_s, yaw_s, pit_s,
                    self.bl, elapsed)

                subs['_gh'] = gh_s
                subs['_gv'] = gv_s
                last_subs   = subs

                # EMA
                self.ema_att = EMA_ALPHA * att_raw + (1-EMA_ALPHA) * self.ema_att
                att_ema = round(self.ema_att, 2)

                # Hysteresis label
                if self.label == "ATTENTIVE":
                    if att_ema < HYST_LOW:  self.label = "DISTRACTED"
                else:
                    if att_ema > HYST_HIGH: self.label = "ATTENTIVE"

                # ── Accumulate ──
                self.att_history.append(att_ema)
                self.head_history.append(subs['head'])
                self.gaze_history.append(subs['gaze'])
                self.ear_history.append(subs['ear'])
                self.blink_history.append(subs['blink'])

                blink_rate_now = blink_rate

                # Landmark dots
                for idx in LEFT_EAR_IDX + RIGHT_EAR_IDX:
                    lm = lms[idx]
                    cv2.circle(frame,(int(lm.x*FW),int(lm.y*FH)),2,(170,170,255),-1)
                for idx in LEFT_IRIS_IDX + RIGHT_IRIS_IDX:
                    lm = lms[idx]
                    cv2.circle(frame,(int(lm.x*FW),int(lm.y*FH)),3,(0,230,180),-1)

            else:
                self.noface_cnt += 1
                blink_rate_now = 0.0
                att_ema = round(self.ema_att, 2)

            draw_live_ui(
                frame,
                att_ema, round(100 - att_ema, 2),
                self.label, self.bl,
                last_subs, remaining,
                self.blinks, blink_rate_now,
                self.noface_cnt, self.fps, self.frame_id
            )

            cv2.imshow(WINDOW_NAME, frame)
            self._tick_fps()
            self.frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO]  Session quit early.")
                break

        # ── Show results ──
        results_data = self._compute_results()
        if results_data:
            self._show_results(results_data)

        self._cleanup()

    def _show_results(self, results_data):
        """Show the final result screen until user presses Q."""
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        frame   = cv2.flip(frame, 1)
        FH, FW  = frame.shape[:2]

        result_frame = draw_result_screen(frame, results_data)

        # Print to console too
        print("\n" + "═"*54)
        print("  2-MINUTE SESSION RESULTS")
        print("═"*54)
        print(f"  Attention Score    : {results_data['attention_avg']:.1f}%")
        print(f"  Distraction Score  : {results_data['distraction_avg']:.1f}%")
        print(f"  Time Attentive     : {results_data['pct_attentive']:.1f}%")
        print(f"  Time Distracted    : {results_data['pct_distracted']:.1f}%")
        print(f"  ─────────────────────────────")
        print(f"  Head Score (avg)   : {results_data['head_avg']:.1f}%")
        print(f"  Gaze Score (avg)   : {results_data['gaze_avg']:.1f}%")
        print(f"  EAR Score (avg)    : {results_data['ear_avg_s']:.1f}%")
        print(f"  Blink Score (avg)  : {results_data['blink_avg']:.1f}%")
        print(f"  ─────────────────────────────")
        print(f"  Total Blinks       : {results_data['total_blinks']}")
        print(f"  Avg Blink Rate     : {results_data['blink_rate_avg']:.1f}/min")
        print(f"  Frames Tracked     : {results_data['frames_tracked']}")
        print("═"*54)

        print("\n[RESULT]  Press Q to close the result window.")

        while True:
            cv2.imshow(WINDOW_NAME, result_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="2-Minute Attention & Distraction Session Scorer")
    ap.add_argument("--source", default="0",
                    help="Camera index (0,1,2…) or video file path")
    args = ap.parse_args()
    src  = int(args.source) if args.source.isdigit() else args.source

    session = TwoMinSession(source=src)
    session.run()