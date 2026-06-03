"""
Eye Tracking & Gaze Detection — ESP32-CAM Edition
==================================================
Same structure as original + calibration + 60s session + terminal score report

Requirements:
    pip install opencv-python mediapipe numpy scipy requests
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import requests
import threading
from collections import deque
from scipy.spatial import distance as dist

# ============================================================
# 🔥 CHANGE THIS TO YOUR ESP32-CAM IP ADDRESS
# ============================================================
ESP32_IP         = "192.168.137.17"
ESP32_STREAM_URL = f"http://{ESP32_IP}/stream"
# ============================================================

WINDOW_NAME     = "Eye Tracking | ESP32-CAM"
CSV_OUTPUT      = "attention_dataset.csv"

SESSION_SECS    = 60      # 1 minute session
CAL_SECS        = 5       # calibration duration

# ── Thresholds (slightly tuned, overridden after calibration) ──
EAR_BLINK_RATIO = 0.72
EAR_OPEN_BASE   = 0.30
EAR_CONSEC      = 3

GAZE_DEAD       = 0.12
GAZE_FULL       = 0.30

YAW_DEAD        = 15
YAW_FULL        = 45
PITCH_DEAD      = 12
PITCH_FULL      = 32

W_HEAD          = 0.45
W_GAZE          = 0.30
W_EAR           = 0.15
W_BLINK         = 0.10

HYST_LOW        = 42
HYST_HIGH       = 58

EMA_ALPHA       = 0.06
SIG_MED         = 20
NOFACE_LIMIT    = 45

# ── Phases ────────────────────────────────────────────────────
PHASE_WAIT    = "WAITING"
PHASE_CAL     = "CALIBRATING"
PHASE_SESSION = "SESSION"
PHASE_DONE    = "DONE"


mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

LEFT_EAR_IDX   = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_IDX  = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDX   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_IDX  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS_IDX  = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]
HEAD_2D_IDX    = [1, 152, 263, 33, 287, 57]

HEAD_3D_PTS = np.array([
    [0.0,    0.0,    0.0  ],
    [0.0,  -330.0,  -65.0 ],
    [-225.0, 170.0, -135.0],
    [ 225.0, 170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [ 150.0,-150.0, -125.0],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# SAME ESP32Stream CLASS — unchanged
# ─────────────────────────────────────────────────────────────
class ESP32Stream:
    def __init__(self, url):
        self.url     = url
        self.frame   = None
        self.grabbed = False
        self.stopped = False
        self._lock   = threading.Lock()
        self._error  = None

        print(f"[INFO] Connecting to: {url}")
        try:
            self._response = requests.get(
                url, stream=True, timeout=10,
                headers={"Connection": "keep-alive"}
            )
            ct = self._response.headers.get("Content-Type", "")
            print(f"[INFO] Connected! Content-Type: {ct}")
        except Exception as e:
            self._error = str(e)
            print(f"[ERROR] {e}")
            return

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        print("[INFO] Waiting for first frame (up to 8s)...")
        for _ in range(80):
            if self.grabbed:
                print(f"[INFO] ✅ First frame OK! Window opening...\n")
                return
            time.sleep(0.1)
        print("[WARN] ⚠️  Still no frame. Check URL or stream format.")

    def _reader(self):
        buf = bytes()
        try:
            for chunk in self._response.iter_content(chunk_size=8192):
                if self.stopped:
                    break
                if not chunk:
                    continue
                buf += chunk
                if len(buf) > 2_000_000:
                    last = buf.rfind(b'\xff\xd8')
                    buf = buf[last:] if last > 0 else bytes()
                while True:
                    start = buf.find(b'\xff\xd8')
                    end   = buf.find(b'\xff\xd9', start + 2) if start != -1 else -1
                    if start == -1 or end == -1:
                        break
                    jpg_data = buf[start:end + 2]
                    buf      = buf[end + 2:]
                    img = cv2.imdecode(
                        np.frombuffer(jpg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        with self._lock:
                            self.frame   = img
                            self.grabbed = True
        except Exception as e:
            print(f"[STREAM ERROR] {e}")

    def read(self):
        with self._lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        try:
            self._response.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# SAME FEATURE FUNCTIONS — unchanged
# ─────────────────────────────────────────────────────────────
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
    pts2d = np.array(
        [[lms[i].x * w, lms[i].y * h] for i in HEAD_2D_IDX],
        dtype=np.float64
    )
    ok, rvec, _ = cv2.solvePnP(
        HEAD_3D_PTS, pts2d, cam_mat, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy    = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
    yaw   = float(np.degrees(np.arctan2(rmat[1, 0],  rmat[0, 0])))
    roll  = float(np.degrees(np.arctan2(rmat[2, 1],  rmat[2, 2])))
    return yaw, pitch, roll

def gaze_score(gh, gv):
    dh = abs(gh - 0.5) * 2
    dv = abs(gv - 0.5) * 2
    d  = np.hypot(dh, dv)
    if d <= GAZE_DEAD: return 1.0
    if d >= GAZE_FULL: return 0.0
    return 1.0 - (d - GAZE_DEAD) / (GAZE_FULL - GAZE_DEAD)

def head_score(yaw, pitch):
    def s1d(v, dead, full):
        a = abs(v)
        if a <= dead: return 1.0
        if a >= full: return 0.0
        return 1.0 - (a - dead) / (full - dead)
    return (s1d(yaw, YAW_DEAD, YAW_FULL) + s1d(pitch, PITCH_DEAD, PITCH_FULL)) / 2.0

def ear_score(ear):
    return float(np.clip((ear - 0.20) / (EAR_OPEN_BASE - 0.20 + 1e-6), 0.0, 1.0))

def draw_bar(img, x, y, w, h, value, color, label):
    cv2.rectangle(img, (x, y), (x+w, y+h), (50,50,50), -1)
    filled = int(w * np.clip(value/100.0, 0, 1))
    cv2.rectangle(img, (x, y), (x+filled, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (120,120,120), 1)
    cv2.putText(img, f"{label}: {value:.0f}%", (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

def draw_gaze_indicator(img, gh, gv, cx, cy, radius=42):
    cv2.circle(img, (cx, cy), radius, (55,55,55), -1)
    cv2.circle(img, (cx, cy), radius, (140,140,140), 1)
    dx = int(cx + (gh - 0.5) * 2 * radius * 0.7)
    dy = int(cy + (gv - 0.5) * 2 * radius * 0.7)
    cv2.circle(img, (dx, dy), 8, (0,255,255), -1)
    cv2.putText(img, "GAZE", (cx-18, cy+radius+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170,170,170), 1)


# ─────────────────────────────────────────────────────────────
# CALIBRATION — slightly adjusts thresholds to your face
# ─────────────────────────────────────────────────────────────
class Calibration:
    def __init__(self):
        self.ears=[]; self.yaws=[]; self.pitchs=[]; self.ghs=[]; self.gvs=[]

    def add(self, ear, yaw, pitch, gh, gv):
        self.ears.append(ear)
        self.yaws.append(abs(yaw))
        self.pitchs.append(abs(pitch))
        self.ghs.append(gh)
        self.gvs.append(gv)

    def apply(self):
        global EAR_BLINK_RATIO, EAR_OPEN_BASE
        global GAZE_DEAD, GAZE_FULL
        global YAW_DEAD, YAW_FULL, PITCH_DEAD, PITCH_FULL

        if len(self.ears) < 10:
            print("[CAL] Not enough data, keeping defaults.")
            return

        ear_mean = float(np.mean(self.ears))
        ear_std  = float(np.std(self.ears))
        EAR_OPEN_BASE   = ear_mean
        EAR_BLINK_RATIO = max(0.55, ear_mean - 2.5 * ear_std)

        nat = max(float(np.std(self.ghs)), float(np.std(self.gvs)))
        GAZE_DEAD = float(np.clip(nat * 1.5, 0.06, 0.18))
        GAZE_FULL = float(np.clip(nat * 4.0, 0.20, 0.50))

        ys = float(np.std(self.yaws))
        ps = float(np.std(self.pitchs))
        YAW_DEAD   = float(np.clip(ys * 2.5 + 5,  8, 20))
        YAW_FULL   = float(np.clip(ys * 8.0 + 20, 30, 55))
        PITCH_DEAD = float(np.clip(ps * 2.5 + 4,  6, 18))
        PITCH_FULL = float(np.clip(ps * 8.0 + 15, 22, 45))

        print(f"[CAL] ✅ Done! EAR open={EAR_OPEN_BASE:.3f}  blink<{EAR_BLINK_RATIO:.3f}")
        print(f"[CAL]    Gaze dead={GAZE_DEAD:.3f}  full={GAZE_FULL:.3f}")
        print(f"[CAL]    Yaw {YAW_DEAD:.1f}–{YAW_FULL:.1f}°  Pitch {PITCH_DEAD:.1f}–{PITCH_FULL:.1f}°\n")


# ─────────────────────────────────────────────────────────────
# TERMINAL RESULT PRINTER
# ─────────────────────────────────────────────────────────────
def print_result(att_pct, dis_pct, err_pct, total_blinks):
    overall = att_pct
    if   overall >= 85: grade = "A — Excellent Focus!"
    elif overall >= 70: grade = "B — Good Attention"
    elif overall >= 55: grade = "C — Moderate Focus"
    elif overall >= 40: grade = "D — Low Attention"
    else:               grade = "F — Very Distracted"

    att_t = att_pct / 100 * SESSION_SECS
    dis_t = dis_pct / 100 * SESSION_SECS
    err_t = err_pct / 100 * SESSION_SECS

    print("\n" + "="*50)
    print("       SESSION COMPLETE — 60s REPORT")
    print("="*50)
    print(f"  Overall Score      : {overall:.1f}%")
    print(f"  Grade              : {grade}")
    print("-"*50)
    print(f"  ✅ Attentive Score  : {att_pct:.1f}%   ({att_t:.0f}s)")
    print(f"  🔴 Distracted Score : {dis_pct:.1f}%   ({dis_t:.0f}s)")
    print(f"  ⚠️  Error Score      : {err_pct:.1f}%   ({err_t:.0f}s)")
    print("-"*50)
    print(f"  Total Blinks       : {total_blinks}")
    print(f"  CSV saved          : {CSV_OUTPUT}")
    print("="*50)
    print("  Press R to restart  |  Q to quit")
    print("="*50 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN — same structure as original, phases layered on top
# ─────────────────────────────────────────────────────────────
def main():
    cap = None
    try:
        cap = ESP32Stream(ESP32_STREAM_URL)
    except Exception:
        print("\n[FATAL] Could not connect to ESP32.")
        print("Check IP and URL then restart.")
        return

    mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # ── Session state ─────────────────────────────────────────
    phase           = PHASE_WAIT
    cal             = Calibration()
    cal_start       = 0.0
    sess_start      = 0.0

    blink_counter   = 0
    total_blinks    = 0
    blink_flag      = False
    noface_counter  = 0
    ema_score       = 50.0
    median_buf      = deque(maxlen=SIG_MED)
    attention_label = "ATTENTIVE"
    prev_label      = "ATTENTIVE"
    frame_count     = 0
    fps_time        = time.time()
    fps             = 0.0
    no_frame_count  = 0

    # Score counters
    att_frames  = 0
    dis_frames  = 0
    err_frames  = 0
    total_frames = 0

    # Cache final percentages for DONE phase
    att_pct = dis_pct = err_pct = 0.0

    csv_file = open(CSV_OUTPUT, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["timestamp","ear","yaw","pitch","roll",
                     "gaze_h","gaze_v","attention_score","label"])

    print("[INFO] Window is open.")
    print("[INFO] Look at camera and press SPACE to start calibration.")
    print("[INFO] Press Q to quit.\n")

    while True:
        ret, frame = cap.read()

        # ── Waiting screen when no frame ──────────────────────
        if not ret or frame is None:
            no_frame_count += 1
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Connecting to ESP32-CAM...",
                        (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
            cv2.putText(blank, ESP32_STREAM_URL,
                        (60, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
            cv2.putText(blank, f"Attempts: {no_frame_count}",
                        (60, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,100,100), 1)
            cv2.putText(blank, "Press Q to quit",
                        (60, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,80), 1)
            cv2.imshow(WINDOW_NAME, blank)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            continue

        no_frame_count = 0
        frame = cv2.flip(frame, 1)
        FH, FW = frame.shape[:2]
        cam_mat = np.array([[FW,0,FW/2],[0,FW,FH/2],[0,0,1]], dtype=np.float64)
        now = time.time()

        frame_count += 1
        if frame_count % 15 == 0:
            fps = 15 / (time.time() - fps_time + 1e-6)
            fps_time = time.time()

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)
        raw_score = ema_score
        face_ok   = False

        # ══════════════════════════════════════════════════════
        # PHASE: WAITING
        # ══════════════════════════════════════════════════════
        if phase == PHASE_WAIT:
            face_ok = results.multi_face_landmarks is not None
            if results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
                mp_draw.draw_landmarks(
                    frame, results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
            col = (0,210,80) if face_ok else (0,70,200)
            msg = "Face detected! Press SPACE to calibrate" if face_ok else "No face — move into frame"
            cv2.putText(frame, msg, (FW//2-230, FH//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 2)
            cv2.putText(frame, f"Session: {SESSION_SECS}s  |  Cal: {CAL_SECS}s",
                        (FW//2-155, FH//2+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,180), 1)
            cv2.putText(frame, f"FPS:{fps:.1f}", (FW-110, FH-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,160,160), 1)

        # ══════════════════════════════════════════════════════
        # PHASE: CALIBRATING
        # ══════════════════════════════════════════════════════
        elif phase == PHASE_CAL:
            elapsed = now - cal_start
            remain  = max(0, CAL_SECS - elapsed)

            if results.multi_face_landmarks:
                face_ok = True
                lms = results.multi_face_landmarks[0].landmark
                ear_l = calc_ear(lms, LEFT_EAR_IDX,  FW, FH)
                ear_r = calc_ear(lms, RIGHT_EAR_IDX, FW, FH)
                ear_c = (ear_l + ear_r) / 2.0
                gh_l, gv_l = calc_iris_ratio(lms, LEFT_EYE_IDX,  LEFT_IRIS_IDX,  FW, FH)
                gh_r, gv_r = calc_iris_ratio(lms, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, FW, FH)
                yaw_c, pitch_c, _ = calc_head_pose(lms, FW, FH, cam_mat)
                cal.add(ear_c, yaw_c, pitch_c, (gh_l+gh_r)/2, (gv_l+gv_r)/2)

                mp_draw.draw_landmarks(
                    frame, results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
                mp_draw.draw_landmarks(
                    frame, results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

            # Progress bar
            cv2.rectangle(frame, (0, FH-60), (FW, FH), (10,10,20), -1)
            bw = int((FW-40) * np.clip(elapsed/CAL_SECS, 0, 1))
            cv2.rectangle(frame, (20, FH-42), (20+bw, FH-22), (0,180,255), -1)
            cv2.rectangle(frame, (20, FH-42), (FW-20, FH-22), (70,70,80), 1)
            cv2.putText(frame, f"CALIBRATING — look straight ahead — {remain:.1f}s",
                        (28, FH-48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
            cv2.putText(frame, f"FPS:{fps:.1f}", (FW-110, FH-65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,160,160), 1)

            if elapsed >= CAL_SECS:
                cal.apply()
                phase      = PHASE_SESSION
                sess_start = now
                print(f"[INFO] Session started! {SESSION_SECS}s timer running...\n")

        # ══════════════════════════════════════════════════════
        # PHASE: SESSION — exact original structure preserved
        # ══════════════════════════════════════════════════════
        elif phase == PHASE_SESSION:
            elapsed = now - sess_start
            remain  = max(0, SESSION_SECS - elapsed)

            if results.multi_face_landmarks:
                face_ok        = True
                noface_counter = 0
                lms = results.multi_face_landmarks[0].landmark

                ear_l = calc_ear(lms, LEFT_EAR_IDX,  FW, FH)
                ear_r = calc_ear(lms, RIGHT_EAR_IDX, FW, FH)
                ear   = (ear_l + ear_r) / 2.0

                gh_l, gv_l = calc_iris_ratio(lms, LEFT_EYE_IDX,  LEFT_IRIS_IDX,  FW, FH)
                gh_r, gv_r = calc_iris_ratio(lms, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, FW, FH)
                gh = (gh_l + gh_r) / 2.0
                gv = (gv_l + gv_r) / 2.0

                yaw, pitch, roll = calc_head_pose(lms, FW, FH, cam_mat)

                if ear < EAR_BLINK_RATIO:
                    blink_counter += 1
                else:
                    if blink_counter >= EAR_CONSEC:
                        total_blinks += 1
                        blink_flag = True
                    blink_counter = 0
                blink_penalty = 0.6 if blink_flag else 1.0
                blink_flag = False

                hs = head_score(yaw, pitch)
                gs = gaze_score(gh, gv)
                es = ear_score(ear)
                raw_score = (W_HEAD*hs + W_GAZE*gs + W_EAR*es + W_BLINK*blink_penalty) * 100.0

                mp_draw.draw_landmarks(
                    frame, results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
                mp_draw.draw_landmarks(
                    frame, results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

                for idx_list, color in [(LEFT_IRIS_IDX,(0,255,255)),(RIGHT_IRIS_IDX,(0,200,255))]:
                    pts = [(int(lms[i].x*FW), int(lms[i].y*FH)) for i in idx_list]
                    cx_ = int(np.mean([p[0] for p in pts]))
                    cy_ = int(np.mean([p[1] for p in pts]))
                    r_  = int(dist.euclidean(pts[0], pts[2])/2) + 4
                    cv2.circle(frame, (cx_,cy_), r_, color, 2)
                    cv2.circle(frame, (cx_,cy_), 2,  color, -1)

                gaze_dir  = "RIGHT" if gh < 0.38 else ("LEFT" if gh > 0.62 else "CENTER")
                gaze_dir += " UP"   if gv < 0.38 else (" DOWN" if gv > 0.62 else "")

                lines = [
                    (f"EAR   : {ear:.3f}", (0,255,100) if ear>=EAR_BLINK_RATIO else (0,60,255)),
                    (f"Yaw   : {yaw:+.1f}",  (255,200,0)),
                    (f"Pitch : {pitch:+.1f}", (255,200,0)),
                    (f"Roll  : {roll:+.1f}",  (200,200,200)),
                    (f"Gaze  : {gaze_dir}",   (0,200,255)),
                    (f"Blinks: {total_blinks}",(180,180,255)),
                ]
                for i,(txt,col) in enumerate(lines):
                    cv2.putText(frame, txt, (10, 35+i*28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)

                draw_gaze_indicator(frame, gh, gv, FW-65, 65)

                writer.writerow([time.strftime("%H:%M:%S"),
                    f"{ear:.4f}",f"{yaw:.2f}",f"{pitch:.2f}",f"{roll:.2f}",
                    f"{gh:.4f}",f"{gv:.4f}",f"{raw_score:.1f}",attention_label])

            else:
                noface_counter += 1
                if noface_counter > NOFACE_LIMIT:
                    raw_score = 0.0
                cv2.putText(frame, "NO FACE DETECTED", (FW//2-130, FH//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            ema_score = EMA_ALPHA * raw_score + (1-EMA_ALPHA) * ema_score
            median_buf.append(ema_score)
            smooth_score = float(np.median(median_buf))

            if prev_label == "ATTENTIVE":
                if smooth_score < HYST_LOW:  attention_label = "DISTRACTED"
            else:
                if smooth_score > HYST_HIGH: attention_label = "ATTENTIVE"
            prev_label = attention_label

            # ── Frame bucket counters ─────────────────────────
            total_frames += 1
            if not face_ok or noface_counter > NOFACE_LIMIT:
                err_frames += 1
            elif attention_label == "ATTENTIVE":
                att_frames += 1
            else:
                dis_frames += 1

            # ── Original HUD (unchanged) ──────────────────────
            bar_color = (0,220,80) if attention_label=="ATTENTIVE" else (0,60,220)
            draw_bar(frame, 10, FH-40, FW-20, 22, smooth_score, bar_color, "Attention")

            lbl_color = (0,230,80) if attention_label=="ATTENTIVE" else (0,80,255)
            cv2.putText(frame, attention_label, (FW-200,35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, lbl_color, 2)
            cv2.putText(frame, f"{smooth_score:.0f}%", (FW-80,65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, lbl_color, 2)
            cv2.putText(frame, f"FPS:{fps:.1f}", (FW-110,FH-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,160,160), 1)
            cv2.putText(frame, f"ESP32:{ESP32_IP}", (10,FH-55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120,120,120), 1)

            # Countdown top-centre
            cv2.putText(frame, f"{remain:.0f}s", (FW//2-20, 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.80, (0,200,255), 2)

            # ── Session end ───────────────────────────────────
            if elapsed >= SESSION_SECS:
                tf      = max(total_frames, 1)
                att_pct = 100 * att_frames / tf
                dis_pct = 100 * dis_frames / tf
                err_pct = 100 * err_frames / tf
                print_result(att_pct, dis_pct, err_pct, total_blinks)
                csv_file.flush()
                phase = PHASE_DONE

        # ══════════════════════════════════════════════════════
        # PHASE: DONE — freeze + result overlay on window
        # ══════════════════════════════════════════════════════
        elif phase == PHASE_DONE:
            overall = att_pct
            if   overall >= 85: g = "A"
            elif overall >= 70: g = "B"
            elif overall >= 55: g = "C"
            elif overall >= 40: g = "D"
            else:               g = "F"
            att_t = att_pct / 100 * SESSION_SECS
            dis_t = dis_pct / 100 * SESSION_SECS
            err_t = err_pct / 100 * SESSION_SECS

            ov = frame.copy()
            px, py = FW//2-220, FH//2-130
            cv2.rectangle(ov, (px,py), (px+440,py+265), (15,15,25), -1)
            cv2.rectangle(ov, (px,py), (px+440,py+265), (70,70,110), 2)

            cv2.putText(ov, "SESSION COMPLETE",
                        (px+90, py+35), cv2.FONT_HERSHEY_DUPLEX, 0.80, (200,200,255), 2)
            cv2.putText(ov, f"Overall: {overall:.1f}%   Grade: {g}",
                        (px+30, py+72), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)

            cv2.putText(ov, f"Attentive Score  : {att_pct:.1f}%   ({att_t:.0f}s)",
                        (px+30, py+115), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0,210,80),  2)
            cv2.putText(ov, f"Distracted Score : {dis_pct:.1f}%   ({dis_t:.0f}s)",
                        (px+30, py+150), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0,80,255),  2)
            cv2.putText(ov, f"Error Score      : {err_pct:.1f}%   ({err_t:.0f}s)",
                        (px+30, py+185), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0,140,255), 2)

            cv2.putText(ov, f"Total Blinks: {total_blinks}",
                        (px+30, py+222), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,220), 1)
            cv2.putText(ov, "R = restart   |   Q = quit",
                        (px+30, py+248), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120,120,150), 1)

            frame = cv2.addWeighted(frame, 0.25, ov, 1.0, 0)

        # ── Show frame ────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Quitting...")
            break

        elif key == ord(' ') and phase == PHASE_WAIT:
            if results.multi_face_landmarks:
                phase     = PHASE_CAL
                cal_start = time.time()
                print(f"[INFO] Calibrating for {CAL_SECS}s — look straight ahead...")
            else:
                print("[WARN] No face in frame. Move closer to the camera.")

        elif key == ord('r') and phase == PHASE_DONE:
            phase           = PHASE_WAIT
            cal             = Calibration()
            blink_counter   = 0
            total_blinks    = 0
            blink_flag      = False
            noface_counter  = 0
            ema_score       = 50.0
            median_buf      = deque(maxlen=SIG_MED)
            attention_label = "ATTENTIVE"
            prev_label      = "ATTENTIVE"
            att_frames      = 0
            dis_frames      = 0
            err_frames      = 0
            total_frames    = 0
            att_pct = dis_pct = err_pct = 0.0
            print("[INFO] Reset. Press SPACE to start again.\n")

    if cap: cap.release()
    mesh.close()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved → {CSV_OUTPUT}  |  Total blinks: {total_blinks}")


if __name__ == "__main__":
    main()