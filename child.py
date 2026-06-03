"""
AUTISM TOY — PHASE 1: GREETING
================================
Flow:
  1. Camera calibrates for 5s
  2. Plays greeting.mp3 from PC speakers
  3. Listens for 10s via PC mic
     — simultaneously tracks eye/gaze from ESP32-CAM
  4. Scores:
       Eye Score  = attentive % (from camera)
       Mic Score  = delay + volume + detection
       Final      = 55% eye + 45% mic
  5. Prints result in terminal + shows on screen

Requirements:
    pip install opencv-python mediapipe numpy scipy requests pyaudio pygame SpeechRecognition
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import wave
import struct
import threading
import pyaudio
import pygame
import speech_recognition as sr
import requests
from collections import deque
from scipy.spatial import distance as dist

# ============================================================
# CONFIGURATION — change these
# ============================================================
ESP32_IP         = "192.168.137.129"
ESP32_STREAM_URL = f"http://{ESP32_IP}/stream"

GREETING_AUDIO   = "greeting.mp3"   # path to your greeting mp3
LISTEN_SECS      = 10               # wait 10s for child response
CAL_SECS         = 5                # calibration seconds

# Mic config
SAMPLE_RATE      = 16000
CHANNELS         = 1
CHUNK            = 1024
MIC_THRESHOLD    = 600              # RMS above this = sound detected
MIC_CORRECT_RMS  = 1800             # RMS above this = clear response

# Score weights
EYE_WEIGHT       = 0.55
MIC_WEIGHT       = 0.45
# ============================================================

WINDOW = "Autism Toy — Phase 1: Greeting"

# Eye tracking thresholds (calibration refines these)
EAR_BLINK_RATIO = 0.72
EAR_OPEN_BASE   = 0.30
EAR_CONSEC      = 3
GAZE_DEAD       = 0.12
GAZE_FULL       = 0.30
YAW_DEAD        = 15.0
YAW_FULL        = 45.0
PITCH_DEAD      = 12.0
PITCH_FULL      = 32.0
W_HEAD          = 0.45
W_GAZE          = 0.30
W_EAR           = 0.15
W_BLINK         = 0.10
HYST_LOW        = 42
HYST_HIGH       = 58
EMA_ALPHA       = 0.06
SIG_MED         = 20
NOFACE_LIMIT    = 30

# Phases
PH_CAL      = "CALIBRATING"
PH_PLAY     = "PLAYING"
PH_LISTEN   = "LISTENING"
PH_RESULT   = "RESULT"

mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

LEFT_EAR_IDX   = [362,385,387,263,373,380]
RIGHT_EAR_IDX  = [33,160,158,133,153,144]
LEFT_EYE_IDX   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_IDX  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS_IDX  = [474,475,476,477]
RIGHT_IRIS_IDX = [469,470,471,472]
HEAD_2D_IDX    = [1,152,263,33,287,57]

HEAD_3D_PTS = np.array([
    [0.0,0.0,0.0],[0.0,-330.0,-65.0],
    [-225.0,170.0,-135.0],[225.0,170.0,-135.0],
    [-150.0,-150.0,-125.0],[150.0,-150.0,-125.0]
], dtype=np.float64)


# ─────────────────────────────────────────────────────────
# ESP32-CAM STREAM
# ─────────────────────────────────────────────────────────
class ESP32Stream:
    def __init__(self, url):
        self.frame=None; self.grabbed=False; self.stopped=False
        self._lock=threading.Lock()
        print(f"[CAM] Connecting to {url} ...")
        try:
            self._resp = requests.get(url, stream=True, timeout=10,
                                      headers={"Connection":"keep-alive"})
            print(f"[CAM] Connected  Content-Type: {self._resp.headers.get('Content-Type','')}")
        except Exception as e:
            print(f"[CAM ERROR] {e}"); return
        threading.Thread(target=self._reader, daemon=True).start()
        for _ in range(80):
            if self.grabbed: print("[CAM] ✅ First frame OK\n"); return
            time.sleep(0.1)
        print("[CAM] ⚠️  No frame yet")

    def _reader(self):
        buf = bytes()
        try:
            for chunk in self._resp.iter_content(chunk_size=8192):
                if self.stopped or not chunk: continue
                buf += chunk
                if len(buf) > 2_000_000:
                    last = buf.rfind(b'\xff\xd8')
                    buf  = buf[last:] if last > 0 else bytes()
                while True:
                    s = buf.find(b'\xff\xd8')
                    e = buf.find(b'\xff\xd9', s+2) if s != -1 else -1
                    if s == -1 or e == -1: break
                    img = cv2.imdecode(np.frombuffer(buf[s:e+2], dtype=np.uint8),
                                       cv2.IMREAD_COLOR)
                    buf = buf[e+2:]
                    if img is not None:
                        with self._lock:
                            self.frame   = img
                            self.grabbed = True
        except Exception as ex:
            print(f"[CAM STREAM] {ex}")

    def read(self):
        with self._lock:
            if not self.grabbed: return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        try: self._resp.close()
        except: pass


# ─────────────────────────────────────────────────────────
# MIC RECORDER  (runs in background thread)
# ─────────────────────────────────────────────────────────
class MicRecorder:
    def __init__(self):
        self.frames       = []
        self.rms_history  = []        # list of RMS per chunk
        self.recording    = False
        self.first_sound  = None      # time of first detected sound
        self.start_time   = None
        self._thread      = None

    def start(self):
        self.frames      = []
        self.rms_history = []
        self.first_sound = None
        self.start_time  = time.time()
        self.recording   = True
        self._thread     = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        print("[MIC] Recording started")

    def _record(self):
        pa     = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                         rate=SAMPLE_RATE, input=True,
                         frames_per_buffer=CHUNK)
        while self.recording:
            data    = stream.read(CHUNK, exception_on_overflow=False)
            self.frames.append(data)
            samples = struct.unpack(f"{len(data)//2}h", data)
            rms     = int(np.sqrt(np.mean(np.array(samples, dtype=np.float32)**2)))
            self.rms_history.append(rms)
            if rms > MIC_THRESHOLD and self.first_sound is None:
                self.first_sound = time.time()
                print(f"[MIC] Sound detected! RMS={rms}")
        stream.stop_stream()
        stream.close()
        pa.terminate()

    def stop(self):
        self.recording = False
        if self._thread: self._thread.join(timeout=3)
        print(f"[MIC] Stopped. Chunks={len(self.rms_history)}")

    def current_rms(self):
        """Latest RMS for live bar on screen."""
        return self.rms_history[-1] if self.rms_history else 0

    def metrics(self):
        """Returns dict of all mic metrics after recording."""
        if not self.rms_history:
            return {"sound":False,"delay":None,"avg_rms":0,"max_rms":0,"response_type":"timeout"}

        avg_rms = float(np.mean(self.rms_history))
        max_rms = float(np.max(self.rms_history))
        sound   = self.first_sound is not None
        delay   = (self.first_sound - self.start_time) if sound else None

        if not sound:
            resp_type = "timeout"
        elif max_rms >= MIC_CORRECT_RMS and delay is not None and delay < 6.0:
            resp_type = "correct"
        else:
            resp_type = "mismatch"

        return {
            "sound"     : sound,
            "delay"     : delay,
            "avg_rms"   : avg_rms,
            "max_rms"   : max_rms,
            "response_type" : resp_type
        }

    def transcribe(self):
        """Google speech-to-text on recorded audio."""
        if not self.frames:
            return "[no audio recorded]"
        try:
            tmp = "tmp_phase1.wav"
            wf  = wave.open(tmp, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            rec   = sr.Recognizer()
            with sr.AudioFile(tmp) as src:
                audio = rec.record(src)
            text = rec.recognize_google(audio, language="en-US")
            import os; os.remove(tmp)
            return text
        except sr.UnknownValueError:
            return "[no speech detected]"
        except sr.RequestError:
            return "[speech API error]"
        except Exception as ex:
            return f"[{ex}]"


# ─────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────
class Calibration:
    def __init__(self):
        self.ears=[]; self.yaws=[]; self.pitchs=[]; self.ghs=[]; self.gvs=[]

    def add(self, ear, yaw, pitch, gh, gv):
        self.ears.append(ear); self.yaws.append(abs(yaw))
        self.pitchs.append(abs(pitch)); self.ghs.append(gh); self.gvs.append(gv)

    def apply(self):
        global EAR_BLINK_RATIO, EAR_OPEN_BASE
        global GAZE_DEAD, GAZE_FULL, YAW_DEAD, YAW_FULL, PITCH_DEAD, PITCH_FULL
        if len(self.ears) < 10: print("[CAL] Not enough data, using defaults"); return
        em = float(np.mean(self.ears)); es = float(np.std(self.ears))
        EAR_OPEN_BASE   = em
        EAR_BLINK_RATIO = max(0.55, em - 2.5*es)
        nat = max(float(np.std(self.ghs)), float(np.std(self.gvs)))
        GAZE_DEAD = float(np.clip(nat*1.5, 0.06, 0.18))
        GAZE_FULL = float(np.clip(nat*4.0, 0.20, 0.50))
        ys = float(np.std(self.yaws)); ps = float(np.std(self.pitchs))
        YAW_DEAD   = float(np.clip(ys*2.5+5,  8, 20))
        YAW_FULL   = float(np.clip(ys*8.0+20, 30, 55))
        PITCH_DEAD = float(np.clip(ps*2.5+4,  6, 18))
        PITCH_FULL = float(np.clip(ps*8.0+15, 22, 45))
        print(f"[CAL] ✅ EAR open={EAR_OPEN_BASE:.3f} blink<{EAR_BLINK_RATIO:.3f}")
        print(f"[CAL]    Gaze dead={GAZE_DEAD:.3f}–full={GAZE_FULL:.3f}")
        print(f"[CAL]    Yaw {YAW_DEAD:.1f}–{YAW_FULL:.1f}°  Pitch {PITCH_DEAD:.1f}–{PITCH_FULL:.1f}°\n")


# ─────────────────────────────────────────────────────────
# EYE FEATURE FUNCTIONS
# ─────────────────────────────────────────────────────────
def calc_ear(lms, idx, w, h):
    pts = np.array([(lms[i].x*w, lms[i].y*h) for i in idx])
    A = dist.euclidean(pts[1],pts[5]); B = dist.euclidean(pts[2],pts[4])
    C = dist.euclidean(pts[0],pts[3])+1e-6
    return (A+B)/(2.0*C)

def calc_iris_ratio(lms, eye_idx, iris_idx, w, h):
    eye  = np.array([(lms[i].x*w, lms[i].y*h) for i in eye_idx])
    iris = np.array([(lms[i].x*w, lms[i].y*h) for i in iris_idx])
    cx,cy = np.mean(iris,axis=0)
    xmn,ymn = eye.min(axis=0); xmx,ymx = eye.max(axis=0)
    return (float(np.clip((cx-xmn)/(xmx-xmn+1e-6),0,1)),
            float(np.clip((cy-ymn)/(ymx-ymn+1e-6),0,1)))

def calc_head_pose(lms, w, h, cam_mat):
    pts2d = np.array([[lms[i].x*w, lms[i].y*h] for i in HEAD_2D_IDX], dtype=np.float64)
    ok,rvec,_ = cv2.solvePnP(HEAD_3D_PTS, pts2d, cam_mat, np.zeros((4,1)),
                              flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0,0.0,0.0
    rmat,_ = cv2.Rodrigues(rvec); sy = np.sqrt(rmat[0,0]**2+rmat[1,0]**2)
    return (float(np.degrees(np.arctan2(rmat[1,0],rmat[0,0]))),
            float(np.degrees(np.arctan2(-rmat[2,0],sy))),
            float(np.degrees(np.arctan2(rmat[2,1],rmat[2,2]))))

def gaze_score_fn(gh, gv):
    d = np.hypot(abs(gh-0.5)*2, abs(gv-0.5)*2)
    if d<=GAZE_DEAD: return 1.0
    if d>=GAZE_FULL: return 0.0
    return 1.0-(d-GAZE_DEAD)/(GAZE_FULL-GAZE_DEAD)

def head_score_fn(yaw, pitch):
    def s(v,d,f): a=abs(v); return 1.0 if a<=d else (0.0 if a>=f else 1.0-(a-d)/(f-d))
    return (s(yaw,YAW_DEAD,YAW_FULL)+s(pitch,PITCH_DEAD,PITCH_FULL))/2.0

def ear_score_fn(ear):
    return float(np.clip((ear-0.20)/(EAR_OPEN_BASE-0.20+1e-6),0.0,1.0))


# ─────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────
def compute_mic_score(m):
    """Convert mic metrics dict → 0-100 score."""
    if not m["sound"] or m["response_type"] == "timeout":
        return 0.0
    delay      = m["delay"] if m["delay"] else LISTEN_SECS
    delay_sc   = max(0.0, 100.0 - (delay/LISTEN_SECS)*100.0)
    vol_sc     = float(np.clip((m["avg_rms"]-MIC_THRESHOLD)/(3000-MIC_THRESHOLD)*100,0,100))
    type_bonus = 20.0 if m["response_type"]=="correct" else 0.0
    return float(np.clip(0.5*delay_sc + 0.3*vol_sc + 0.2*type_bonus, 0, 100))

def compute_eye_score(att_f, total_f):
    return round(100.0*att_f/max(total_f,1), 1)

def compute_final(eye_sc, mic_sc):
    return round(EYE_WEIGHT*eye_sc + MIC_WEIGHT*mic_sc, 1)

def grade(score):
    if score>=85: return "A",(0,210,80), "Excellent!"
    if score>=70: return "B",(50,200,160),"Good"
    if score>=55: return "C",(0,180,255), "Moderate"
    if score>=40: return "D",(30,120,255),"Low"
    return             "F",(0,60,220),  "Very Low"


# ─────────────────────────────────────────────────────────
# AUDIO PLAYBACK
# ─────────────────────────────────────────────────────────
def play_audio_blocking(path):
    """Play mp3 and block until done."""
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        print(f"[AUDIO] Playing {path} ...")
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        print("[AUDIO] Done")
    except Exception as ex:
        print(f"[AUDIO ERROR] {ex}")


# ─────────────────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────────────────
def draw_bar(img, x, y, w, h, value, color, label):
    cv2.rectangle(img,(x,y),(x+w,y+h),(50,50,50),-1)
    cv2.rectangle(img,(x,y),(x+int(w*np.clip(value/100,0,1)),y+h),color,-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),(110,110,110),1)
    cv2.putText(img,f"{label}: {value:.0f}%",(x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,color,1)

def draw_gaze_dot(img, gh, gv, cx, cy, r=36):
    cv2.circle(img,(cx,cy),r,(40,40,40),-1)
    cv2.circle(img,(cx,cy),r,(100,100,100),1)
    cv2.line(img,(cx-r,cy),(cx+r,cy),(65,65,65),1)
    cv2.line(img,(cx,cy-r),(cx,cy+r),(65,65,65),1)
    dx=int(cx+(gh-0.5)*2*r*0.75); dy=int(cy+(gv-0.5)*2*r*0.75)
    cv2.circle(img,(dx,dy),8,(0,255,255),-1)
    cv2.putText(img,"GAZE",(cx-18,cy+r+13),cv2.FONT_HERSHEY_SIMPLEX,0.38,(140,140,140),1)

def draw_result_overlay(frame, mic_m, eye_sc, mic_sc, final_sc,
                        att_pct, dis_pct, err_pct, transcript):
    H,W = frame.shape[:2]
    ov  = frame.copy()
    g,gc,gt = grade(final_sc)

    px,py,pw,ph = W//2-255, 25, 510, H-50
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(12,12,22),-1)
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(70,70,110),2)

    # Header
    cv2.putText(ov,"PHASE 1: GREETING — RESULT",
                (px+25,py+32),cv2.FONT_HERSHEY_DUPLEX,0.72,(200,200,255),2)
    cv2.line(ov,(px+15,py+42),(px+pw-15,py+42),(60,60,100),1)

    # Grade circle
    cv2.circle(ov,(px+pw-48,py+62),34,(20,20,35),-1)
    cv2.circle(ov,(px+pw-48,py+62),34,gc,2)
    cv2.putText(ov,g,(px+pw-63,py+76),cv2.FONT_HERSHEY_DUPLEX,1.5,gc,3)

    # Final score
    cv2.putText(ov,f"Final Score: {final_sc:.0f}%  —  {gt}",
                (px+20,py+68),cv2.FONT_HERSHEY_SIMPLEX,0.70,(255,255,255),2)

    y = py+100

    # ── EYE section ──────────────────────────────────────
    cv2.putText(ov,"EYE TRACKING  (ESP32-CAM)",(px+20,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,180,220),1); y+=8
    draw_bar(ov,px+20,y,   pw-40,14,att_pct,(0,200,80), "Attentive");  y+=30
    draw_bar(ov,px+20,y,   pw-40,14,dis_pct,(0,80,220), "Distracted"); y+=30
    draw_bar(ov,px+20,y,   pw-40,14,err_pct,(0,140,255),"Error");      y+=30
    cv2.putText(ov,f"Eye Score: {eye_sc:.0f}%",
                (px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.56,(160,220,160),1); y+=22

    cv2.line(ov,(px+15,y),(px+pw-15,y),(50,50,80),1); y+=14

    # ── MIC section ──────────────────────────────────────
    cv2.putText(ov,"MIC RESPONSE  (PC Microphone)",(px+20,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,180,220),1); y+=22

    s_col = (0,210,80) if mic_m["sound"] else (0,60,200)
    cv2.putText(ov,f"Sound Detected : {'YES' if mic_m['sound'] else 'NO'}",
                (px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.58,s_col,2); y+=24

    d_str = f"{mic_m['delay']:.1f}s" if mic_m["delay"] is not None else "--"
    cv2.putText(ov,f"Response Delay : {d_str}",
                (px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1); y+=22

    cv2.putText(ov,f"Avg Volume     : {mic_m['avg_rms']:.0f}  Max: {mic_m['max_rms']:.0f}",
                (px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1); y+=22

    rt_col = (0,200,80) if mic_m["response_type"]=="correct" else \
             ((0,80,255) if mic_m["response_type"]=="timeout" else (0,140,255))
    cv2.putText(ov,f"Response Type  : {mic_m['response_type'].upper()}",
                (px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.58,rt_col,2); y+=24

    cv2.putText(ov,f"Mic Score      : {mic_sc:.0f}%",
                (px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.56,(160,220,160),1); y+=22

    cv2.line(ov,(px+15,y),(px+pw-15,y),(50,50,80),1); y+=14

    # Speech transcript
    cv2.putText(ov,"Speech Recognized:",(px+20,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.50,(180,180,220),1); y+=20
    # Word wrap at ~52 chars
    words=transcript.split(); buf=""
    for w in words:
        if len(buf)+len(w)+1 > 52:
            cv2.putText(ov,buf,(px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,220,180),1)
            y+=18; buf=w
        else:
            buf+=(" " if buf else "")+w
    if buf:
        cv2.putText(ov,buf,(px+20,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,220,180),1); y+=18

    cv2.putText(ov,"Press Q to quit",
                (px+20,py+ph-14),cv2.FONT_HERSHEY_SIMPLEX,0.46,(110,110,140),1)

    return cv2.addWeighted(frame,0.20,ov,1.0,0)


# ─────────────────────────────────────────────────────────
# TERMINAL RESULT
# ─────────────────────────────────────────────────────────
def print_result(mic_m, eye_sc, mic_sc, final_sc, att, dis, err, transcript):
    g,_,gt = grade(final_sc)
    print("\n" + "="*55)
    print("  PHASE 1: GREETING — SCORE REPORT")
    print("="*55)
    print("  EYE TRACKING (ESP32-CAM)")
    print(f"    Attentive  : {att:.1f}%")
    print(f"    Distracted : {dis:.1f}%")
    print(f"    Error      : {err:.1f}%")
    print(f"    Eye Score  : {eye_sc:.1f}%")
    print("  MIC RESPONSE")
    print(f"    Sound      : {'YES' if mic_m['sound'] else 'NO'}")
    d = f"{mic_m['delay']:.1f}s" if mic_m['delay'] else "--"
    print(f"    Delay      : {d}")
    print(f"    Avg Volume : {mic_m['avg_rms']:.0f}")
    print(f"    Max Volume : {mic_m['max_rms']:.0f}")
    print(f"    Type       : {mic_m['response_type'].upper()}")
    print(f"    Mic Score  : {mic_sc:.1f}%")
    print(f"    Transcript : {transcript}")
    print("  " + "─"*51)
    print(f"  FINAL SCORE  : {final_sc:.1f}%   Grade: {g} — {gt}")
    print("="*55 + "\n")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    # Init pygame audio
    pygame.mixer.init()

    # Connect ESP32-CAM
    cam = ESP32Stream(ESP32_STREAM_URL)

    # FaceMesh
    mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    cal         = Calibration()
    mic_rec     = MicRecorder()
    phase       = PH_CAL
    cal_start   = time.time()
    listen_start= 0.0
    fps         = 0.0; fps_t=time.time(); fcount=0

    # Eye counters
    att_f=dis_f=err_f=total_f=0
    blink_ctr=blinks=0; blink_flag=False
    noface_ctr=0
    ema_score=50.0; median_buf=deque(maxlen=SIG_MED)
    att_label="ATTENTIVE"; prev_label="ATTENTIVE"

    # Result cache
    result_frame = None

    print(f"[INFO] Calibrating for {CAL_SECS}s — child look at toy...")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            blank = np.zeros((480,640,3),dtype=np.uint8)
            cv2.putText(blank,"Connecting to ESP32-CAM...",(60,230),
                        cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,200,255),2)
            cv2.imshow(WINDOW, blank)
            if cv2.waitKey(30)&0xFF==ord('q'): break
            continue

        frame = cv2.flip(frame,1)
        FH,FW = frame.shape[:2]
        cam_mat = np.array([[FW,0,FW/2],[0,FW,FH/2],[0,0,1]],dtype=np.float64)
        now = time.time()

        fcount+=1
        if fcount%15==0: fps=15/(now-fps_t+1e-6); fps_t=now

        # FaceMesh
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)
        face_ok = False
        ear=gh=gv=yaw=pitch=roll=0.0

        if results.multi_face_landmarks:
            face_ok=True; noface_ctr=0
            lms=results.multi_face_landmarks[0].landmark
            ear=(calc_ear(lms,LEFT_EAR_IDX,FW,FH)+
                 calc_ear(lms,RIGHT_EAR_IDX,FW,FH))/2
            gh_l,gv_l=calc_iris_ratio(lms,LEFT_EYE_IDX,LEFT_IRIS_IDX,FW,FH)
            gh_r,gv_r=calc_iris_ratio(lms,RIGHT_EYE_IDX,RIGHT_IRIS_IDX,FW,FH)
            gh=(gh_l+gh_r)/2; gv=(gv_l+gv_r)/2
            yaw,pitch,roll=calc_head_pose(lms,FW,FH,cam_mat)

            # Draw mesh
            mp_draw.draw_landmarks(frame,results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
            mp_draw.draw_landmarks(frame,results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
            # Iris dots
            for idxl,col in [(LEFT_IRIS_IDX,(0,255,255)),(RIGHT_IRIS_IDX,(0,200,255))]:
                pts=[(int(lms[i].x*FW),int(lms[i].y*FH)) for i in idxl]
                cx_=int(np.mean([p[0] for p in pts]))
                cy_=int(np.mean([p[1] for p in pts]))
                r_=int(dist.euclidean(pts[0],pts[2])/2)+4
                cv2.circle(frame,(cx_,cy_),r_,col,2)
                cv2.circle(frame,(cx_,cy_),2,col,-1)
        else:
            noface_ctr+=1

        # ══════════════════════════════════════════════════
        # PHASE: CALIBRATING
        # ══════════════════════════════════════════════════
        if phase == PH_CAL:
            elapsed = now - cal_start
            remain  = max(0, CAL_SECS - elapsed)
            if face_ok: cal.add(ear, yaw, pitch, gh, gv)

            # Progress bar at bottom
            cv2.rectangle(frame,(0,FH-58),(FW,FH),(10,10,20),-1)
            bw=int((FW-40)*np.clip(elapsed/CAL_SECS,0,1))
            cv2.rectangle(frame,(20,FH-40),(20+bw,FH-22),(0,180,255),-1)
            cv2.rectangle(frame,(20,FH-40),(FW-20,FH-22),(70,70,80),1)
            cv2.putText(frame,f"CALIBRATING — child look at toy — {remain:.1f}s",
                        (28,FH-46),cv2.FONT_HERSHEY_SIMPLEX,0.54,(0,200,255),1)
            cv2.putText(frame,f"FPS:{fps:.0f}",(10,22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.44,(120,120,120),1)

            if elapsed >= CAL_SECS:
                cal.apply()
                # Play greeting in a background thread so camera keeps running
                phase = PH_PLAY
                def _play():
                    play_audio_blocking(GREETING_AUDIO)
                threading.Thread(target=_play, daemon=True).start()
                print("[INFO] Greeting audio started. Camera still running...")

        # ══════════════════════════════════════════════════
        # PHASE: PLAYING  — audio playing, camera watching
        # ══════════════════════════════════════════════════
        elif phase == PH_PLAY:
            # Eye tracking continues during playback
            if face_ok:
                hs=head_score_fn(yaw,pitch); gs=gaze_score_fn(gh,gv)
                es=ear_score_fn(ear)
                raw=(W_HEAD*hs+W_GAZE*gs+W_EAR*es)*100.0
            else:
                raw=0.0

            ema_score=EMA_ALPHA*raw+(1-EMA_ALPHA)*ema_score
            median_buf.append(ema_score); smooth=float(np.median(median_buf))
            if prev_label=="ATTENTIVE": att_label="DISTRACTED" if smooth<HYST_LOW else "ATTENTIVE"
            else:                       att_label="ATTENTIVE"  if smooth>HYST_HIGH else "DISTRACTED"
            prev_label=att_label
            total_f+=1
            if not face_ok: err_f+=1
            elif att_label=="ATTENTIVE": att_f+=1
            else: dis_f+=1

            # HUD
            lc=(0,230,80) if att_label=="ATTENTIVE" else (0,80,255)
            if face_ok:
                gd="RIGHT" if gh<0.38 else("LEFT" if gh>0.62 else"CENTER")
                gd+=" UP" if gv<0.38 else(" DOWN" if gv>0.62 else"")
                for i,(t,c) in enumerate([
                    (f"EAR  :{ear:.3f}",(0,255,100) if ear>=EAR_BLINK_RATIO else (0,60,255)),
                    (f"Yaw  :{yaw:+.1f}",(255,200,0)),
                    (f"Pitch:{pitch:+.1f}",(255,200,0)),
                    (f"Gaze :{gd}",(0,200,255)),
                ]):
                    cv2.putText(frame,t,(10,33+i*24),cv2.FONT_HERSHEY_SIMPLEX,0.54,c,2)
                draw_gaze_dot(frame,gh,gv,FW-58,58)
            cv2.putText(frame,"▶ PLAYING GREETING",(FW//2-130,FH//2-10),
                        cv2.FONT_HERSHEY_DUPLEX,0.72,(0,200,100),2)
            cv2.putText(frame,att_label,(FW-195,30),cv2.FONT_HERSHEY_DUPLEX,0.72,lc,2)
            draw_bar(frame,10,FH-35,FW-20,14,smooth,
                     (0,220,80) if att_label=="ATTENTIVE" else (0,60,220),"Attention")

            # Switch to LISTEN when audio finishes
            if not pygame.mixer.music.get_busy():
                phase        = PH_LISTEN
                listen_start = time.time()
                mic_rec.start()
                print(f"[INFO] Listening for {LISTEN_SECS}s ...")

        # ══════════════════════════════════════════════════
        # PHASE: LISTENING — mic + camera both running
        # ══════════════════════════════════════════════════
        elif phase == PH_LISTEN:
            elapsed = now - listen_start
            remain  = max(0, LISTEN_SECS - elapsed)

            # Eye scoring
            if face_ok:
                if ear<EAR_BLINK_RATIO: blink_ctr+=1
                else:
                    if blink_ctr>=EAR_CONSEC: blinks+=1; blink_flag=True
                    blink_ctr=0
                bpen=0.65 if blink_flag else 1.0; blink_flag=False
                hs=head_score_fn(yaw,pitch); gs=gaze_score_fn(gh,gv); es=ear_score_fn(ear)
                raw=(W_HEAD*hs+W_GAZE*gs+W_EAR*es+W_BLINK*bpen)*100.0
            else:
                raw=0.0

            ema_score=EMA_ALPHA*raw+(1-EMA_ALPHA)*ema_score
            median_buf.append(ema_score); smooth=float(np.median(median_buf))
            if prev_label=="ATTENTIVE": att_label="DISTRACTED" if smooth<HYST_LOW else "ATTENTIVE"
            else:                       att_label="ATTENTIVE"  if smooth>HYST_HIGH else "DISTRACTED"
            prev_label=att_label
            total_f+=1
            if not face_ok or noface_ctr>NOFACE_LIMIT: err_f+=1
            elif att_label=="ATTENTIVE": att_f+=1
            else: dis_f+=1

            # HUD
            lc=(0,230,80) if att_label=="ATTENTIVE" else (0,80,255)
            if face_ok:
                gd="RIGHT" if gh<0.38 else("LEFT" if gh>0.62 else"CENTER")
                gd+=" UP" if gv<0.38 else(" DOWN" if gv>0.62 else"")
                for i,(t,c) in enumerate([
                    (f"EAR  :{ear:.3f}",(0,255,100) if ear>=EAR_BLINK_RATIO else (0,60,255)),
                    (f"Yaw  :{yaw:+.1f}",(255,200,0)),
                    (f"Pitch:{pitch:+.1f}",(255,200,0)),
                    (f"Gaze :{gd}",(0,200,255)),
                    (f"Blinks:{blinks}",(180,180,255)),
                ]):
                    cv2.putText(frame,t,(10,33+i*24),cv2.FONT_HERSHEY_SIMPLEX,0.54,c,2)
                draw_gaze_dot(frame,gh,gv,FW-58,58)
            else:
                cv2.putText(frame,"NO FACE",(FW//2-70,FH//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),2)

            cv2.putText(frame,att_label,(FW-195,30),cv2.FONT_HERSHEY_DUPLEX,0.72,lc,2)
            draw_bar(frame,10,FH-60,FW-20,14,smooth,
                     (0,220,80) if att_label=="ATTENTIVE" else (0,60,220),"Attention")

            # Live mic volume bar
            cur_rms = mic_rec.current_rms()
            vol_pct = min(100, cur_rms/30)
            vc = (0,200,80) if cur_rms>MIC_THRESHOLD else (80,80,80)
            draw_bar(frame,10,FH-35,FW-20,14,vol_pct,vc,"Mic Volume")

            # Countdown
            cv2.putText(frame,f"🎤 LISTENING... {remain:.0f}s",
                        (FW//2-155,FH//2-10),cv2.FONT_HERSHEY_DUPLEX,0.72,(0,255,150),2)

            # Done
            if elapsed >= LISTEN_SECS:
                mic_rec.stop()
                mic_m = mic_rec.metrics()

                print("[INFO] Transcribing speech...")
                transcript = mic_rec.transcribe()

                tf  = max(total_f,1)
                att = 100*att_f/tf
                dis = 100*dis_f/tf
                err = 100*err_f/tf

                eye_sc   = compute_eye_score(att_f, total_f)
                mic_sc   = compute_mic_score(mic_m)
                final_sc = compute_final(eye_sc, mic_sc)

                print_result(mic_m, eye_sc, mic_sc, final_sc, att, dis, err, transcript)

                result_frame = draw_result_overlay(
                    frame.copy(), mic_m, eye_sc, mic_sc, final_sc,
                    att, dis, err, transcript
                )
                phase = PH_RESULT

        # ══════════════════════════════════════════════════
        # PHASE: RESULT — show frozen score screen
        # ══════════════════════════════════════════════════
        elif phase == PH_RESULT:
            frame = result_frame.copy()

        # ── Show ─────────────────────────────────────────
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    mic_rec.stop()
    cam.release()
    mesh.close()
    pygame.mixer.quit()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()