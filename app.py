import streamlit as st
import cv2
import numpy as np
import time
import math
import os
import threading
from collections import deque

from dotenv import load_dotenv
load_dotenv()

from scipy.signal import butter, filtfilt

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

try:
    import requests
except Exception:
    requests = None

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "drowsiness_model.h5")
SOUND_PATH = os.getenv("SOUND_PATH", "alert.wav")

EYE_CLOSED_EAR = 0.20
DROWSY_CLOSED_SECONDS = 2.0
SMS_DELAY = 10
ALERT_SOUND_DELAY = 5

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

HR_LOW = 45.0
HR_HIGH = 150.0
RPPG_MIN_SAMPLES = 180
MIN_SNR = 1.15
GOOD_SIGNAL_SNR = 2.0
FAIR_SIGNAL_SNR = 1.35

FACE_OVAL = [
    10, 338, 297, 332, 284, 251,
    389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109, 10
]

# ============================================================
# TWILIO
# ============================================================

client = None
if TWILIO_AVAILABLE:
    try:
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
    except Exception:
        client = None

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    if tf is None or not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception:
        return None

model = load_model()

# ============================================================
# MEDIAPIPE
# ============================================================

@st.cache_resource
def load_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    )

face_mesh = load_face_mesh()

# ============================================================
# HELPERS
# ============================================================

def point_xy(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def eye_ratio(landmarks, eye_indices, w, h):
    pts = np.array([point_xy(landmarks[i], w, h) for i in eye_indices], dtype=np.float32)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    if hz < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * hz))

def eye_closed_old(lm, eye):
    return abs(lm[eye[1]].y - lm[eye[5]].y) < 0.012

def predict_drowsy(frame):
    if model is None:
        return False
    try:
        img = cv2.resize(frame, (64, 64)) / 255.0
        img = img.reshape(1, 64, 64, 3)
        prob = float(model.predict(img, verbose=0)[0][0])
        return prob > 0.95
    except Exception:
        return False

def get_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5).json()
        return "https://maps.google.com/?q=" + res["loc"]
    except Exception:
        return "Location not available"

def send_sms(frame):
    if client is None:
        return False, "TWILIO NOT CONFIGURED"
    try:
        cv2.imwrite("driver.jpg", frame)
        location = get_location()
        client.messages.create(
            body=f"🚨 EMERGENCY!\nDriver unsafe.\nLocation:\n{location}",
            from_=os.getenv("TWILIO_NUMBER"),
            to=os.getenv("ALERT_PHONE")
        )
        return True, "SMS SENT ✅"
    except Exception as e:
        return False, f"SMS ERROR: {e}"

def draw_overlay(frame, landmarks):
    h, w = frame.shape[:2]
    pts = np.array([point_xy(landmarks[i], w, h) for i in FACE_OVAL], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=(75, 180, 205), thickness=1, lineType=cv2.LINE_AA)
    for eye in (LEFT_EYE, RIGHT_EYE):
        ep = np.array([point_xy(landmarks[i], w, h) for i in eye], dtype=np.int32)
        cv2.polylines(frame, [ep], isClosed=True, color=(70, 220, 150), thickness=2, lineType=cv2.LINE_AA)
    for iris in (LEFT_IRIS, RIGHT_IRIS):
        ip = np.array([point_xy(landmarks[i], w, h) for i in iris], dtype=np.float32)
        center = np.mean(ip, axis=0).astype(int)
        radius = int(max(2, np.mean(np.linalg.norm(ip - center, axis=1))))
        cv2.circle(frame, tuple(center), radius, (70, 190, 255), 1, cv2.LINE_AA)

# ============================================================
# rPPG
# ============================================================

class RPPGProcessor:
    def __init__(self):
        self.samples = deque(maxlen=900)
        self.times   = deque(maxlen=900)
        self.last_hr = None
        self.last_quality = "WAITING"
        self.last_snr = 0.0
        self.last_update = 0.0
        self.stable_hr = None
        self.candidate_history = deque(maxlen=5)

    def add(self, rgb, ts):
        self.samples.append(np.asarray(rgb, dtype=np.float64))
        self.times.append(float(ts))
        cutoff = ts - 20.0
        while self.times and self.times[0] < cutoff:
            self.times.popleft(); self.samples.popleft()

    def estimate(self):
        now = time.time()
        if now - self.last_update < 1.5:
            return self.last_hr, self.last_quality, self.last_snr
        self.last_update = now
        if len(self.samples) < RPPG_MIN_SAMPLES:
            self.last_quality = "COLLECTING"
            return self.last_hr, self.last_quality, self.last_snr
        t   = np.asarray(self.times,   dtype=np.float64)
        rgb = np.asarray(self.samples, dtype=np.float64)
        if t[-1] - t[0] < 8.0:
            self.last_quality = "COLLECTING"
            return self.last_hr, self.last_quality, self.last_snr
        dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) < 20:
            self.last_quality = "WEAK"; return self.last_hr, self.last_quality, self.last_snr
        fs = 1.0 / np.median(dt)
        if fs < 12:
            self.last_quality = "LOW FPS"; return self.last_hr, self.last_quality, self.last_snr
        rgb = rgb[np.all(np.isfinite(rgb), axis=1)]
        if len(rgb) < RPPG_MIN_SAMPLES:
            self.last_quality = "WEAK"; return self.last_hr, self.last_quality, self.last_snr
        mean_rgb = np.maximum(np.mean(rgb, axis=0), 1e-6)
        C  = rgb / mean_rgb - 1.0
        s1 = C[:,1] - C[:,2]
        s2 = C[:,1] + C[:,2] - 2.0*C[:,0]
        std1, std2 = np.std(s1), np.std(s2)
        if std1 < 1e-6 or std2 < 1e-6:
            self.last_hr = None; self.last_quality = "NO SIGNAL"; self.last_snr = 0.0
            return self.last_hr, self.last_quality, self.last_snr
        pulse = s1 + (std1/std2)*s2
        pulse -= np.mean(pulse)
        ps = np.std(pulse)
        if ps < 1e-6:
            self.last_hr = None; self.last_quality = "NO SIGNAL"; self.last_snr = 0.0
            return self.last_hr, self.last_quality, self.last_snr
        pulse /= ps
        low, high, nyq = 0.70, 2.50, fs/2.0
        if nyq <= high:
            self.last_quality = "LOW FPS"; return self.last_hr, self.last_quality, self.last_snr
        try:
            b, a = butter(3, [low/nyq, high/nyq], btype="band")
            padlen = 3*max(len(a), len(b))
            if len(pulse) <= padlen+5:
                self.last_quality = "COLLECTING"; return self.last_hr, self.last_quality, self.last_snr
            filtered = filtfilt(b, a, pulse)
        except Exception:
            self.last_quality = "FILTER ERROR"; return self.last_hr, self.last_quality, self.last_snr
        n = len(filtered)
        spectrum = np.abs(np.fft.rfft(filtered*np.hanning(n), n=n*8))
        freqs    = np.fft.rfftfreq(n*8, d=1.0/fs)
        valid    = (freqs >= low) & (freqs <= high)
        if not np.any(valid):
            return self.last_hr, "NO SIGNAL", 0.0
        mag = spectrum[valid]; f = freqs[valid]
        k   = int(np.argmax(mag))
        peak_mag = float(mag[k])
        noise_mask = np.ones(len(mag), dtype=bool)
        noise_mask[max(0,k-12):min(len(mag),k+13)] = False
        noise = float(np.median(mag[noise_mask])) if np.any(noise_mask) else 1e-6
        snr   = peak_mag / max(noise, 1e-6)
        quality = "GOOD" if snr >= GOOD_SIGNAL_SNR else ("FAIR" if snr >= FAIR_SIGNAL_SNR else "WEAK")
        self.last_snr = snr
        if snr < MIN_SNR:
            self.last_quality = quality; return self.last_hr, self.last_quality, self.last_snr
        peak_f = float(f[k])
        candidate = peak_f * 60.0
        if not math.isfinite(candidate) or not (HR_LOW <= candidate <= HR_HIGH):
            self.last_quality = "NO SIGNAL"; return self.last_hr, self.last_quality, self.last_snr
        self.candidate_history.append(candidate)
        consensus = float(np.median(self.candidate_history))
        if self.stable_hr is None:
            self.stable_hr = consensus
        else:
            jump = abs(consensus - self.stable_hr)
            if jump <= 6.0:   self.stable_hr = 0.20*consensus + 0.80*self.stable_hr
            elif jump <= 12.0: self.stable_hr = 0.08*consensus + 0.92*self.stable_hr
        self.last_hr = self.stable_hr
        self.last_quality = quality
        return self.last_hr, self.last_quality, self.last_snr

def get_rppg_rois(frame, landmarks):
    h, w = frame.shape[:2]
    def p(idx): return point_xy(landmarks[idx], w, h)
    forehead, left_cheek, right_cheek = p(10), p(234), p(454)
    rois = []
    def add(cx, cy, rw, rh):
        x1,y1 = max(0,int(cx-rw/2)), max(0,int(cy-rh/2))
        x2,y2 = min(w,int(cx+rw/2)), min(h,int(cy+rh/2))
        if x2>x1 and y2>y1: rois.append((x1,y1,x2,y2))
    add(forehead[0],    forehead[1]+18,    55, 28)
    add(left_cheek[0]+18,  left_cheek[1]+8,  45, 38)
    add(right_cheek[0]-18, right_cheek[1]+8, 45, 38)
    return rois

def rppg_rgb_sample(frame, rois):
    values = []
    for x1,y1,x2,y2 in rois:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: continue
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)
        ch = []
        for c in range(3):
            col = rgb[:,c]
            lo,hi = np.percentile(col,[10,90])
            clean = col[(col>=lo)&(col<=hi)]
            ch.append(np.median(clean) if len(clean)>10 else np.median(col))
        values.append(ch)
    if not values: return None
    return np.mean(np.asarray(values, dtype=np.float64), axis=0)

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="VisionGuard", page_icon="🚗", layout="wide")

st.markdown("""
<style>
body { background-color: #0d0e10; }
.status-green  { color: #37dc41; font-size: 2rem; font-weight: bold; }
.status-orange { color: #ffa500; font-size: 2rem; font-weight: bold; }
.status-red    { color: #eb3b3b; font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🚗 VisionGuard — AI Driver Monitoring System")

col1, col2 = st.columns([2, 1])

with col2:
    status_box    = st.empty()
    drowsy_box    = st.empty()
    hr_box        = st.empty()
    ear_box       = st.empty()
    sms_box       = st.empty()
    stop_alarm_btn = st.button("🔕 Stop Alarm (A)")
    emergency_btn  = st.button("🚨 Emergency SMS (E)")

with col1:
    frame_box = st.empty()

# ============================================================
# SESSION STATE
# ============================================================

if "eye_start"         not in st.session_state: st.session_state.eye_start         = None
if "alert_start"       not in st.session_state: st.session_state.alert_start       = None
if "sms_sent"          not in st.session_state: st.session_state.sms_sent          = False
if "sms_status"        not in st.session_state: st.session_state.sms_status        = "SMS READY"
if "rppg"              not in st.session_state: st.session_state.rppg              = RPPGProcessor()
if "alarm_active"      not in st.session_state: st.session_state.alarm_active      = False

rppg_processor = st.session_state.rppg

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ Camera could not be opened.")
    st.stop()

EYE_THRESHOLD = 2

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now   = time.time()
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        left_ear = right_ear = 0.30
        drowsy_eye = False
        bpm = rppg_quality = None
        rppg_snr = 0.0

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            draw_overlay(frame, lm)

            left_closed  = eye_closed_old(lm, LEFT_EYE)
            right_closed = eye_closed_old(lm, RIGHT_EYE)

            if left_closed or right_closed:
                if st.session_state.eye_start is None:
                    st.session_state.eye_start = now
                if now - st.session_state.eye_start >= EYE_THRESHOLD:
                    drowsy_eye = True
            else:
                st.session_state.eye_start = None

            left_ear  = eye_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_ratio(lm, RIGHT_EYE, w, h)

            rois      = get_rppg_rois(frame, lm)
            rgb_val   = rppg_rgb_sample(frame, rois)
            if rgb_val is not None:
                rppg_processor.add(rgb_val, now)
                bpm, rppg_quality, rppg_snr = rppg_processor.estimate()

        model_drowsy = predict_drowsy(frame)
        final_drowsy = drowsy_eye or model_drowsy

        display_bpm  = bpm if bpm is not None else rppg_processor.last_hr
        hr_quality   = rppg_quality if rppg_quality else rppg_processor.last_quality
        signal_snr   = rppg_snr if rppg_snr else rppg_processor.last_snr

        if display_bpm is None:
            hr_state = "NORMAL"
        elif 60 <= display_bpm <= 100:
            hr_state = "NORMAL"
        elif (50 <= display_bpm < 60) or (100 < display_bpm <= 110):
            hr_state = "WARNING"
        else:
            hr_state = "CRITICAL"

        if final_drowsy and hr_state == "CRITICAL":
            final_status = "RED"
        elif final_drowsy or hr_state == "WARNING":
            final_status = "ORANGE"
        else:
            final_status = "GREEN"

        # Alert + SMS
        if drowsy_eye:
            if st.session_state.alert_start is None:
                st.session_state.alert_start = now
                st.session_state.sms_sent    = False
            elapsed = now - st.session_state.alert_start
            if elapsed >= SMS_DELAY and not st.session_state.sms_sent:
                sent, msg = send_sms(frame)
                st.session_state.sms_status = msg
                if sent:
                    st.session_state.sms_sent = True
        else:
            st.session_state.alert_start = None
            st.session_state.sms_sent    = False
            st.session_state.sms_status  = "SMS READY"

        if stop_alarm_btn:
            st.session_state.alert_start = None
            st.session_state.sms_sent    = False
            st.session_state.sms_status  = "SMS READY"

        if emergency_btn:
            sent, msg = send_sms(frame)
            st.session_state.sms_status = msg

        # UI Update
        color_map = {"GREEN": "green", "ORANGE": "orange", "RED": "red"}
        status_box.markdown(f'<div class="status-{color_map[final_status]}">● {final_status}</div>', unsafe_allow_html=True)
        drowsy_box.metric("Drowsiness", "DETECTED 😴" if final_drowsy else "NORMAL 😊")
        hr_text = f"{int(round(display_bpm))} BPM" if display_bpm else "Collecting..."
        hr_box.metric("Heart Rate", hr_text, delta=hr_quality)
        ear_box.metric("EAR", f"L:{left_ear:.2f}  R:{right_ear:.2f}")
        sms_box.info(st.session_state.sms_status)

        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

finally:
    cap.release()
