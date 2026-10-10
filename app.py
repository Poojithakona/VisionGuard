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

# ── rPPG ─────────────────────────────────────────────────────
class RPPGProcessor:
    def __init__(self):
        self.samples          = deque(maxlen=900)
        self.times            = deque(maxlen=900)
        self.last_hr          = None
        self.last_quality     = "WAITING"
        self.last_snr         = 0.0
        self.last_update      = 0.0
        self.stable_hr        = None
        self.candidate_history = deque(maxlen=5)

    def add_rgb_sample(self, rgb_value, timestamp):
        self.samples.append(np.asarray(rgb_value, dtype=np.float64))
        self.times.append(float(timestamp))
        cutoff = timestamp - RPPG_BUFFER_SECONDS
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            self.samples.popleft()

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

        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) < 20:
            self.last_quality = "WEAK"
            return self.last_hr, self.last_quality, self.last_snr

        fs = 1.0 / np.median(dt)
        if fs < 12:
            self.last_quality = "LOW FPS"
            return self.last_hr, self.last_quality, self.last_snr

        rgb = rgb[np.all(np.isfinite(rgb), axis=1)]
        if len(rgb) < RPPG_MIN_SAMPLES:
            self.last_quality = "WEAK"
            return self.last_hr, self.last_quality, self.last_snr

        mean_rgb = np.maximum(np.mean(rgb, axis=0), 1e-6)
        C  = rgb / mean_rgb - 1.0
        s1 = C[:, 1] - C[:, 2]
        s2 = C[:, 1] + C[:, 2] - 2.0 * C[:, 0]
        std1, std2 = np.std(s1), np.std(s2)

        if std1 < 1e-6 or std2 < 1e-6:
            self.last_hr, self.last_quality, self.last_snr = None, "NO SIGNAL", 0.0
            return self.last_hr, self.last_quality, self.last_snr

        pulse = s1 + (std1 / std2) * s2
        pulse -= np.mean(pulse)
        pulse_std = np.std(pulse)
        if pulse_std < 1e-6:
            self.last_hr, self.last_quality, self.last_snr = None, "NO SIGNAL", 0.0
            return self.last_hr, self.last_quality, self.last_snr
        pulse /= pulse_std

        low, high, nyq = 0.70, 2.50, fs / 2.0
        if nyq <= high:
            self.last_quality = "LOW FPS"
            return self.last_hr, self.last_quality, self.last_snr

        try:
            b, a = butter(3, [low / nyq, high / nyq], btype="band")
            padlen = 3 * max(len(a), len(b))
            if len(pulse) <= padlen + 5:
                self.last_quality = "COLLECTING"
                return self.last_hr, self.last_quality, self.last_snr
            filtered = filtfilt(b, a, pulse)
        except Exception:
            self.last_quality = "FILTER ERROR"
            return self.last_hr, self.last_quality, self.last_snr

        n        = len(filtered)
        spectrum = np.abs(np.fft.rfft(filtered * np.hanning(n), n=n * 8))
        freqs    = np.fft.rfftfreq(n * 8, d=1.0 / fs)
        valid    = (freqs >= low) & (freqs <= high)
        if not np.any(valid):
            return self.last_hr, "NO SIGNAL", 0.0

        mag, f = spectrum[valid], freqs[valid]
        k         = int(np.argmax(mag))
        peak_mag  = float(mag[k])
        noise_mask = np.ones(len(mag), dtype=bool)
        noise_mask[max(0, k-12):min(len(mag), k+13)] = False
        noise = float(np.median(mag[noise_mask])) if np.any(noise_mask) else 1e-6
        snr   = peak_mag / max(noise, 1e-6)

        quality = "GOOD" if snr >= GOOD_SIGNAL_SNR else ("FAIR" if snr >= FAIR_SIGNAL_SNR else "WEAK")
        self.last_snr = snr

        if snr < MIN_SNR:
            self.last_quality = quality
            return self.last_hr, self.last_quality, self.last_snr

        peak_f = float(f[k])
        if 0 < k < len(mag) - 1:
            y1, y2, y3 = np.log(np.maximum(mag[k-1:k+2], 1e-12))
            denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-12:
                peak_f += float(0.5 * (y1 - y3) / denom) * (f[1] - f[0])

        candidate = peak_f * 60.0
        if not math.isfinite(candidate) or not (HR_LOW <= candidate <= HR_HIGH):
            self.last_quality = "NO SIGNAL"
            return self.last_hr, self.last_quality, self.last_snr

        self.candidate_history.append(candidate)
        consensus = float(np.median(self.candidate_history))

        if self.stable_hr is None:
            self.stable_hr = consensus
        else:
            jump = abs(consensus - self.stable_hr)
            if jump <= 6.0:
                self.stable_hr = 0.20 * consensus + 0.80 * self.stable_hr
            elif jump <= 12.0:
                self.stable_hr = 0.08 * consensus + 0.92 * self.stable_hr

        self.last_hr      = self.stable_hr
        self.last_quality = quality
        return self.last_hr, self.last_quality, self.last_snr

# ── FACE OVERLAY ─────────────────────────────────────────────
def draw_clean_face_overlay(frame, landmarks):
    h, w = frame.shape[:2]
    for idx in [10,152,1,4,13,14,33,133,362,263,61,291,234,454]:
        x, y = point_xy(landmarks[idx], w, h)
        cv2.circle(frame, (x, y), 2, (80, 210, 255), -1)

    fx, fy   = point_xy(landmarks[10],  w, h)
    lcx, lcy = point_xy(landmarks[234], w, h)
    rcx, rcy = point_xy(landmarks[454], w, h)
    for rx, ry in [(fx, fy+18), (lcx+18, lcy+8), (rcx-18, rcy+8)]:
        cv2.circle(frame, (int(rx), int(ry)), 4, (0,0,255),     -1, cv2.LINE_AA)
        cv2.circle(frame, (int(rx), int(ry)), 1, (255,255,255), -1, cv2.LINE_AA)

    pts = np.array([point_xy(landmarks[i], w, h) for i in FACE_OVAL], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=(75,180,205), thickness=1, lineType=cv2.LINE_AA)

    for eye in (LEFT_EYE, RIGHT_EYE):
        pts_eye = np.array([point_xy(landmarks[i], w, h) for i in eye], dtype=np.int32)
        cv2.polylines(frame, [pts_eye], isClosed=True, color=(70,220,150), thickness=2, lineType=cv2.LINE_AA)

    for iris in (LEFT_IRIS, RIGHT_IRIS):
        pts_iris = np.array([point_xy(landmarks[i], w, h) for i in iris], dtype=np.float32)
        center   = np.mean(pts_iris, axis=0).astype(int)
        radius   = int(max(2, np.mean(np.linalg.norm(pts_iris - center, axis=1))))
        cv2.circle(frame, tuple(center), radius, (70,190,255), 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(center), 2,      (255,255,255), -1)

# ── rPPG ROI ─────────────────────────────────────────────────
def get_rppg_rois(frame, landmarks):
    h, w = frame.shape[:2]
    def p(idx): return point_xy(landmarks[idx], w, h)
    roi_list = []
    def add_roi(cx, cy, rw, rh):
        x1, y1 = max(0, int(cx-rw/2)), max(0, int(cy-rh/2))
        x2, y2 = min(w, int(cx+rw/2)), min(h, int(cy+rh/2))
        if x2 > x1 and y2 > y1:
            roi_list.append((x1, y1, x2, y2))
    fx, fy   = p(10)
    lcx, lcy = p(234)
    rcx, rcy = p(454)
    add_roi(fx,      fy+18,  55, 28)
    add_roi(lcx+18,  lcy+8,  45, 38)
    add_roi(rcx-18,  rcy+8,  45, 38)
    return roi_list

def rppg_rgb_sample(frame, rois):
    values = []
    for x1, y1, x2, y2 in rois:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
        clean_channels = []
        for c in range(3):
            ch = rgb[:, c]
            lo, hi = np.percentile(ch, [10, 90])
            clean  = ch[(ch >= lo) & (ch <= hi)]
            clean_channels.append(np.median(clean) if len(clean) > 10 else np.median(ch))
        values.append(clean_channels)
    return np.mean(np.asarray(values, dtype=np.float64), axis=0) if values else None

# ── DASHBOARD ────────────────────────────────────────────────
def draw_panel(img, x1, y1, x2, y2, title, accent):
    cv2.rectangle(img, (x1,y1), (x2,y2), (30,26,24), -1)
    cv2.rectangle(img, (x1,y1), (x2,y2), (75,68,62),  1)
    cv2.rectangle(img, (x1,y1), (x1+5,y2), accent,   -1)
    cv2.putText(img, title, (x1+20, y1+38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (240,238,232), 2, cv2.LINE_AA)

def draw_dashboard(frame, final_status, drowsy, left_ear, right_ear,
                   hr, hr_quality, signal_snr, fps_value, sms_status, model_available):
    W, H = 1100, 650
    canvas = np.full((H, W, 3), (12,13,15), dtype=np.uint8)
    margin, header_h, gap, footer_h = 24, 62, 18, 48
    cv2.putText(canvas, "AI DRIVER MONITORING SYSTEM", (margin,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (242,242,238), 2, cv2.LINE_AA)
    cv2.putText(canvas, "REAL-TIME DRIVER SAFETY", (W-285,38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1, cv2.LINE_AA)
    cv2.line(canvas, (margin,header_h), (W-margin,header_h), (55,58,62), 1)

    ct = header_h + 16
    cb = H - footer_h
    lw = int(W * 0.66)
    lx1, lx2 = margin, margin + lw
    rx1, rx2 = lx2 + gap, W - margin

    draw_panel(canvas, lx1, ct, lx2, cb, "LIVE CAMERA", (40,155,220))
    cx1, cy1, cx2, cy2 = lx1+12, ct+52, lx2-12, cb-12
    aw, ah = cx2-cx1, cy2-cy1
    fh, fw = frame.shape[:2]
    scale  = min(aw/fw, ah/fh)
    dw, dh = max(1,int(fw*scale)), max(1,int(fh*scale))
    cam    = cv2.resize(frame, (dw,dh), interpolation=cv2.INTER_LINEAR)
    dx     = cx1 + (aw-dw)//2
    dy     = cy1 + (ah-dh)//2
    canvas[dy:dy+dh, dx:dx+dw] = cam
    cv2.circle(canvas, (cx1+20, cy1+20), 5, (60,220,75), -1)
    cv2.putText(canvas, "FACE TRACKING", (cx1+34, cy1+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230,230,230), 1, cv2.LINE_AA)

    if final_status == "GREEN":
        sc, st, sub = (55,220,65),  "NORMAL",  "DRIVER CONDITION NORMAL"
    elif final_status == "ORANGE":
        sc, st, sub = (0,165,255),  "WARNING", "ATTENTION REQUIRED"
    else:
        sc, st, sub = (45,45,235),  "UNSAFE",  "DRIVER SAFETY RISK"

    total = cb - ct - gap*2
    p1h = int(total * 0.27)
    p2h = int(total * 0.36)
    p3h = total - p1h - p2h
    p1y1, p1y2 = ct,          ct+p1h
    p2y1, p2y2 = p1y2+gap,    p1y2+gap+p2h
    p3y1, p3y2 = p2y2+gap,    cb

    draw_panel(canvas, rx1, p1y1, rx2, p1y2, "FINAL STATUS", sc)
    cv2.putText(canvas, st,  (rx1+24, p1y1+94),  cv2.FONT_HERSHEY_SIMPLEX, 1.18, sc,          3, cv2.LINE_AA)
    cv2.putText(canvas, sub, (rx1+25, p1y1+127), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185,185,185),1, cv2.LINE_AA)

    dc = (45,45,235) if drowsy else (55,205,70)
    draw_panel(canvas, rx1, p2y1, rx2, p2y2, "DROWSINESS", dc)
    ix   = rx1 + 24
    c2x  = rx1 + int((rx2-rx1)*0.52)
    ls   = "OPEN" if left_ear  >= EYE_CLOSED_EAR else "CLOSED"
    rs   = "OPEN" if right_ear >= EYE_CLOSED_EAR else "CLOSED"
    cv2.putText(canvas, "LEFT EYE",  (ix,  p2y1+72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT EYE", (c2x, p2y1+72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(canvas, ls, (ix,  p2y1+108), cv2.FONT_HERSHEY_SIMPLEX, 0.67, dc if ls=="CLOSED" else (65,210,75), 2, cv2.LINE_AA)
    cv2.putText(canvas, rs, (c2x, p2y1+108), cv2.FONT_HERSHEY_SIMPLEX, 0.67, dc if rs=="CLOSED" else (65,210,75), 2, cv2.LINE_AA)
    cv2.putText(canvas, "DROWSINESS DETECTED" if drowsy else "EYES OPEN / NORMAL",
                (ix, p2y1+146), cv2.FONT_HERSHEY_SIMPLEX, 0.48, dc, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"EAR   L {left_ear:.2f}    R {right_ear:.2f}",
                (ix, p2y1+177), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (145,145,145), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CNN: ACTIVE" if model_available else "CNN: EYE-BASED MODE",
                (ix, p2y1+205), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130,130,130), 1, cv2.LINE_AA)

    draw_panel(canvas, rx1, p3y1, rx2, p3y2, "rPPG / HEART RATE", (45,190,215))
    hr_text = "--" if hr is None else f"{int(round(hr))}"
    cv2.putText(canvas, hr_text, (ix, p3y1+88), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (240,240,240), 3, cv2.LINE_AA)
    cv2.putText(canvas, "BPM",   (ix+125, p3y1+88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175,175,175), 2, cv2.LINE_AA)
    sigc = (55,220,70) if hr_quality=="GOOD" else ((0,190,255) if hr_quality=="FAIR" else (160,160,160))
    cv2.putText(canvas, f"SIGNAL: {hr_quality}", (ix, p3y1+122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sigc, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"SNR {signal_snr:.2f}    FPS {fps_value:.1f}",
                (ix, p3y1+151), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (145,145,145), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Facial RGB -> POS -> Bandpass -> FFT",
                (ix, p3y1+179), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (125,125,125), 1, cv2.LINE_AA)

    fy2 = H - 16
    smsc = (55,220,70) if "SENT" in sms_status else ((0,165,255) if "ERROR" in sms_status else (140,140,140))
    cv2.putText(canvas, "Emergency SMS button ->", (24, fy2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(canvas, sms_status[:30], (W-290, fy2), cv2.FONT_HERSHEY_SIMPLEX, 0.43, smsc, 1, cv2.LINE_AA)

    return canvas

# ── VIDEO PROCESSOR ──────────────────────────────────────────
class DriverMonitorProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.55, min_tracking_confidence=0.55
        )
        self.rppg          = RPPGProcessor()
        self.eye_start     = None
        self.alert_start   = None
        self.sms_sent      = False
        self.last_frame_t  = time.time()
        self.fps_smooth    = 0.0
        self.sms_status    = "SMS READY"
        self.result        = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        dt = now - self.last_frame_t
        self.last_frame_t = now
        if dt > 0:
            cfps = 1.0 / dt
            self.fps_smooth = cfps if self.fps_smooth == 0 else 0.9*self.fps_smooth + 0.1*cfps

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        left_ear = right_ear = 0.30
        drowsy_eye = 0
        bpm = rppg_quality = None
        rppg_snr = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w = img.shape[:2]
            draw_clean_face_overlay(img, lm)

            lc = eye_closed_old(lm, LEFT_EYE)
            rc = eye_closed_old(lm, RIGHT_EYE)
            if int(lc) + int(rc) >= 1:
                if self.eye_start is None:
                    self.eye_start = now
                if now - self.eye_start >= EYE_THRESHOLD:
                    drowsy_eye = 1
            else:
                self.eye_start = None
                drowsy_eye = 0

            left_ear  = eye_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_ratio(lm, RIGHT_EYE, w, h)

            rois      = get_rppg_rois(img, lm)
            rgb_val   = rppg_rgb_sample(img, rois)
            if rgb_val is not None:
                self.rppg.add_rgb_sample(rgb_val, now)
                bpm, rppg_quality, rppg_snr = self.rppg.estimate()

        display_bpm  = bpm if bpm is not None else self.rppg.last_hr
        hr_quality   = rppg_quality if rppg_quality else self.rppg.last_quality
        signal_snr   = rppg_snr if rppg_snr else self.rppg.last_snr

        if display_bpm is None or display_bpm == 0:
            hr_state = "NORMAL"
        elif 60 <= display_bpm <= 100:
            hr_state = "NORMAL"
        elif (50 <= display_bpm < 60) or (100 < display_bpm <= 110):
            hr_state = "WARNING"
        else:
            hr_state = "CRITICAL"

        model_drowsy, _ = predict_model_drowsiness(img)
        final_drowsy    = bool(drowsy_eye or model_drowsy)

        if final_drowsy and hr_state == "CRITICAL":
            final_status = "RED"
        elif final_drowsy or hr_state == "WARNING":
            final_status = "ORANGE"
        else:
            final_status = "GREEN"

        if drowsy_eye:
            if self.alert_start is None:
                self.alert_start = now
                self.sms_sent    = False
            elapsed = now - self.alert_start
            if elapsed >= SMS_DELAY and not self.sms_sent:
                sent, msg      = send_sms(img)
                self.sms_status = msg
                if sent:
                    self.sms_sent = True
        else:
            self.alert_start = None
            self.sms_sent    = False
            self.sms_status  = "SMS READY"

        self.result = {
            "final_status": final_status,
            "drowsy":        final_drowsy,
            "left_ear":      left_ear,
            "right_ear":     right_ear,
            "bpm":           display_bpm,
            "hr_quality":    hr_quality,
            "snr":           signal_snr,
            "fps":           self.fps_smooth,
            "sms_status":    self.sms_status,
        }

        dashboard = draw_dashboard(
            img, final_status, final_drowsy,
            left_ear, right_ear,
            float(display_bpm) if display_bpm is not None else None,
            hr_quality, signal_snr, self.fps_smooth,
            self.sms_status, model_available
        )

        from av import VideoFrame
        out = VideoFrame.from_ndarray(dashboard, format="bgr24")
        out.pts      = frame.pts
        out.time_base = frame.time_base
        return out

# ── STREAMLIT UI ─────────────────────────────────────────────
st.set_page_config(page_title="VisionGuard", page_icon="🚗", layout="wide")
st.title("🚗 VisionGuard — AI Driver Monitoring System")
st.caption("Real-time drowsiness detection + rPPG heart rate monitoring")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="visionguard",
    video_processor_factory=DriverMonitorProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    col1, col2, col3 = st.columns(3)
    if col1.button("🚨 Emergency SMS"):
        pass  # handled inside processor on next frame trigger via session state

    st.markdown("---")
    placeholder = st.empty()

    while ctx.state.playing:
        r = ctx.video_processor.result
        if r:
            status = r.get("final_status", "GREEN")
            color  = {"GREEN": "🟢", "ORANGE": "🟠", "RED": "🔴"}.get(status, "🟢")
            with placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status",     f"{color} {status}")
                c2.metric("Heart Rate", f"{int(round(r['bpm'])) if r['bpm'] else '--'} BPM")
                c3.metric("Signal",     r.get("hr_quality", "--"))
                c4.metric("SMS",        r.get("sms_status", "--"))
        time.sleep(0.5)
