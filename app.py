import streamlit as st
st.set_page_config(page_title="VisionGuard", page_icon="🚗", layout="wide")

import os
import time
import math
import threading
from collections import deque

import cv2
import numpy as np
import requests
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv
from av import VideoFrame

load_dotenv()

# ── TWILIO ───────────────────────────────────────────────────
try:
    from twilio.rest import Client
    _client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
except Exception:
    _client = None

TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
ALERT_PHONE   = os.getenv("ALERT_PHONE")

# ── MODEL ────────────────────────────────────────────────────
try:
    import tensorflow as tf
    _model = tf.keras.models.load_model("drowsiness_model.h5", compile=False) if os.path.exists("drowsiness_model.h5") else None
    _model_available = _model is not None
except Exception:
    _model = None
    _model_available = False

# ── CONSTANTS ────────────────────────────────────────────────
LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,
              397,365,379,378,400,377,152,148,176,149,150,136,
              172,58,132,93,234,127,162,21,54,103,67,109,10]
REFERENCE_INDICES = [10,152,1,4,13,14,33,133,362,263,61,291,234,454]

EYE_CLOSED_EAR    = 0.20
EYE_THRESHOLD     = 2
ALERT_SOUND_DELAY = 5
SMS_DELAY         = 10
HR_LOW, HR_HIGH   = 45.0, 150.0
RPPG_BUFFER_SECS  = 20.0
RPPG_MIN_SAMPLES  = 180
MIN_SNR           = 1.15
GOOD_SIGNAL_SNR   = 2.0
FAIR_SIGNAL_SNR   = 1.35

# ── HELPERS ──────────────────────────────────────────────────
def get_location():
    try:
        return "https://maps.google.com/?q=" + requests.get("https://ipinfo.io/json", timeout=5).json()["loc"]
    except Exception:
        return "Location not available"

def send_sms(frame):
    if _client is None:
        return False, "TWILIO NOT CONFIGURED"
    try:
        cv2.imwrite("driver.jpg", frame)
        _client.messages.create(
            body=f"🚨 EMERGENCY!\nDriver unsafe.\nLocation:\n{get_location()}",
            from_=TWILIO_NUMBER, to=ALERT_PHONE)
        return True, "SMS SENT"
    except Exception:
        return False, "SMS ERROR"

def point_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def eye_ratio(landmarks, idx, w, h):
    pts = np.array([point_xy(landmarks[i], w, h) for i in idx], dtype=np.float32)
    v1 = np.linalg.norm(pts[1]-pts[5]); v2 = np.linalg.norm(pts[2]-pts[4])
    hz = np.linalg.norm(pts[0]-pts[3])
    return float((v1+v2)/(2.0*hz)) if hz > 1e-6 else 0.0

def eye_closed_check(lm, eye):
    return abs(lm[eye[1]].y - lm[eye[5]].y) < 0.012

def predict_drowsy(frame):
    if not _model_available: return False
    try:
        img = cv2.resize(frame,(64,64))/255.0
        return float(_model.predict(img.reshape(1,64,64,3), verbose=0)[0][0]) > 0.95
    except Exception: return False

def draw_clean_face_overlay(frame, landmarks):
    h, w = frame.shape[:2]
    for idx in REFERENCE_INDICES:
        x, y = point_xy(landmarks[idx], w, h)
        cv2.circle(frame, (x,y), 2, (80,210,255), -1)
    fxy = point_xy(landmarks[10],w,h)
    lxy = point_xy(landmarks[234],w,h)
    rxy = point_xy(landmarks[454],w,h)
    for rx,ry in [(fxy[0],fxy[1]+18),(lxy[0]+18,lxy[1]+8),(rxy[0]-18,rxy[1]+8)]:
        cv2.circle(frame,(int(rx),int(ry)),4,(0,0,255),-1,cv2.LINE_AA)
        cv2.circle(frame,(int(rx),int(ry)),1,(255,255,255),-1,cv2.LINE_AA)
    pts = np.array([point_xy(landmarks[i],w,h) for i in FACE_OVAL], dtype=np.int32)
    cv2.polylines(frame,[pts],False,(75,180,205),1,cv2.LINE_AA)
    for eye in (LEFT_EYE, RIGHT_EYE):
        ep = np.array([point_xy(landmarks[i],w,h) for i in eye], dtype=np.int32)
        cv2.polylines(frame,[ep],True,(70,220,150),2,cv2.LINE_AA)
    for iris in (LEFT_IRIS, RIGHT_IRIS):
        ip = np.array([point_xy(landmarks[i],w,h) for i in iris], dtype=np.float32)
        c  = np.mean(ip,axis=0).astype(int)
        r  = int(max(2, np.mean(np.linalg.norm(ip-c,axis=1))))
        cv2.circle(frame,tuple(c),r,(70,190,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(c),2,(255,255,255),-1)

# ── rPPG PROCESSOR ───────────────────────────────────────────
class RPPGProcessor:
    def __init__(self):
        self.samples = deque(maxlen=900); self.times = deque(maxlen=900)
        self.last_hr = None; self.last_quality = "WAITING"; self.last_snr = 0.0
        self.last_update = 0.0; self.stable_hr = None
        self.candidate_history = deque(maxlen=5)

    def add_rgb_sample(self, rgb_value, timestamp):
        self.samples.append(np.asarray(rgb_value,dtype=np.float64))
        self.times.append(float(timestamp))
        cutoff = timestamp - RPPG_BUFFER_SECS
        while self.times and self.times[0] < cutoff:
            self.times.popleft(); self.samples.popleft()

    def estimate(self):
        now = time.time()
        if now - self.last_update < 1.5: return self.last_hr, self.last_quality, self.last_snr
        self.last_update = now
        if len(self.samples) < RPPG_MIN_SAMPLES:
            self.last_quality = "COLLECTING"; self.last_snr = 0.0
            return self.last_hr, self.last_quality, self.last_snr
        t = np.asarray(self.times,dtype=np.float64)
        rgb = np.asarray(self.samples,dtype=np.float64)
        if t[-1]-t[0] < 8.0:
            self.last_quality = "COLLECTING"; self.last_snr = 0.0
            return self.last_hr, self.last_quality, self.last_snr
        dt = np.diff(t); dt = dt[np.isfinite(dt)&(dt>0)]
        if len(dt) < 20:
            self.last_quality = "WEAK"; return self.last_hr, self.last_quality, self.last_snr
        fs = 1.0/np.median(dt)
        if fs < 12:
            self.last_quality = "LOW FPS"; self.last_snr = 0.0
            return self.last_hr, self.last_quality, self.last_snr
        rgb = rgb[np.all(np.isfinite(rgb),axis=1)]
        if len(rgb) < RPPG_MIN_SAMPLES:
            self.last_quality = "WEAK"; self.last_snr = 0.0
            return self.last_hr, self.last_quality, self.last_snr
        mean_rgb = np.maximum(np.mean(rgb,axis=0),1e-6); C = rgb/mean_rgb-1.0
        s1=C[:,1]-C[:,2]; s2=C[:,1]+C[:,2]-2.0*C[:,0]
        std1,std2 = np.std(s1),np.std(s2)
        if std1<1e-6 or std2<1e-6:
            self.last_hr=None; self.last_quality="NO SIGNAL"; self.last_snr=0.0
            return self.last_hr, self.last_quality, self.last_snr
        pulse = s1+(std1/std2)*s2; pulse -= np.mean(pulse); ps=np.std(pulse)
        if ps<1e-6:
            self.last_hr=None; self.last_quality="NO SIGNAL"; self.last_snr=0.0
            return self.last_hr, self.last_quality, self.last_snr
        pulse/=ps; low,high,nyq=0.70,2.50,fs/2.0
        if nyq<=high:
            self.last_quality="LOW FPS"; self.last_snr=0.0
            return self.last_hr, self.last_quality, self.last_snr
        try:
            b,a=butter(3,[low/nyq,high/nyq],btype="band")
            padlen=3*max(len(a),len(b))
            if len(pulse)<=padlen+5:
                self.last_quality="COLLECTING"; return self.last_hr, self.last_quality, self.last_snr
            filtered=filtfilt(b,a,pulse)
        except Exception:
            self.last_quality="FILTER ERROR"; self.last_snr=0.0
            return self.last_hr, self.last_quality, self.last_snr
        n=len(filtered)
        spectrum=np.abs(np.fft.rfft(filtered*np.hanning(n),n=n*8))
        freqs=np.fft.rfftfreq(n*8,d=1.0/fs)
        valid=(freqs>=low)&(freqs<=high)
        if not np.any(valid): return self.last_hr,"NO SIGNAL",0.0
        mag,f=spectrum[valid],freqs[valid]; k=int(np.argmax(mag)); peak_mag=float(mag[k])
        nm=np.ones(len(mag),dtype=bool); nm[max(0,k-12):min(len(mag),k+13)]=False
        noise=float(np.median(mag[nm])) if np.any(nm) else 1e-6
        snr=peak_mag/max(noise,1e-6)
        quality="GOOD" if snr>=GOOD_SIGNAL_SNR else ("FAIR" if snr>=FAIR_SIGNAL_SNR else "WEAK")
        self.last_snr=snr
        if snr<MIN_SNR:
            self.last_quality=quality; return self.last_hr, self.last_quality, self.last_snr
        peak_f=float(f[k])
        if 0<k<len(mag)-1:
            y1,y2,y3=np.log(np.maximum(mag[k-1:k+2],1e-12))
            denom=y1-2*y2+y3
            if abs(denom)>1e-12: peak_f+=float(0.5*(y1-y3)/denom)*(f[1]-f[0])
        candidate=peak_f*60.0
        if not math.isfinite(candidate) or not (HR_LOW<=candidate<=HR_HIGH):
            self.last_quality="NO SIGNAL"; return self.last_hr, self.last_quality, self.last_snr
        self.candidate_history.append(candidate)
        consensus=float(np.median(self.candidate_history))
        if self.stable_hr is None: self.stable_hr=consensus
        else:
            jump=abs(consensus-self.stable_hr)
            if jump<=6.0: self.stable_hr=0.20*consensus+0.80*self.stable_hr
            elif jump<=12.0: self.stable_hr=0.08*consensus+0.92*self.stable_hr
        self.last_hr=self.stable_hr; self.last_quality=quality
        return self.last_hr, self.last_quality, self.last_snr

def get_rppg_rois(frame, landmarks):
    h,w=frame.shape[:2]
    def p(i): return point_xy(landmarks[i],w,h)
    rois=[]
    def add(cx,cy,rw,rh):
        x1,y1=max(0,int(cx-rw/2)),max(0,int(cy-rh/2))
        x2,y2=min(w,int(cx+rw/2)),min(h,int(cy+rh/2))
        if x2>x1 and y2>y1: rois.append((x1,y1,x2,y2))
    fx,fy=p(10); lx,ly=p(234); rx,ry=p(454)
    add(fx,fy+18,55,28); add(lx+18,ly+8,45,38); add(rx-18,ry+8,45,38)
    return rois

def rppg_rgb_sample(frame, rois):
    values=[]
    for x1,y1,x2,y2 in rois:
        roi=frame[y1:y2,x1:x2]
        if roi.size==0: continue
        rgb=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)
        ch=[]
        for c in range(3):
            col=rgb[:,c]; lo,hi=np.percentile(col,[10,90]); clean=col[(col>=lo)&(col<=hi)]
            ch.append(np.median(clean) if len(clean)>10 else np.median(col))
        values.append(ch)
    return np.mean(np.asarray(values,dtype=np.float64),axis=0) if values else None

# ── DASHBOARD ────────────────────────────────────────────────
def fit_text(text, max_chars):
    return text if len(text)<=max_chars else text[:max_chars-1]+"…"

def draw_panel(img,x1,y1,x2,y2,title,accent):
    cv2.rectangle(img,(x1,y1),(x2,y2),(30,26,24),-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),(75,68,62),1)
    cv2.rectangle(img,(x1,y1),(x1+5,y2),accent,-1)
    cv2.putText(img,title,(x1+20,y1+38),cv2.FONT_HERSHEY_SIMPLEX,0.82,(240,238,232),2,cv2.LINE_AA)

def draw_dashboard(frame, final_status, drowsy, left_ear, right_ear,
                   hr, hr_quality, signal_snr, fps_value, sms_status):
    screen_w, screen_h = 1100, 650
    canvas = np.zeros((screen_h,screen_w,3),dtype=np.uint8)
    canvas[:] = (12,13,15)
    margin=24; header_h=62; gap=18; footer_h=48
    cv2.putText(canvas,"AI DRIVER MONITORING SYSTEM",(margin,40),cv2.FONT_HERSHEY_SIMPLEX,0.95,(242,242,238),2,cv2.LINE_AA)
    cv2.putText(canvas,"REAL-TIME DRIVER SAFETY",(screen_w-285,38),cv2.FONT_HERSHEY_SIMPLEX,0.55,(150,150,150),1,cv2.LINE_AA)
    cv2.line(canvas,(margin,header_h),(screen_w-margin,header_h),(55,58,62),1)
    content_top=header_h+16; content_bottom=screen_h-footer_h
    left_w=int(screen_w*0.66)
    left_x1=margin; left_x2=left_x1+left_w
    right_x1=left_x2+gap; right_x2=screen_w-margin
    # camera panel
    draw_panel(canvas,left_x1,content_top,left_x2,content_bottom,"LIVE CAMERA",(40,155,220))
    cam_x1=left_x1+12; cam_y1=content_top+52; cam_x2=left_x2-12; cam_y2=content_bottom-12
    aw=cam_x2-cam_x1; ah=cam_y2-cam_y1
    fh,fw=frame.shape[:2]; scale=min(aw/fw,ah/fh)
    dw=max(1,int(fw*scale)); dh=max(1,int(fh*scale))
    cam=cv2.resize(frame,(dw,dh),interpolation=cv2.INTER_LINEAR)
    dx=cam_x1+(aw-dw)//2; dy=cam_y1+(ah-dh)//2
    canvas[dy:dy+dh,dx:dx+dw]=cam
    cv2.circle(canvas,(cam_x1+20,cam_y1+20),5,(60,220,75),-1)
    cv2.putText(canvas,"FACE TRACKING",(cam_x1+34,cam_y1+25),cv2.FONT_HERSHEY_SIMPLEX,0.48,(230,230,230),1,cv2.LINE_AA)
    # status colors
    if final_status=="GREEN":
        status_color=(55,220,65); status_text="NORMAL"; sub="DRIVER CONDITION NORMAL"
    elif final_status=="ORANGE":
        status_color=(0,165,255); status_text="WARNING"; sub="ATTENTION REQUIRED"
    else:
        status_color=(45,45,235); status_text="UNSAFE"; sub="DRIVER SAFETY RISK"
    tot=content_bottom-content_top-gap*2
    p1h=int(tot*0.27); p2h=int(tot*0.36)
    p1y1=content_top; p1y2=p1y1+p1h
    p2y1=p1y2+gap;    p2y2=p2y1+p2h
    p3y1=p2y2+gap;    p3y2=content_bottom
    # final status panel
    draw_panel(canvas,right_x1,p1y1,right_x2,p1y2,"FINAL STATUS",status_color)
    cv2.putText(canvas,status_text,(right_x1+24,p1y1+94),cv2.FONT_HERSHEY_SIMPLEX,1.18,status_color,3,cv2.LINE_AA)
    cv2.putText(canvas,sub,(right_x1+25,p1y1+127),cv2.FONT_HERSHEY_SIMPLEX,0.48,(185,185,185),1,cv2.LINE_AA)
    # drowsiness panel
    drowsy_color=(45,45,235) if drowsy else (55,205,70)
    draw_panel(canvas,right_x1,p2y1,right_x2,p2y2,"DROWSINESS",drowsy_color)
    inner_x=right_x1+24; col2_x=right_x1+int((right_x2-right_x1)*0.52)
    left_state="OPEN" if left_ear>=EYE_CLOSED_EAR else "CLOSED"
    right_state="OPEN" if right_ear>=EYE_CLOSED_EAR else "CLOSED"
    cv2.putText(canvas,"LEFT EYE",(inner_x,p2y1+72),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1,cv2.LINE_AA)
    cv2.putText(canvas,"RIGHT EYE",(col2_x,p2y1+72),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1,cv2.LINE_AA)
    cv2.putText(canvas,left_state,(inner_x,p2y1+108),cv2.FONT_HERSHEY_SIMPLEX,0.67,drowsy_color if left_state=="CLOSED" else (65,210,75),2,cv2.LINE_AA)
    cv2.putText(canvas,right_state,(col2_x,p2y1+108),cv2.FONT_HERSHEY_SIMPLEX,0.67,drowsy_color if right_state=="CLOSED" else (65,210,75),2,cv2.LINE_AA)
    cv2.putText(canvas,"DROWSINESS DETECTED" if drowsy else "EYES OPEN / NORMAL",(inner_x,p2y1+146),cv2.FONT_HERSHEY_SIMPLEX,0.48,drowsy_color,1,cv2.LINE_AA)
    cv2.putText(canvas,f"EAR   L {left_ear:.2f}    R {right_ear:.2f}",(inner_x,p2y1+177),cv2.FONT_HERSHEY_SIMPLEX,0.42,(145,145,145),1,cv2.LINE_AA)
    cv2.putText(canvas,"CNN: ACTIVE" if _model_available else "CNN: EYE-BASED MODE",(inner_x,p2y1+205),cv2.FONT_HERSHEY_SIMPLEX,0.40,(130,130,130),1,cv2.LINE_AA)
    # heart rate panel
    draw_panel(canvas,right_x1,p3y1,right_x2,p3y2,"rPPG / HEART RATE",(45,190,215))
    hr_text="--" if hr is None else f"{int(round(hr))}"
    cv2.putText(canvas,hr_text,(inner_x,p3y1+88),cv2.FONT_HERSHEY_SIMPLEX,1.25,(240,240,240),3,cv2.LINE_AA)
    cv2.putText(canvas,"BPM",(inner_x+125,p3y1+88),cv2.FONT_HERSHEY_SIMPLEX,0.55,(175,175,175),2,cv2.LINE_AA)
    sigc=(55,220,70) if hr_quality=="GOOD" else ((0,190,255) if hr_quality=="FAIR" else (160,160,160))
    cv2.putText(canvas,f"SIGNAL: {hr_quality}",(inner_x,p3y1+122),cv2.FONT_HERSHEY_SIMPLEX,0.45,sigc,1,cv2.LINE_AA)
    cv2.putText(canvas,f"SNR {signal_snr:.2f}    FPS {fps_value:.1f}",(inner_x,p3y1+151),cv2.FONT_HERSHEY_SIMPLEX,0.40,(145,145,145),1,cv2.LINE_AA)
    cv2.putText(canvas,"Facial RGB -> POS -> Bandpass -> FFT",(inner_x,p3y1+179),cv2.FONT_HERSHEY_SIMPLEX,0.38,(125,125,125),1,cv2.LINE_AA)
    # footer
    footer_y=screen_h-16
    cv2.putText(canvas,"A  Stop Alarm",(24,footer_y),cv2.FONT_HERSHEY_SIMPLEX,0.48,(220,220,220),1,cv2.LINE_AA)
    cv2.putText(canvas,"E  Emergency SMS",(160,footer_y),cv2.FONT_HERSHEY_SIMPLEX,0.48,(220,220,220),1,cv2.LINE_AA)
    smsc=(55,220,70) if "SENT" in sms_status else ((0,165,255) if "ERROR" in sms_status or "FAILED" in sms_status else (140,140,140))
    cv2.putText(canvas,fit_text(sms_status,30),(screen_w-290,footer_y),cv2.FONT_HERSHEY_SIMPLEX,0.43,smsc,1,cv2.LINE_AA)
    return canvas

# ── VIDEO PROCESSOR ──────────────────────────────────────────
class DriverMonitor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh   = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.55, min_tracking_confidence=0.55)
        self.rppg        = RPPGProcessor()
        self.eye_start   = None
        self.alert_start = None
        self.sms_sent_for_event = False
        self.sms_status  = "SMS READY"
        self.fps_smooth  = 0.0
        self.last_t      = time.time()
        self.alarm_active   = False
        self.send_emergency = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        dt  = now - self.last_t; self.last_t = now
        if dt > 0:
            self.fps_smooth = (1.0/dt) if self.fps_smooth==0 else 0.90*self.fps_smooth+0.10*(1.0/dt)
        img = cv2.flip(img,1)
        res = self.face_mesh.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        left_ear=right_ear=0.30; drowsy_eye=False; bpm=rppg_quality=None; rppg_snr=0.0
        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark; h,w=img.shape[:2]
            draw_clean_face_overlay(img,lm)
            left_closed  = eye_closed_check(lm,LEFT_EYE)
            right_closed = eye_closed_check(lm,RIGHT_EYE)
            if left_closed or right_closed:
                if self.eye_start is None: self.eye_start=now
                if now-self.eye_start>=EYE_THRESHOLD: drowsy_eye=True
            else:
                self.eye_start=None
            left_ear  = eye_ratio(lm,LEFT_EYE,w,h)
            right_ear = eye_ratio(lm,RIGHT_EYE,w,h)
            rois = get_rppg_rois(img,lm)
            rgb_value = rppg_rgb_sample(img,rois)
            if rgb_value is not None:
                self.rppg.add_rgb_sample(rgb_value,now)
                bpm,rppg_quality,rppg_snr = self.rppg.estimate()
        model_drowsy = predict_drowsy(img)
        final_drowsy = bool(drowsy_eye or model_drowsy)
        display_bpm  = bpm if bpm is not None else self.rppg.last_hr
        hr_quality   = rppg_quality if rppg_quality is not None else self.rppg.last_quality
        signal_snr   = rppg_snr if rppg_snr else self.rppg.last_snr
        if display_bpm is None or display_bpm==0 or 60<=display_bpm<=100: hr_state="NORMAL"
        elif 50<=display_bpm<60 or 100<display_bpm<=110: hr_state="WARNING"
        else: hr_state="CRITICAL"
        if final_drowsy and hr_state=="CRITICAL": final_status="RED"
        elif final_drowsy or hr_state=="WARNING":  final_status="ORANGE"
        else:                                       final_status="GREEN"
        # emergency button
        if self.send_emergency:
            _,msg=send_sms(img); self.sms_status=msg; self.send_emergency=False
        # auto alert + SMS (exact original logic)
        if drowsy_eye:
            if self.alert_start is None:
                self.alert_start=now; self.sms_sent_for_event=False
            elapsed=now-self.alert_start
            if elapsed>=ALERT_SOUND_DELAY: self.alarm_active=True
            if elapsed>=SMS_DELAY and not self.sms_sent_for_event:
                sent,msg=send_sms(img); self.sms_status=msg
                if sent: self.sms_sent_for_event=True
        else:
            self.alarm_active=False; self.alert_start=None
            self.sms_sent_for_event=False; self.sms_status="SMS READY"
        dash = draw_dashboard(img, final_status, final_drowsy, left_ear, right_ear,
                              float(display_bpm) if display_bpm else None,
                              hr_quality, signal_snr, self.fps_smooth, self.sms_status)
        out=VideoFrame.from_ndarray(dash,format="bgr24")
        out.pts=frame.pts; out.time_base=frame.time_base
        return out

# ── BROWSER ALARM (replaces pygame) ──────────────────────────
ALARM_JS = """<script>
var _vgAlarm=null;
function vgStart(){
  if(_vgAlarm)return;
  var ctx=new(window.AudioContext||window.webkitAudioContext)();
  function beep(){
    var o=ctx.createOscillator(),g=ctx.createGain();
    o.connect(g);g.connect(ctx.destination);
    o.frequency.value=880;
    g.gain.setValueAtTime(0.7,ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+0.45);
    o.start(ctx.currentTime);o.stop(ctx.currentTime+0.45);
  }
  beep();_vgAlarm=setInterval(beep,700);
}
function vgStop(){if(_vgAlarm){clearInterval(_vgAlarm);_vgAlarm=null;}}
</script>"""

# ── UI ───────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-top:0.5rem !important;}
video{width:100% !important;}
</style>""", unsafe_allow_html=True)

st.title("🚗 VisionGuard — AI Driver Monitoring System")
st.caption("Real-time drowsiness detection + rPPG heart rate monitoring")

st.components.v1.html(ALARM_JS, height=0)

RTC = RTCConfiguration({"iceServers":[
    {"urls":["stun:stun.l.google.com:19302"]},
    {"urls":["stun:stun1.l.google.com:19302"]},
    {"urls":["turn:a.relay.metered.ca:80"],"username":"83eebabf8b1cfa513a649607","credential":"2D7JvfkOQtBdYW3R"},
    {"urls":["turn:a.relay.metered.ca:443"],"username":"83eebabf8b1cfa513a649607","credential":"2D7JvfkOQtBdYW3R"},
    {"urls":["turns:a.relay.metered.ca:443?transport=tcp"],"username":"83eebabf8b1cfa513a649607","credential":"2D7JvfkOQtBdYW3R"}
]})
ctx = webrtc_streamer(
    key="vg",
    video_processor_factory=DriverMonitor,
    rtc_configuration=RTC,
    media_stream_constraints={"video":True,"audio":False},
    async_processing=True,
)

if ctx.video_processor:
    c1,c2 = st.columns([1,1])
    with c1:
        if st.button("🚨 Emergency SMS"):
            ctx.video_processor.send_emergency = True
    with c2:
        if st.button("🔕 Stop Alarm"):
            ctx.video_processor.alarm_active = False
            st.components.v1.html("<script>vgStop();</script>", height=0)
    # alarm trigger
    if ctx.video_processor.alarm_active:
        st.components.v1.html("<script>vgStart();</script>", height=0)
    else:
        st.components.v1.html("<script>vgStop();</script>", height=0)
