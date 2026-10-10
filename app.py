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
except Exception:
    _model = None

# ── CONSTANTS ────────────────────────────────────────────────
LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,
              397,365,379,378,400,377,152,148,176,149,150,136,
              172,58,132,93,234,127,162,21,54,103,67,109,10]
REFERENCE_INDICES = [10,152,1,4,13,14,33,133,362,263,61,291,234,454]

EYE_CLOSED_EAR   = 0.20
EYE_THRESHOLD    = 2
ALERT_SOUND_DELAY = 5
SMS_DELAY        = 10
HR_LOW, HR_HIGH  = 45.0, 150.0
RPPG_MIN_SAMPLES = 180
MIN_SNR          = 1.15
GOOD_SNR         = 2.0
FAIR_SNR         = 1.35

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
        return True, "SMS SENT ✅"
    except Exception:
        return False, "SMS ERROR"

def point_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def eye_ratio(landmarks, idx, w, h):
    pts = np.array([point_xy(landmarks[i], w, h) for i in idx], dtype=np.float32)
    v1 = np.linalg.norm(pts[1]-pts[5]); v2 = np.linalg.norm(pts[2]-pts[4])
    hz = np.linalg.norm(pts[0]-pts[3])
    return float((v1+v2)/(2.0*hz)) if hz > 1e-6 else 0.0

def eye_closed(lm, eye):
    return abs(lm[eye[1]].y - lm[eye[5]].y) < 0.012

def predict_drowsy(frame):
    if _model is None: return False
    try:
        img = cv2.resize(frame,(64,64))/255.0
        return float(_model.predict(img.reshape(1,64,64,3), verbose=0)[0][0]) > 0.95
    except Exception: return False

def draw_overlay(frame, lm):
    h, w = frame.shape[:2]
    # reference dots
    for idx in REFERENCE_INDICES:
        x, y = point_xy(lm[idx], w, h)
        cv2.circle(frame, (x,y), 2, (80,210,255), -1)
    # rPPG ROI dots
    for cx, cy in [
        (point_xy(lm[10],w,h)[0],  point_xy(lm[10],w,h)[1]+18),
        (point_xy(lm[234],w,h)[0]+18, point_xy(lm[234],w,h)[1]+8),
        (point_xy(lm[454],w,h)[0]-18, point_xy(lm[454],w,h)[1]+8),
    ]:
        cv2.circle(frame, (int(cx),int(cy)), 4, (0,0,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx),int(cy)), 1, (255,255,255), -1, cv2.LINE_AA)
    # face oval
    pts = np.array([point_xy(lm[i],w,h) for i in FACE_OVAL], dtype=np.int32)
    cv2.polylines(frame,[pts],False,(75,180,205),1,cv2.LINE_AA)
    # eyes
    for eye in (LEFT_EYE, RIGHT_EYE):
        ep = np.array([point_xy(lm[i],w,h) for i in eye], dtype=np.int32)
        cv2.polylines(frame,[ep],True,(70,220,150),2,cv2.LINE_AA)
    # iris
    for iris in (LEFT_IRIS, RIGHT_IRIS):
        ip = np.array([point_xy(lm[i],w,h) for i in iris], dtype=np.float32)
        c  = np.mean(ip,axis=0).astype(int)
        r  = int(max(2, np.mean(np.linalg.norm(ip-c,axis=1))))
        cv2.circle(frame,tuple(c),r,(70,190,255),1,cv2.LINE_AA)
        cv2.circle(frame,tuple(c),2,(255,255,255),-1)

# ── rPPG ─────────────────────────────────────────────────────
class RPPGProcessor:
    def __init__(self):
        self.samples = deque(maxlen=900); self.times = deque(maxlen=900)
        self.last_hr = None; self.last_quality = "WAITING"; self.last_snr = 0.0
        self.last_update = 0.0; self.stable_hr = None; self.hist = deque(maxlen=5)

    def add(self, rgb, ts):
        self.samples.append(np.asarray(rgb,dtype=np.float64)); self.times.append(float(ts))
        cutoff = ts - 20.0
        while self.times and self.times[0] < cutoff: self.times.popleft(); self.samples.popleft()

    def estimate(self):
        now = time.time()
        if now - self.last_update < 1.5: return self.last_hr, self.last_quality, self.last_snr
        self.last_update = now
        if len(self.samples) < RPPG_MIN_SAMPLES: self.last_quality="COLLECTING"; return self.last_hr,self.last_quality,self.last_snr
        t = np.asarray(self.times,dtype=np.float64); rgb = np.asarray(self.samples,dtype=np.float64)
        if t[-1]-t[0] < 8.0: self.last_quality="COLLECTING"; return self.last_hr,self.last_quality,self.last_snr
        dt = np.diff(t); dt = dt[np.isfinite(dt)&(dt>0)]
        if len(dt)<20: self.last_quality="WEAK"; return self.last_hr,self.last_quality,self.last_snr
        fs = 1.0/np.median(dt)
        if fs<12: self.last_quality="LOW FPS"; return self.last_hr,self.last_quality,self.last_snr
        rgb = rgb[np.all(np.isfinite(rgb),axis=1)]
        if len(rgb)<RPPG_MIN_SAMPLES: self.last_quality="WEAK"; return self.last_hr,self.last_quality,self.last_snr
        mean_rgb = np.maximum(np.mean(rgb,axis=0),1e-6); C = rgb/mean_rgb-1.0
        s1=C[:,1]-C[:,2]; s2=C[:,1]+C[:,2]-2.0*C[:,0]
        std1,std2 = np.std(s1),np.std(s2)
        if std1<1e-6 or std2<1e-6: self.last_hr=None;self.last_quality="NO SIGNAL";self.last_snr=0.0; return self.last_hr,self.last_quality,self.last_snr
        pulse = s1+(std1/std2)*s2; pulse -= np.mean(pulse); ps=np.std(pulse)
        if ps<1e-6: self.last_hr=None;self.last_quality="NO SIGNAL";self.last_snr=0.0; return self.last_hr,self.last_quality,self.last_snr
        pulse/=ps; low,high,nyq=0.70,2.50,fs/2.0
        if nyq<=high: self.last_quality="LOW FPS"; return self.last_hr,self.last_quality,self.last_snr
        try:
            b,a=butter(3,[low/nyq,high/nyq],btype="band"); padlen=3*max(len(a),len(b))
            if len(pulse)<=padlen+5: self.last_quality="COLLECTING"; return self.last_hr,self.last_quality,self.last_snr
            filtered=filtfilt(b,a,pulse)
        except Exception: self.last_quality="FILTER ERROR"; return self.last_hr,self.last_quality,self.last_snr
        n=len(filtered); spectrum=np.abs(np.fft.rfft(filtered*np.hanning(n),n=n*8)); freqs=np.fft.rfftfreq(n*8,d=1.0/fs)
        valid=(freqs>=low)&(freqs<=high)
        if not np.any(valid): return self.last_hr,"NO SIGNAL",0.0
        mag,f=spectrum[valid],freqs[valid]; k=int(np.argmax(mag)); peak_mag=float(mag[k])
        nm=np.ones(len(mag),dtype=bool); nm[max(0,k-12):min(len(mag),k+13)]=False
        noise=float(np.median(mag[nm])) if np.any(nm) else 1e-6; snr=peak_mag/max(noise,1e-6)
        quality="GOOD" if snr>=GOOD_SNR else ("FAIR" if snr>=FAIR_SNR else "WEAK"); self.last_snr=snr
        if snr<MIN_SNR: self.last_quality=quality; return self.last_hr,self.last_quality,self.last_snr
        peak_f=float(f[k])
        if 0<k<len(mag)-1:
            y1,y2,y3=np.log(np.maximum(mag[k-1:k+2],1e-12)); denom=y1-2*y2+y3
            if abs(denom)>1e-12: peak_f+=float(0.5*(y1-y3)/denom)*(f[1]-f[0])
        candidate=peak_f*60.0
        if not math.isfinite(candidate) or not (HR_LOW<=candidate<=HR_HIGH): self.last_quality="NO SIGNAL"; return self.last_hr,self.last_quality,self.last_snr
        self.hist.append(candidate); consensus=float(np.median(self.hist))
        if self.stable_hr is None: self.stable_hr=consensus
        else:
            jump=abs(consensus-self.stable_hr)
            if jump<=6.0: self.stable_hr=0.20*consensus+0.80*self.stable_hr
            elif jump<=12.0: self.stable_hr=0.08*consensus+0.92*self.stable_hr
        self.last_hr=self.stable_hr; self.last_quality=quality
        return self.last_hr,self.last_quality,self.last_snr

def get_rppg_rois(frame, lm):
    h,w=frame.shape[:2]
    def p(i): return point_xy(lm[i],w,h)
    rois=[]
    def add(cx,cy,rw,rh):
        x1,y1=max(0,int(cx-rw/2)),max(0,int(cy-rh/2)); x2,y2=min(w,int(cx+rw/2)),min(h,int(cy+rh/2))
        if x2>x1 and y2>y1: rois.append((x1,y1,x2,y2))
    fx,fy=p(10); lx,ly=p(234); rx,ry=p(454)
    add(fx,fy+18,55,28); add(lx+18,ly+8,45,38); add(rx-18,ry+8,45,38)
    return rois

def rppg_sample(frame, rois):
    vals=[]
    for x1,y1,x2,y2 in rois:
        roi=frame[y1:y2,x1:x2]
        if roi.size==0: continue
        rgb=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)
        ch=[]
        for c in range(3):
            col=rgb[:,c]; lo,hi=np.percentile(col,[10,90]); clean=col[(col>=lo)&(col<=hi)]
            ch.append(np.median(clean) if len(clean)>10 else np.median(col))
        vals.append(ch)
    return np.mean(np.asarray(vals,dtype=np.float64),axis=0) if vals else None

# ── DASHBOARD ────────────────────────────────────────────────
def draw_panel(img,x1,y1,x2,y2,title,accent):
    cv2.rectangle(img,(x1,y1),(x2,y2),(30,26,24),-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),(75,68,62),1)
    cv2.rectangle(img,(x1,y1),(x1+5,y2),accent,-1)
    cv2.putText(img,title,(x1+20,y1+38),cv2.FONT_HERSHEY_SIMPLEX,0.82,(240,238,232),2,cv2.LINE_AA)

def draw_dashboard(frame,status,drowsy,le,re,hr,hrq,snr,fps,sms,model_ok):
    W,H=1100,650
    canvas=np.full((H,W,3),(12,13,15),dtype=np.uint8)
    mg,hh,gap,fh=24,62,18,48
    cv2.putText(canvas,"AI DRIVER MONITORING SYSTEM",(mg,40),cv2.FONT_HERSHEY_SIMPLEX,0.95,(242,242,238),2,cv2.LINE_AA)
    cv2.putText(canvas,"REAL-TIME DRIVER SAFETY",(W-285,38),cv2.FONT_HERSHEY_SIMPLEX,0.55,(150,150,150),1,cv2.LINE_AA)
    cv2.line(canvas,(mg,hh),(W-mg,hh),(55,58,62),1)
    ct,cb=hh+16,H-fh; lw=int(W*0.66); lx1,lx2=mg,mg+lw; rx1,rx2=lx2+gap,W-mg
    # camera panel
    draw_panel(canvas,lx1,ct,lx2,cb,"LIVE CAMERA",(40,155,220))
    cx1,cy1,cx2,cy2=lx1+12,ct+52,lx2-12,cb-12; aw,ah=cx2-cx1,cy2-cy1
    fh2,fw=frame.shape[:2]; sc=min(aw/fw,ah/fh2); dw,dh=max(1,int(fw*sc)),max(1,int(fh2*sc))
    cam=cv2.resize(frame,(dw,dh),interpolation=cv2.INTER_LINEAR)
    dx,dy=cx1+(aw-dw)//2,cy1+(ah-dh)//2
    canvas[dy:dy+dh,dx:dx+dw]=cam
    cv2.circle(canvas,(cx1+20,cy1+20),5,(60,220,75),-1)
    cv2.putText(canvas,"FACE TRACKING",(cx1+34,cy1+25),cv2.FONT_HERSHEY_SIMPLEX,0.48,(230,230,230),1,cv2.LINE_AA)
    # status colors
    sc2=(55,220,65) if status=="GREEN" else ((0,165,255) if status=="ORANGE" else (45,45,235))
    st2="NORMAL" if status=="GREEN" else ("WARNING" if status=="ORANGE" else "UNSAFE")
    sub="DRIVER CONDITION NORMAL" if status=="GREEN" else ("ATTENTION REQUIRED" if status=="ORANGE" else "DRIVER SAFETY RISK")
    tot=cb-ct-gap*2; p1h=int(tot*0.27); p2h=int(tot*0.36)
    p1y1,p1y2=ct,ct+p1h; p2y1,p2y2=p1y2+gap,p1y2+gap+p2h; p3y1,p3y2=p2y2+gap,cb
    # status panel
    draw_panel(canvas,rx1,p1y1,rx2,p1y2,"FINAL STATUS",sc2)
    cv2.putText(canvas,st2,(rx1+24,p1y1+94),cv2.FONT_HERSHEY_SIMPLEX,1.18,sc2,3,cv2.LINE_AA)
    cv2.putText(canvas,sub,(rx1+25,p1y1+127),cv2.FONT_HERSHEY_SIMPLEX,0.48,(185,185,185),1,cv2.LINE_AA)
    # drowsiness panel
    dc=(45,45,235) if drowsy else (55,205,70)
    draw_panel(canvas,rx1,p2y1,rx2,p2y2,"DROWSINESS",dc)
    ix=rx1+24; c2x=rx1+int((rx2-rx1)*0.52)
    ls="OPEN" if le>=EYE_CLOSED_EAR else "CLOSED"; rs="OPEN" if re>=EYE_CLOSED_EAR else "CLOSED"
    cv2.putText(canvas,"LEFT EYE",(ix,p2y1+72),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1,cv2.LINE_AA)
    cv2.putText(canvas,"RIGHT EYE",(c2x,p2y1+72),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1,cv2.LINE_AA)
    cv2.putText(canvas,ls,(ix,p2y1+108),cv2.FONT_HERSHEY_SIMPLEX,0.67,dc if ls=="CLOSED" else (65,210,75),2,cv2.LINE_AA)
    cv2.putText(canvas,rs,(c2x,p2y1+108),cv2.FONT_HERSHEY_SIMPLEX,0.67,dc if rs=="CLOSED" else (65,210,75),2,cv2.LINE_AA)
    cv2.putText(canvas,"DROWSINESS DETECTED" if drowsy else "EYES OPEN / NORMAL",(ix,p2y1+146),cv2.FONT_HERSHEY_SIMPLEX,0.48,dc,1,cv2.LINE_AA)
    cv2.putText(canvas,f"EAR   L {le:.2f}    R {re:.2f}",(ix,p2y1+177),cv2.FONT_HERSHEY_SIMPLEX,0.42,(145,145,145),1,cv2.LINE_AA)
    cv2.putText(canvas,"CNN: ACTIVE" if model_ok else "CNN: EYE-BASED MODE",(ix,p2y1+205),cv2.FONT_HERSHEY_SIMPLEX,0.40,(130,130,130),1,cv2.LINE_AA)
    # heart rate panel
    draw_panel(canvas,rx1,p3y1,rx2,p3y2,"rPPG / HEART RATE",(45,190,215))
    cv2.putText(canvas,"--" if hr is None else f"{int(round(hr))}",(ix,p3y1+88),cv2.FONT_HERSHEY_SIMPLEX,1.25,(240,240,240),3,cv2.LINE_AA)
    cv2.putText(canvas,"BPM",(ix+125,p3y1+88),cv2.FONT_HERSHEY_SIMPLEX,0.55,(175,175,175),2,cv2.LINE_AA)
    sigc=(55,220,70) if hrq=="GOOD" else ((0,190,255) if hrq=="FAIR" else (160,160,160))
    cv2.putText(canvas,f"SIGNAL: {hrq}",(ix,p3y1+122),cv2.FONT_HERSHEY_SIMPLEX,0.45,sigc,1,cv2.LINE_AA)
    cv2.putText(canvas,f"SNR {snr:.2f}    FPS {fps:.1f}",(ix,p3y1+151),cv2.FONT_HERSHEY_SIMPLEX,0.40,(145,145,145),1,cv2.LINE_AA)
    cv2.putText(canvas,"Facial RGB -> POS -> Bandpass -> FFT",(ix,p3y1+179),cv2.FONT_HERSHEY_SIMPLEX,0.38,(125,125,125),1,cv2.LINE_AA)
    # footer
    fy=H-16
    cv2.putText(canvas,"🚨 Emergency SMS button below",(24,fy),cv2.FONT_HERSHEY_SIMPLEX,0.48,(220,220,220),1,cv2.LINE_AA)
    smsc=(55,220,70) if "SENT" in sms else ((0,165,255) if "ERROR" in sms else (140,140,140))
    cv2.putText(canvas,sms[:30],(W-290,fy),cv2.FONT_HERSHEY_SIMPLEX,0.43,smsc,1,cv2.LINE_AA)
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
        self.sms_sent    = False
        self.sms_status  = "SMS READY"
        self.fps_smooth  = 0.0
        self.last_t      = time.time()
        self.result      = {}
        self.send_emergency = False
        self.alarm_active   = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        dt  = now - self.last_t; self.last_t = now
        if dt > 0:
            self.fps_smooth = (1.0/dt) if self.fps_smooth==0 else 0.9*self.fps_smooth+0.1*(1.0/dt)
        img = cv2.flip(img,1)
        res = self.face_mesh.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        le=re=0.30; drowsy_eye=False; bpm=rq=None; rsnr=0.0
        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark; h,w=img.shape[:2]
            draw_overlay(img,lm)
            if eye_closed(lm,LEFT_EYE) or eye_closed(lm,RIGHT_EYE):
                if self.eye_start is None: self.eye_start=now
                if now-self.eye_start>=EYE_THRESHOLD: drowsy_eye=True
            else: self.eye_start=None
            le=eye_ratio(lm,LEFT_EYE,w,h); re=eye_ratio(lm,RIGHT_EYE,w,h)
            rv=rppg_sample(img,get_rppg_rois(img,lm))
            if rv is not None:
                self.rppg.add(rv,now); bpm,rq,rsnr=self.rppg.estimate()
        final_drowsy = drowsy_eye or predict_drowsy(img)
        bpm_d = bpm if bpm is not None else self.rppg.last_hr
        hrq   = rq  if rq  is not None else self.rppg.last_quality
        snr   = rsnr if rsnr else self.rppg.last_snr
        hr_state = "NORMAL" if bpm_d is None or bpm_d==0 or 60<=bpm_d<=100 else ("WARNING" if (50<=bpm_d<60 or 100<bpm_d<=110) else "CRITICAL")
        status = "RED" if (final_drowsy and hr_state=="CRITICAL") else ("ORANGE" if (final_drowsy or hr_state=="WARNING") else "GREEN")
        # emergency button
        if self.send_emergency:
            _,msg=send_sms(img); self.sms_status=msg; self.send_emergency=False
        # auto alert + SMS
        if drowsy_eye:
            if self.alert_start is None: self.alert_start=now; self.sms_sent=False
            elapsed=now-self.alert_start
            if elapsed>=ALERT_SOUND_DELAY: self.alarm_active=True
            if elapsed>=SMS_DELAY and not self.sms_sent:
                sent,msg=send_sms(img); self.sms_status=msg
                if sent: self.sms_sent=True
        else:
            self.alarm_active=False; self.alert_start=None; self.sms_sent=False; self.sms_status="SMS READY"
        self.result={"status":status,"drowsy":final_drowsy,"le":le,"re":re,
                     "bpm":bpm_d,"hrq":hrq,"snr":snr,"fps":self.fps_smooth,
                     "sms":self.sms_status,"alarm":self.alarm_active}
        dash=draw_dashboard(img,status,final_drowsy,le,re,
                            float(bpm_d) if bpm_d else None,
                            hrq,snr,self.fps_smooth,self.sms_status,_model is not None)
        out=VideoFrame.from_ndarray(dash,format="bgr24"); out.pts=frame.pts; out.time_base=frame.time_base
        return out

# ── ALERT SOUND (browser-side) ───────────────────────────────
ALARM_JS = """
<script>
var _vg_alarm = null;
function vgStartAlarm() {
    if (_vg_alarm) return;
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    function beep() {
        var o = ctx.createOscillator();
        var g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.frequency.value = 880;
        g.gain.setValueAtTime(0.6, ctx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime+0.4);
        o.start(ctx.currentTime); o.stop(ctx.currentTime+0.4);
    }
    beep();
    _vg_alarm = setInterval(beep, 600);
}
function vgStopAlarm() {
    if (_vg_alarm) { clearInterval(_vg_alarm); _vg_alarm = null; }
}
</script>
"""

# ── UI ───────────────────────────────────────────────────────
st.title("🚗 VisionGuard — AI Driver Monitoring System")
st.caption("Real-time drowsiness detection + rPPG heart rate monitoring")

st.components.v1.html(ALARM_JS, height=0)

RTC = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
ctx = webrtc_streamer(
    key="vg",
    video_processor_factory=DriverMonitor,
    rtc_configuration=RTC,
    media_stream_constraints={"video":True,"audio":False},
    async_processing=True,
)

if ctx.video_processor:
    col_btn1, col_btn2, _ = st.columns([1,1,4])
    with col_btn1:
        if st.button("🚨 Emergency SMS"):
            ctx.video_processor.send_emergency = True
    with col_btn2:
        if st.button("🔕 Stop Alarm"):
            ctx.video_processor.alarm_active = False
            st.components.v1.html("<script>vgStopAlarm();</script>", height=0)

    st.markdown("---")
    r = ctx.video_processor.result
    if r:
        s = r.get("status","GREEN")
        e = {"GREEN":"🟢","ORANGE":"🟠","RED":"🔴"}.get(s,"🟢")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Status",    f"{e} {s}")
        c2.metric("Heart Rate",f"{int(round(r['bpm'])) if r['bpm'] else '--'} BPM")
        c3.metric("Signal",    r.get("hrq","--"))
        c4.metric("SMS",       r.get("sms","--"))
        # trigger browser alarm
        if r.get("alarm"):
            st.components.v1.html("<script>vgStartAlarm();</script>", height=0)
        else:
            st.components.v1.html("<script>vgStopAlarm();</script>", height=0)
