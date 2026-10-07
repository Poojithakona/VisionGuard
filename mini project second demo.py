# ============================================================
# AI DRIVER MONITORING SYSTEM - FINAL VERSION
# ============================================================
# FINAL BEHAVIOUR
# 1. GREEN  = drowsiness normal AND rPPG normal
# 2. ORANGE = either drowsiness OR rPPG abnormal
# 3. RED    = both abnormal
# 4. Short blink is NOT drowsiness
# 5. Continuous eye closure for 2 seconds = drowsiness
# 6. EYES OPEN -> alarm/timer cleared immediately
# 7. SMS is sent only after sustained drowsiness
# 8. E -> manual emergency SMS
# 9. No random/fake heart-rate values
# 10. Clean MediaPipe face overlay
# 11. SMS uses OLD WORKING TWILIO LOGIC
# ============================================================

import os
import time
import math
import threading
from collections import deque

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import pygame
import requests
import mediapipe as mp

from scipy.signal import butter, filtfilt


# ============================================================
# OPTIONAL TENSORFLOW
# ============================================================

try:
    import tensorflow as tf
except Exception:
    tf = None


# ============================================================
# WINDOWS LOCATION
# ============================================================

try:
    from winrt.windows.devices.geolocation import Geolocator
    WINRT_LOCATION_AVAILABLE = True
except Exception:
    WINRT_LOCATION_AVAILABLE = False


# ============================================================
# TWILIO
# ============================================================

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_TITLE = "AI DRIVER MONITORING SYSTEM"

MODEL_PATH = os.getenv("MODEL_PATH", "drowsiness_model.h5")

SOUND_PATH = os.getenv("SOUND_PATH", "alert.wav")


# ============================================================
# DROWSINESS SETTINGS
# ============================================================

DROWSY_CLOSED_SECONDS = 2.0

EYE_OPEN_RECOVERY_SECONDS = 0.05

SMS_AFTER_DROWSY_SECONDS = 10.0

SMS_COOLDOWN_SECONDS = 60.0


# ============================================================
# EYE LANDMARKS
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

LEFT_IRIS = [
    468,
    469,
    470,
    471,
    472
]

RIGHT_IRIS = [
    473,
    474,
    475,
    476,
    477
]

EYE_CLOSED_EAR = 0.20


# ============================================================
# rPPG SETTINGS
# ============================================================

RPPG_BUFFER_SECONDS = 20.0

RPPG_MIN_SECONDS = 8.0

RPPG_MIN_SAMPLES = 180

HR_LOW = 45.0

HR_HIGH = 150.0

HR_NORMAL_LOW = 60.0

HR_NORMAL_HIGH = 100.0

BOTH_ABNORMAL_SMS_SECONDS = 3.0

MIN_SNR = 1.15

HR_UPDATE_SECONDS = 0.8

GOOD_SIGNAL_SNR = 2.0

FAIR_SIGNAL_SNR = 1.35


# ============================================================
# TWILIO - SAME OLD WORKING LOGIC
# ============================================================

if TWILIO_AVAILABLE:

    # ========================================================
    # PUT YOUR TWILIO DETAILS HERE
    # ========================================================

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")

    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    twilio_number = os.getenv("TWILIO_NUMBER")

    phone = os.getenv("ALERT_PHONE")

    # ========================================================
    # CREATE CLIENT
    # ========================================================

    try:

        client = Client(
            account_sid,
            auth_token
        )

        print("Twilio initialized successfully.")

    except Exception as e:

        print("Twilio initialization error:", e)

        client = None

else:

    client = None

    print("WARNING: Twilio package not available.")


# ============================================================
# LOCATION
# ============================================================

location_lock = threading.Lock()

location_cache = {

    "url": None,

    "status": "LOCATION STARTING",

    "latitude": None,

    "longitude": None,

    "accuracy": None,

    "updated": 0.0
}

location_stop_event = threading.Event()

LOCATION_REFRESH_SECONDS = 10.0


# ============================================================
# WINDOWS LOCATION
# ============================================================

async def _get_winrt_position():

    locator = Geolocator()

    try:

        locator.allow_fallback_to_consentless_positions()

    except Exception:

        pass

    try:

        from winrt.windows.devices.geolocation import PositionAccuracy

        locator.desired_accuracy = PositionAccuracy.HIGH

    except Exception:

        pass

    position = await locator.get_geoposition_async()

    return position


# ============================================================
# UPDATE WINDOWS LOCATION
# ============================================================

def _update_windows_location_once():

    if not WINRT_LOCATION_AVAILABLE:

        with location_lock:

            location_cache["status"] = (
                "LOCATION API UNAVAILABLE"
            )

        return False

    try:

        import asyncio

        position = asyncio.run(
            _get_winrt_position()
        )

        point = position.coordinate.point

        latitude = float(
            point.position.latitude
        )

        longitude = float(
            point.position.longitude
        )

        if not (
            math.isfinite(latitude)
            and
            math.isfinite(longitude)
        ):

            raise ValueError(
                "Invalid coordinates"
            )

        accuracy = getattr(
            position.coordinate,
            "accuracy",
            None
        )

        if accuracy is not None:

            try:

                accuracy = float(
                    accuracy
                )

            except Exception:

                accuracy = None

        maps_url = (
            "https://maps.google.com/?q="
            f"{latitude:.7f},{longitude:.7f}"
        )

        with location_lock:

            location_cache["url"] = maps_url

            location_cache["latitude"] = latitude

            location_cache["longitude"] = longitude

            location_cache["accuracy"] = accuracy

            location_cache["updated"] = time.time()

            if accuracy is not None:

                location_cache["status"] = (
                    f"DEVICE LOCATION OK "
                    f"(±{accuracy:.0f} m)"
                )

            else:

                location_cache["status"] = (
                    "DEVICE LOCATION OK"
                )

        print(
            "Windows device location:",
            f"{latitude:.7f}, {longitude:.7f}",
            f"(accuracy={accuracy})"
        )

        return True

    except Exception as exc:

        with location_lock:

            location_cache["status"] = (
                f"LOCATION ERROR: "
                f"{type(exc).__name__}"
            )

        print(
            "Windows location unavailable:",
            type(exc).__name__,
            str(exc)
        )

        return False


# ============================================================
# LOCATION WORKER
# ============================================================

def location_worker():

    while not location_stop_event.is_set():

        _update_windows_location_once()

        location_stop_event.wait(
            LOCATION_REFRESH_SECONDS
        )


# ============================================================
# START LOCATION SERVICE
# ============================================================

def start_location_service():

    if not WINRT_LOCATION_AVAILABLE:

        return None

    thread = threading.Thread(
        target=location_worker,
        daemon=True,
        name="WindowsLocationService"
    )

    thread.start()

    return thread


# ============================================================
# GET CACHED LOCATION
# ============================================================

def get_cached_windows_location():

    with location_lock:

        return (
            location_cache["url"],
            location_cache["status"],
            location_cache["accuracy"],
            location_cache["updated"]
        )


# ============================================================
# GET WINDOWS LOCATION
# ============================================================

def get_windows_location(
    timeout_seconds=6.0
):

    url, status, accuracy, updated = (
        get_cached_windows_location()
    )

    if (
        url
        and
        (time.time() - updated) <= 30.0
    ):

        return url, status

    finished = {
        "done": False
    }

    def worker():

        try:

            _update_windows_location_once()

        finally:

            finished["done"] = True

    thread = threading.Thread(
        target=worker,
        daemon=True
    )

    thread.start()

    thread.join(timeout_seconds)

    url, status, accuracy, updated = (
        get_cached_windows_location()
    )

    if url:

        return url, status

    return None, status


# ============================================================
# START LOCATION
# ============================================================

location_thread = start_location_service()


# ============================================================
# SMS - OLD WORKING LOGIC
# ============================================================
#
# IMPORTANT:
#
# This is intentionally kept like your OLD working code.
#
# It uses:
#
# requests -> ipinfo.io
#
# and then:
#
# client.messages.create()
#
# There is NO Twilio status fetch.
# There is NO queued/failed/undelivered checking.
#
# ============================================================

def get_location():

    try:

        res = requests.get(
            "https://ipinfo.io/json",
            timeout=5
        ).json()

        loc = res["loc"]

        return (
            "https://maps.google.com/?q="
            f"{loc}"
        )

    except Exception as e:

        print(
            "LOCATION ERROR:",
            e
        )

        return "Location not available"


# ============================================================
# SEND SMS
# ============================================================

def send_sms(frame):

    try:

        # Save driver image
        cv2.imwrite(
            "driver.jpg",
            frame
        )

        # Get location
        location = get_location()

        # ====================================================
        # SAME OLD WORKING TWILIO CALL
        # ====================================================

        client.messages.create(

            body=(
                "🚨 EMERGENCY!\n"
                "Driver unsafe.\n"
                "Location:\n"
                f"{location}"
            ),

            from_=twilio_number,

            to=phone
        )

        # ====================================================
        # DO NOT CHECK TWILIO STATUS
        # ====================================================

        print(
            "SMS SENT ✅"
        )

        return True, "SMS SENT"

    except Exception as e:

        print(
            "SMS ERROR:",
            e
        )

        return False, "SMS ERROR"


# ============================================================
# SOUND
# ============================================================

sound = None

try:

    pygame.mixer.init()

    if os.path.exists(
        SOUND_PATH
    ):

        sound = pygame.mixer.Sound(
            SOUND_PATH
        )

except Exception as exc:

    print(
        "Sound initialization error:",
        exc
    )

    sound = None


# ============================================================
# ALARM ON
# ============================================================

def alarm_on():

    if sound is not None:

        try:

            if not pygame.mixer.get_busy():

                sound.play(-1)

        except Exception:

            pass


# ============================================================
# ALARM OFF
# ============================================================

def alarm_off():

    try:

        pygame.mixer.stop()

    except Exception:

        pass


# ============================================================
# DROWSINESS MODEL
# ============================================================

model = None

model_available = False

model_input_size = (
    224,
    224
)

print(
    "Loading drowsiness model..."
)


# ============================================================
# LOAD MODEL
# ============================================================

if (
    tf is not None
    and
    os.path.exists(MODEL_PATH)
):

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        model_available = True

        shape = model.input_shape

        if isinstance(
            shape,
            list
        ):

            shape = shape[0]

        if len(shape) >= 3:

            ih = shape[1]

            iw = shape[2]

            if (
                isinstance(ih, int)
                and
                isinstance(iw, int)
            ):

                model_input_size = (
                    iw,
                    ih
                )

        print(
            "Drowsiness model loaded."
        )

        print(
            "Model input size:",
            model_input_size
        )

    except Exception as exc:

        fallback_error = None

        fallback_loaded = False

        try:

            from tensorflow.keras.layers import InputLayer

            original_inputlayer_init = (
                InputLayer.__init__
            )

            def compatible_inputlayer_init(
                self,
                *args,
                **kwargs
            ):

                if (
                    "batch_shape" in kwargs
                    and
                    "batch_input_shape"
                    not in kwargs
                ):

                    kwargs[
                        "batch_input_shape"
                    ] = kwargs.pop(
                        "batch_shape"
                    )

                return original_inputlayer_init(
                    self,
                    *args,
                    **kwargs
                )

            InputLayer.__init__ = (
                compatible_inputlayer_init
            )

            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

            model_available = True

            fallback_loaded = True

            print(
                "Drowsiness model loaded "
                "using H5 compatibility fallback."
            )

        except Exception as e2:

            fallback_error = e2

            model = None

            model_available = False

        if not fallback_loaded:

            print(
                "WARNING: Drowsiness model "
                "could not be loaded."
            )

            print(
                "Reason:",
                exc
            )

            print(
                "Compatibility fallback:",
                fallback_error
            )

            print(
                "Eye-based drowsiness monitoring "
                "will continue."
            )

else:

    print(
        "WARNING: Drowsiness model file not found."
    )

    print(
        "Eye-based drowsiness monitoring "
        "will continue."
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_model_drowsiness(frame):

    model_drowsy = 0

    if not model_available:

        return False, None

    try:

        img = cv2.resize(
            frame,
            (64, 64)
        ) / 255.0

        img = img.reshape(
            1,
            64,
            64,
            3
        )

        probability = float(
            model.predict(
                img,
                verbose=0
            )[0][0]
        )

        model_drowsy = (
            probability > 0.95
        )

        return (
            bool(model_drowsy),
            probability
        )

    except Exception:

        return False, None


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = (
    mp.solutions.face_mesh
)

face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.55,

    min_tracking_confidence=0.55
)


# ============================================================
# POINT XY
# ============================================================

def point_xy(
    landmark,
    w,
    h
):

    return (
        int(landmark.x * w),
        int(landmark.y * h)
    )


# ============================================================
# EYE RATIO
# ============================================================

def eye_ratio(
    landmarks,
    eye_indices,
    w,
    h
):

    pts = np.array(

        [
            point_xy(
                landmarks[i],
                w,
                h
            )

            for i in eye_indices
        ],

        dtype=np.float32
    )

    vertical_1 = np.linalg.norm(
        pts[1] - pts[5]
    )

    vertical_2 = np.linalg.norm(
        pts[2] - pts[4]
    )

    horizontal = np.linalg.norm(
        pts[0] - pts[3]
    )

    if horizontal < 1e-6:

        return 0.0

    return float(
        (
            vertical_1
            +
            vertical_2
        )
        /
        (
            2.0 * horizontal
        )
    )


# ============================================================
# FACE OVAL
# ============================================================

FACE_OVAL = [

    10, 338, 297, 332, 284, 251,

    389, 356, 454, 323, 361, 288,

    397, 365, 379, 378, 400, 377,

    152, 148, 176, 149, 150, 136,

    172, 58, 132, 93, 234, 127,

    162, 21, 54, 103, 67, 109, 10

]


# ============================================================
# CLEAN FACE OVERLAY
# ============================================================

def draw_clean_face_overlay(
    frame,
    landmarks
):

    h, w = frame.shape[:2]

    reference_indices = [

        10,
        152,
        1,
        4,
        13,
        14,
        33,
        133,
        362,
        263,
        61,
        291,
        234,
        454

    ]

    for idx in reference_indices:

        x, y = point_xy(
            landmarks[idx],
            w,
            h
        )

        cv2.circle(
            frame,
            (x, y),
            2,
            (80, 210, 255),
            -1
        )

    forehead_x, forehead_y = (
        point_xy(
            landmarks[10],
            w,
            h
        )
    )

    left_cheek_x, left_cheek_y = (
        point_xy(
            landmarks[234],
            w,
            h
        )
    )

    right_cheek_x, right_cheek_y = (
        point_xy(
            landmarks[454],
            w,
            h
        )
    )

    roi_points = [

        (
            forehead_x,
            forehead_y + 18
        ),

        (
            left_cheek_x + 18,
            left_cheek_y + 8
        ),

        (
            right_cheek_x - 18,
            right_cheek_y + 8
        )

    ]

    for rx, ry in roi_points:

        cv2.circle(
            frame,
            (
                int(rx),
                int(ry)
            ),
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            (
                int(rx),
                int(ry)
            ),
            1,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

    pts = np.array(

        [
            point_xy(
                landmarks[i],
                w,
                h
            )

            for i in FACE_OVAL
        ],

        dtype=np.int32
    )

    cv2.polylines(

        frame,

        [pts],

        isClosed=False,

        color=(75, 180, 205),

        thickness=1,

        lineType=cv2.LINE_AA
    )

    for eye in (
        LEFT_EYE,
        RIGHT_EYE
    ):

        pts_eye = np.array(

            [
                point_xy(
                    landmarks[i],
                    w,
                    h
                )

                for i in eye
            ],

            dtype=np.int32
        )

        cv2.polylines(

            frame,

            [pts_eye],

            isClosed=True,

            color=(70, 220, 150),

            thickness=2,

            lineType=cv2.LINE_AA
        )

    for iris in (
        LEFT_IRIS,
        RIGHT_IRIS
    ):

        pts_iris = np.array(

            [
                point_xy(
                    landmarks[i],
                    w,
                    h
                )

                for i in iris
            ],

            dtype=np.float32
        )

        center = (
            np.mean(
                pts_iris,
                axis=0
            )
            .astype(int)
        )

        radii = np.linalg.norm(
            pts_iris - center,
            axis=1
        )

        radius = int(
            max(
                2,
                np.mean(radii)
            )
        )

        cv2.circle(

            frame,

            tuple(center),

            radius,

            (70, 190, 255),

            1,

            cv2.LINE_AA
        )

        cv2.circle(

            frame,

            tuple(center),

            2,

            (255, 255, 255),

            -1
        )


# ============================================================
# RPPG PROCESSOR
# ============================================================

class RPPGProcessor:

    def __init__(self):

        self.samples = deque(
            maxlen=900
        )

        self.times = deque(
            maxlen=900
        )

        self.last_hr = None

        self.last_quality = "WAITING"

        self.last_snr = 0.0

        self.last_update = 0.0

        self.stable_hr = None

        self.candidate_history = deque(
            maxlen=5
        )

        self.last_candidate = None


    # ========================================================
    # ADD RGB
    # ========================================================

    def add_rgb_sample(
        self,
        rgb_value,
        timestamp
    ):

        self.samples.append(
            np.asarray(
                rgb_value,
                dtype=np.float64
            )
        )

        self.times.append(
            float(timestamp)
        )

        cutoff = (
            timestamp
            -
            RPPG_BUFFER_SECONDS
        )

        while (
            self.times
            and
            self.times[0] < cutoff
        ):

            self.times.popleft()

            self.samples.popleft()


    # ========================================================
    # ESTIMATE BPM
    # ========================================================

    def estimate(self):

        now = time.time()

        if (
            now
            -
            self.last_update
            < 1.5
        ):

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        self.last_update = now

        if len(
            self.samples
        ) < RPPG_MIN_SAMPLES:

            self.last_quality = (
                "COLLECTING"
            )

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        t = np.asarray(
            self.times,
            dtype=np.float64
        )

        rgb = np.asarray(
            self.samples,
            dtype=np.float64
        )

        duration = (
            t[-1]
            -
            t[0]
        )

        if duration < 8.0:

            self.last_quality = (
                "COLLECTING"
            )

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        dt = np.diff(t)

        dt = dt[
            np.isfinite(dt)
            &
            (dt > 0)
        ]

        if len(dt) < 20:

            self.last_quality = "WEAK"

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        fs = 1.0 / np.median(dt)

        if fs < 12:

            self.last_quality = (
                "LOW FPS"
            )

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        good = np.all(
            np.isfinite(rgb),
            axis=1
        )

        rgb = rgb[good]

        if len(rgb) < RPPG_MIN_SAMPLES:

            self.last_quality = "WEAK"

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        mean_rgb = np.mean(
            rgb,
            axis=0
        )

        mean_rgb = np.maximum(
            mean_rgb,
            1e-6
        )

        C = (
            rgb
            /
            mean_rgb
            -
            1.0
        )

        s1 = (
            C[:, 1]
            -
            C[:, 2]
        )

        s2 = (
            C[:, 1]
            +
            C[:, 2]
            -
            2.0 * C[:, 0]
        )

        std1 = np.std(s1)

        std2 = np.std(s2)

        if (
            std1 < 1e-6
            or
            std2 < 1e-6
        ):

            self.last_hr = None

            self.last_quality = (
                "NO SIGNAL"
            )

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        alpha = (
            std1
            /
            std2
        )

        pulse = (
            s1
            +
            alpha * s2
        )

        pulse -= np.mean(pulse)

        pulse_std = np.std(
            pulse
        )

        if pulse_std < 1e-6:

            self.last_hr = None

            self.last_quality = (
                "NO SIGNAL"
            )

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        pulse /= pulse_std

        low = 0.70

        high = 2.50

        nyq = fs / 2.0

        if nyq <= high:

            self.last_quality = (
                "LOW FPS"
            )

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        try:

            b, a = butter(

                3,

                [
                    low / nyq,
                    high / nyq
                ],

                btype="band"
            )

            padlen = (
                3
                *
                max(
                    len(a),
                    len(b)
                )
            )

            if len(pulse) <= (
                padlen + 5
            ):

                self.last_quality = (
                    "COLLECTING"
                )

                return (
                    self.last_hr,
                    self.last_quality,
                    self.last_snr
                )

            filtered = filtfilt(
                b,
                a,
                pulse
            )

        except Exception:

            self.last_quality = (
                "FILTER ERROR"
            )

            self.last_snr = 0.0

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        n = len(filtered)

        window = np.hanning(n)

        spectrum = np.abs(
            np.fft.rfft(
                filtered * window,
                n=n * 8
            )
        )

        freqs = np.fft.rfftfreq(
            n * 8,
            d=1.0 / fs
        )

        valid = (
            (freqs >= low)
            &
            (freqs <= high)
        )

        if not np.any(valid):

            return (
                self.last_hr,
                "NO SIGNAL",
                0.0
            )

        mag = spectrum[valid]

        f = freqs[valid]

        k = int(
            np.argmax(mag)
        )

        peak_mag = float(
            mag[k]
        )

        noise_mask = np.ones(
            len(mag),
            dtype=bool
        )

        noise_mask[
            max(0, k - 12):
            min(len(mag), k + 13)
        ] = False

        noise = float(
            np.median(
                mag[noise_mask]
            )
        ) if np.any(
            noise_mask
        ) else 1e-6

        snr = (
            peak_mag
            /
            max(
                noise,
                1e-6
            )
        )

        if snr >= GOOD_SIGNAL_SNR:

            quality = "GOOD"

        elif snr >= FAIR_SIGNAL_SNR:

            quality = "FAIR"

        else:

            quality = "WEAK"

        self.last_snr = snr

        if snr < MIN_SNR:

            self.last_quality = quality

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        peak_f = float(
            f[k]
        )

        if (
            0 < k
            <
            len(mag) - 1
        ):

            y1, y2, y3 = np.log(
                np.maximum(
                    mag[k - 1:k + 2],
                    1e-12
                )
            )

            denom = (
                y1
                -
                2 * y2
                +
                y3
            )

            if abs(denom) > 1e-12:

                delta = (
                    0.5
                    *
                    (y1 - y3)
                    /
                    denom
                )

                bin_hz = (
                    f[1]
                    -
                    f[0]
                )

                peak_f += (
                    float(delta)
                    *
                    bin_hz
                )

        candidate = (
            peak_f
            *
            60.0
        )

        if (
            not math.isfinite(
                candidate
            )
            or
            not (
                HR_LOW
                <= candidate
                <= HR_HIGH
            )
        ):

            self.last_quality = (
                "NO SIGNAL"
            )

            return (
                self.last_hr,
                self.last_quality,
                self.last_snr
            )

        self.candidate_history.append(
            candidate
        )

        consensus = float(
            np.median(
                self.candidate_history
            )
        )

        if self.stable_hr is None:

            self.stable_hr = consensus

        else:

            jump = abs(
                consensus
                -
                self.stable_hr
            )

            if jump <= 6.0:

                self.stable_hr = (
                    0.20 * consensus
                    +
                    0.80 * self.stable_hr
                )

            elif jump <= 12.0:

                self.stable_hr = (
                    0.08 * consensus
                    +
                    0.92 * self.stable_hr
                )

        self.last_hr = (
            self.stable_hr
        )

        self.last_quality = quality

        self.last_candidate = (
            candidate
        )

        return (
            self.last_hr,
            self.last_quality,
            self.last_snr
        )


# ============================================================
# RPPG OBJECT
# ============================================================

rppg_processor = RPPGProcessor()


# ============================================================
# RPPG ROI
# ============================================================

def get_rppg_rois(
    frame,
    landmarks
):

    h, w = frame.shape[:2]

    def p(idx):

        return point_xy(
            landmarks[idx],
            w,
            h
        )

    forehead = p(10)

    left_cheek = p(234)

    right_cheek = p(454)

    roi_list = []


    def add_roi(
        cx,
        cy,
        rw,
        rh
    ):

        x1 = max(
            0,
            int(cx - rw / 2)
        )

        y1 = max(
            0,
            int(cy - rh / 2)
        )

        x2 = min(
            w,
            int(cx + rw / 2)
        )

        y2 = min(
            h,
            int(cy + rh / 2)
        )

        if (
            x2 > x1
            and
            y2 > y1
        ):

            roi_list.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


    add_roi(
        forehead[0],
        forehead[1] + 18,
        55,
        28
    )

    add_roi(
        left_cheek[0] + 18,
        left_cheek[1] + 8,
        45,
        38
    )

    add_roi(
        right_cheek[0] - 18,
        right_cheek[1] + 8,
        45,
        38
    )

    return roi_list


# ============================================================
# RPPG RGB SAMPLE
# ============================================================

def rppg_rgb_sample(
    frame,
    rois
):

    values = []

    for (
        x1,
        y1,
        x2,
        y2
    ) in rois:

        roi = frame[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:

            continue

        rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        ).reshape(
            -1,
            3
        ).astype(
            np.float32
        )

        clean_channels = []

        for c in range(3):

            channel = rgb[:, c]

            lo, hi = np.percentile(
                channel,
                [10, 90]
            )

            clean = channel[
                (channel >= lo)
                &
                (channel <= hi)
            ]

            if len(clean) > 10:

                clean_channels.append(
                    np.median(clean)
                )

            else:

                clean_channels.append(
                    np.median(channel)
                )

        values.append(
            clean_channels
        )

    if not values:

        return None

    return np.mean(
        np.asarray(
            values,
            dtype=np.float64
        ),
        axis=0
    )


# ============================================================
# DASHBOARD HELPERS
# ============================================================

def fit_text(
    text,
    max_chars
):

    if len(text) <= max_chars:

        return text

    return (
        text[:max_chars - 1]
        +
        "…"
    )


# ============================================================
# DRAW PANEL
# ============================================================

def draw_panel(
    img,
    x1,
    y1,
    x2,
    y2,
    title,
    accent
):

    cv2.rectangle(

        img,

        (x1, y1),

        (x2, y2),

        (30, 26, 24),

        -1
    )

    cv2.rectangle(

        img,

        (x1, y1),

        (x2, y2),

        (75, 68, 62),

        1
    )

    cv2.rectangle(

        img,

        (x1, y1),

        (x1 + 5, y2),

        accent,

        -1
    )

    cv2.putText(

        img,

        title,

        (x1 + 20, y1 + 38),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.82,

        (240, 238, 232),

        2,

        cv2.LINE_AA
    )


# ============================================================
# SCREEN SIZE
# ============================================================

def get_screen_size():

    try:

        import ctypes

        user32 = ctypes.windll.user32

        return (
            int(
                user32.GetSystemMetrics(0)
            ),
            int(
                user32.GetSystemMetrics(1)
            )
        )

    except Exception:

        return (
            1600,
            900
        )


# ============================================================
# DASHBOARD
# ============================================================

def draw_dashboard(

    frame,

    final_status,

    drowsy,

    left_ear,

    right_ear,

    hr,

    hr_quality,

    signal_snr,

    fps_value,

    sms_status

):

    screen_w, screen_h = (
        get_screen_size()
    )

    screen_w = max(
        1100,
        screen_w
    )

    screen_h = max(
        650,
        screen_h
    )

    canvas = np.zeros(
        (
            screen_h,
            screen_w,
            3
        ),
        dtype=np.uint8
    )

    canvas[:] = (
        12,
        13,
        15
    )

    margin = 24

    header_h = 62

    cv2.putText(

        canvas,

        "AI DRIVER MONITORING SYSTEM",

        (margin, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.95,

        (242, 242, 238),

        2,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        "REAL-TIME DRIVER SAFETY",

        (
            screen_w - 285,
            38
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (150, 150, 150),

        1,

        cv2.LINE_AA
    )

    cv2.line(

        canvas,

        (margin, header_h),

        (
            screen_w - margin,
            header_h
        ),

        (55, 58, 62),

        1
    )

    gap = 18

    footer_h = 48

    content_top = (
        header_h
        +
        16
    )

    content_bottom = (
        screen_h
        -
        footer_h
    )

    left_w = int(
        screen_w * 0.66
    )

    right_w = (
        screen_w
        -
        margin * 2
        -
        left_w
        -
        gap
    )

    left_x1 = margin

    left_x2 = (
        left_x1
        +
        left_w
    )

    right_x1 = (
        left_x2
        +
        gap
    )

    right_x2 = (
        screen_w
        -
        margin
    )


    # ========================================================
    # LIVE CAMERA
    # ========================================================

    draw_panel(

        canvas,

        left_x1,

        content_top,

        left_x2,

        content_bottom,

        "LIVE CAMERA",

        (40, 155, 220)
    )

    cam_x1 = (
        left_x1 + 12
    )

    cam_y1 = (
        content_top + 52
    )

    cam_x2 = (
        left_x2 - 12
    )

    cam_y2 = (
        content_bottom - 12
    )

    available_w = (
        cam_x2
        -
        cam_x1
    )

    available_h = (
        cam_y2
        -
        cam_y1
    )

    fh, fw = frame.shape[:2]

    scale = min(
        available_w / fw,
        available_h / fh
    )

    draw_w = max(
        1,
        int(fw * scale)
    )

    draw_h = max(
        1,
        int(fh * scale)
    )

    camera = cv2.resize(

        frame,

        (
            draw_w,
            draw_h
        ),

        interpolation=cv2.INTER_LINEAR
    )

    dx = (
        cam_x1
        +
        (
            available_w
            -
            draw_w
        )
        //
        2
    )

    dy = (
        cam_y1
        +
        (
            available_h
            -
            draw_h
        )
        //
        2
    )

    canvas[
        dy:dy + draw_h,
        dx:dx + draw_w
    ] = camera

    cv2.circle(

        canvas,

        (
            cam_x1 + 20,
            cam_y1 + 20
        ),

        5,

        (60, 220, 75),

        -1
    )

    cv2.putText(

        canvas,

        "FACE TRACKING",

        (
            cam_x1 + 34,
            cam_y1 + 25
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (230, 230, 230),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # STATUS
    # ========================================================

    if final_status == "GREEN":

        status_color = (
            55,
            220,
            65
        )

        status_text = "NORMAL"

        sub = (
            "DRIVER CONDITION NORMAL"
        )

    elif final_status == "ORANGE":

        status_color = (
            0,
            165,
            255
        )

        status_text = "WARNING"

        sub = (
            "ATTENTION REQUIRED"
        )

    else:

        status_color = (
            45,
            45,
            235
        )

        status_text = "UNSAFE"

        sub = (
            "DRIVER SAFETY RISK"
        )


    # ========================================================
    # PANEL HEIGHTS
    # ========================================================

    p1_h = int(
        (
            content_bottom
            -
            content_top
            -
            gap * 2
        )
        *
        0.27
    )

    p2_h = int(
        (
            content_bottom
            -
            content_top
            -
            gap * 2
        )
        *
        0.36
    )

    p3_h = (

        content_bottom
        -
        content_top
        -
        gap * 2
        -
        p1_h
        -
        p2_h
    )

    p1_y1 = content_top

    p1_y2 = (
        p1_y1
        +
        p1_h
    )

    p2_y1 = (
        p1_y2
        +
        gap
    )

    p2_y2 = (
        p2_y1
        +
        p2_h
    )

    p3_y1 = (
        p2_y2
        +
        gap
    )

    p3_y2 = content_bottom


    # ========================================================
    # FINAL STATUS PANEL
    # ========================================================

    draw_panel(

        canvas,

        right_x1,

        p1_y1,

        right_x2,

        p1_y2,

        "FINAL STATUS",

        status_color
    )

    cv2.putText(

        canvas,

        status_text,

        (
            right_x1 + 24,
            p1_y1 + 94
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.18,

        status_color,

        3,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        sub,

        (
            right_x1 + 25,
            p1_y1 + 127
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (185, 185, 185),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # DROWSINESS PANEL
    # ========================================================

    drowsy_color = (

        (45, 45, 235)
        if drowsy
        else
        (55, 205, 70)

    )

    draw_panel(

        canvas,

        right_x1,

        p2_y1,

        right_x2,

        p2_y2,

        "DROWSINESS",

        drowsy_color
    )

    inner_x = (
        right_x1 + 24
    )

    col2_x = (
        right_x1
        +
        int(
            (
                right_x2
                -
                right_x1
            )
            *
            0.52
        )
    )

    left_state = (
        "OPEN"
        if left_ear >= EYE_CLOSED_EAR
        else
        "CLOSED"
    )

    right_state = (
        "OPEN"
        if right_ear >= EYE_CLOSED_EAR
        else
        "CLOSED"
    )

    cv2.putText(

        canvas,

        "LEFT EYE",

        (
            inner_x,
            p2_y1 + 72
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (150, 150, 150),

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        "RIGHT EYE",

        (
            col2_x,
            p2_y1 + 72
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (150, 150, 150),

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        left_state,

        (
            inner_x,
            p2_y1 + 108
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.67,

        (
            drowsy_color
            if left_state == "CLOSED"
            else
            (65, 210, 75)
        ),

        2,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        right_state,

        (
            col2_x,
            p2_y1 + 108
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.67,

        (
            drowsy_color
            if right_state == "CLOSED"
            else
            (65, 210, 75)
        ),

        2,

        cv2.LINE_AA
    )

    eye_message = (

        "DROWSINESS DETECTED"
        if drowsy
        else
        "EYES OPEN / NORMAL"

    )

    cv2.putText(

        canvas,

        eye_message,

        (
            inner_x,
            p2_y1 + 146
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        drowsy_color,

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        f"EAR   L {left_ear:.2f}    R {right_ear:.2f}",

        (
            inner_x,
            p2_y1 + 177
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (145, 145, 145),

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        (
            "CNN: ACTIVE"
            if model_available
            else
            "CNN: EYE-BASED MODE"
        ),

        (
            inner_x,
            p2_y1 + 205
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.40,

        (130, 130, 130),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # HEART RATE PANEL
    # ========================================================

    draw_panel(

        canvas,

        right_x1,

        p3_y1,

        right_x2,

        p3_y2,

        "rPPG / HEART RATE",

        (45, 190, 215)
    )

    hr_text = (

        "--"
        if hr is None
        else
        f"{int(round(hr))}"

    )

    cv2.putText(

        canvas,

        hr_text,

        (
            inner_x,
            p3_y1 + 88
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.25,

        (240, 240, 240),

        3,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        "BPM",

        (
            inner_x + 125,
            p3_y1 + 88
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (175, 175, 175),

        2,

        cv2.LINE_AA
    )

    if hr_quality == "GOOD":

        signal_color = (
            55,
            220,
            70
        )

    elif hr_quality == "FAIR":

        signal_color = (
            0,
            190,
            255
        )

    else:

        signal_color = (
            160,
            160,
            160
        )

    cv2.putText(

        canvas,

        f"SIGNAL: {hr_quality}",

        (
            inner_x,
            p3_y1 + 122
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        signal_color,

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        f"SNR {signal_snr:.2f}    FPS {fps_value:.1f}",

        (
            inner_x,
            p3_y1 + 151
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.40,

        (145, 145, 145),

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        "Facial RGB -> POS -> Bandpass -> FFT",

        (
            inner_x,
            p3_y1 + 179
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.38,

        (125, 125, 125),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # FOOTER
    # ========================================================

    footer_y = (
        screen_h - 16
    )

    cv2.putText(

        canvas,

        "Q  Quit",

        (
            24,
            footer_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        "A  Stop Alarm",

        (
            105,
            footer_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1,

        cv2.LINE_AA
    )

    cv2.putText(

        canvas,

        "E  Emergency SMS",

        (
            240,
            footer_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (220, 220, 220),

        1,

        cv2.LINE_AA
    )

    if "SENT" in sms_status:

        sms_color = (
            55,
            220,
            70
        )

    elif (
        "ERROR" in sms_status
        or
        "FAILED" in sms_status
    ):

        sms_color = (
            0,
            165,
            255
        )

    else:

        sms_color = (
            140,
            140,
            140
        )

    cv2.putText(

        canvas,

        fit_text(
            sms_status,
            30
        ),

        (
            screen_w - 290,
            footer_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.43,

        sms_color,

        1,

        cv2.LINE_AA
    )

    return canvas


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    cap = cv2.VideoCapture(
        0
    )

if not cap.isOpened():

    raise RuntimeError(
        "Camera could not be opened. "
        "Check Windows camera permission."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# CONTROL VARIABLES
# ============================================================

eye_start = None

alert_start = None

sms_sent_for_event = False

last_sms_time = 0.0

sms_status = "SMS READY"


# ============================================================
# TIMINGS
# ============================================================

EYE_THRESHOLD = 2

ALERT_SOUND_DELAY = 5

SMS_DELAY = 10


# ============================================================
# FPS
# ============================================================

last_frame_time = time.time()

fps_smooth = 0.0


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_TITLE,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(

    WINDOW_TITLE,

    cv2.WND_PROP_FULLSCREEN,

    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# START MESSAGE
# ============================================================

print()

print(
    "=============================================="
)

print(
    "       AI DRIVER MONITORING SYSTEM"
)

print(
    "=============================================="
)

print(
    "Q -> Quit"
)

print(
    "A -> Stop alarm"
)

print(
    "E -> Emergency SMS"
)

print(
    "Eye threshold: 2 seconds"
)

print(
    "Alert delay: 5 seconds"
)

print(
    "SMS delay: 10 seconds"
)

print(
    "SMS: OLD WORKING TWILIO LOGIC"
)

print(
    "Location: IPINFO LOCATION"
)

print(
    "=============================================="
)

print()


# ============================================================
# OLD EYE DETECTOR
# ============================================================

def eye_closed_old(
    lm,
    eye
):

    return (
        abs(
            lm[eye[1]].y
            -
            lm[eye[5]].y
        )
        <
        0.012
    )


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        now = time.time()


        # ====================================================
        # FPS
        # ====================================================

        dt = (
            now
            -
            last_frame_time
        )

        last_frame_time = now

        if dt > 0:

            current_fps = (
                1.0 / dt
            )

            fps_smooth = (

                current_fps
                if fps_smooth == 0
                else
                0.90 * fps_smooth
                +
                0.10 * current_fps

            )


        # ====================================================
        # FLIP CAMERA
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = face_mesh.process(
            rgb
        )


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        left_ear = 0.30

        right_ear = 0.30

        drowsy_eye = 0

        bpm = None

        rppg_quality = "WAITING"

        rppg_snr = 0.0

        face_found = False


        # ====================================================
        # FACE FOUND
        # ====================================================

        if result.multi_face_landmarks:

            face_found = True

            face_landmarks = (
                result
                .multi_face_landmarks[0]
                .landmark
            )

            h, w = frame.shape[:2]


            # =================================================
            # FACE OVERLAY
            # =================================================

            draw_clean_face_overlay(
                frame,
                face_landmarks
            )


            # =================================================
            # EYE DETECTION
            # =================================================

            left_closed = eye_closed_old(
                face_landmarks,
                LEFT_EYE
            )

            right_closed = eye_closed_old(
                face_landmarks,
                RIGHT_EYE
            )

            closed_count = (
                int(left_closed)
                +
                int(right_closed)
            )


            # =================================================
            # DROWSINESS TIMER
            # =================================================

            if closed_count >= 1:

                if eye_start is None:

                    eye_start = now

                if (
                    now
                    -
                    eye_start
                    >=
                    EYE_THRESHOLD
                ):

                    drowsy_eye = 1

            else:

                eye_start = None

                drowsy_eye = 0


            # =================================================
            # EAR
            # =================================================

            left_ear = eye_ratio(

                face_landmarks,

                LEFT_EYE,

                w,

                h
            )

            right_ear = eye_ratio(

                face_landmarks,

                RIGHT_EYE,

                w,

                h
            )


            # =================================================
            # RPPG
            # =================================================

            rois = get_rppg_rois(

                frame,

                face_landmarks
            )

            rgb_value = rppg_rgb_sample(

                frame,

                rois
            )

            if rgb_value is not None:

                rppg_processor.add_rgb_sample(

                    rgb_value,

                    now
                )

                (
                    bpm,
                    rppg_quality,
                    rppg_snr
                ) = (
                    rppg_processor.estimate()
                )

            else:

                bpm = None

                rppg_quality = (
                    "NO FACE"
                )

                rppg_snr = 0.0


        # ====================================================
        # BPM
        # ====================================================

        if bpm is None:

            display_bpm = (
                rppg_processor.last_hr
            )

            hr_quality = (
                rppg_processor.last_quality
            )

            signal_snr = (
                rppg_processor.last_snr
            )

        else:

            display_bpm = bpm

            hr_quality = (
                rppg_quality
            )

            signal_snr = (
                rppg_snr
            )


        # ====================================================
        # NO RANDOM BPM
        # ====================================================

        if display_bpm is None:

            hr_state = "NORMAL"

        elif display_bpm == 0:

            hr_state = "NORMAL"

        elif (
            60
            <=
            display_bpm
            <=
            100
        ):

            hr_state = "NORMAL"

        elif (
            50
            <=
            display_bpm
            <
            60
            or
            100
            <
            display_bpm
            <=
            110
        ):

            hr_state = "WARNING"

        else:

            hr_state = "CRITICAL"


        # ====================================================
        # MODEL
        # ====================================================

        model_drowsy = 0

        if model_available:

            model_drowsy, _ = (
                predict_model_drowsiness(
                    frame
                )
            )


        # ====================================================
        # FINAL DROWSINESS
        # ====================================================

        final_drowsy = bool(

            drowsy_eye
            or
            model_drowsy

        )


        # ====================================================
        # FINAL COLOR LOGIC
        # ====================================================

        if (
            final_drowsy
            and
            hr_state == "CRITICAL"
        ):

            state = "RED"

        elif (
            final_drowsy
            or
            hr_state == "WARNING"
        ):

            state = "ORANGE"

        else:

            state = "GREEN"


        # ====================================================
        # FINAL STATUS
        # ====================================================

        if state == "GREEN":

            final_status = "GREEN"

        elif state == "ORANGE":

            final_status = "ORANGE"

        else:

            final_status = "RED"


        # ====================================================
        # HR QUALITY
        # ====================================================

        if display_bpm is None:

            hr_quality = (
                rppg_processor.last_quality
            )

            signal_snr = (
                rppg_processor.last_snr
            )


        # ====================================================
        # ALERT + AUTOMATIC SMS
        # ====================================================
        #
        # IMPORTANT:
        #
        # Automatic SMS starts ONLY after actual
        # eye-based drowsiness.
        #
        # HR abnormality alone does NOT send SMS.
        #
        # ====================================================

        if drowsy_eye:

            if alert_start is None:

                alert_start = now

                sms_sent_for_event = False

                print(
                    "DROWSINESS DETECTED - "
                    "ALERT TIMER STARTED"
                )


            elapsed = (
                now
                -
                alert_start
            )


            # =================================================
            # ALARM AFTER 5 SECONDS
            # =================================================

            if (
                elapsed
                >=
                ALERT_SOUND_DELAY
            ):

                alarm_on()


            # =================================================
            # SMS AFTER 10 SECONDS
            # =================================================

            if (
                elapsed
                >=
                SMS_DELAY
                and
                not sms_sent_for_event
            ):

                # =============================================
                # SAME OLD SMS FUNCTION
                # =============================================

                sent, message = send_sms(
                    frame
                )

                sms_status = message

                if sent:

                    sms_sent_for_event = True

                    last_sms_time = now

                    print(
                        "AUTOMATIC SMS SENT - "
                        "THIS EVENT"
                    )


        else:

            # =================================================
            # EYES OPEN
            # =================================================

            alarm_off()

            alert_start = None

            sms_sent_for_event = False

            sms_status = "SMS READY"


        # ====================================================
        # DASHBOARD
        # ====================================================

        dashboard = draw_dashboard(

            frame,

            final_status,

            final_drowsy,

            left_ear,

            right_ear,

            (
                float(display_bpm)
                if display_bpm is not None
                else
                None
            ),

            hr_quality,

            signal_snr,

            fps_smooth,

            sms_status
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(

            WINDOW_TITLE,

            dashboard
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # ====================================================
        # Q = QUIT
        # ====================================================

        if key in (
            ord("q"),
            ord("Q")
        ):

            break


        # ====================================================
        # A = STOP ALARM
        # ====================================================

        if key in (
            ord("a"),
            ord("A")
        ):

            alarm_off()

            alert_start = None

            sms_sent_for_event = False

            sms_status = (
                "SMS READY"
            )


        # ====================================================
        # E = MANUAL EMERGENCY SMS
        # ====================================================

        if key in (
            ord("e"),
            ord("E")
        ):

            print()

            print(
                "MANUAL EMERGENCY SMS"
            )

            # ================================================
            # SAME OLD SMS LOGIC
            # ================================================

            sent, message = send_sms(
                frame
            )

            sms_status = message

            if sent:

                last_sms_time = now

                print(
                    "MANUAL SMS SENT ✅"
                )


# ============================================================
# CLEANUP
# ============================================================

finally:

    alarm_off()

    try:

        location_stop_event.set()

    except Exception:

        pass

    try:

        cap.release()

    except Exception:

        pass

    try:

        face_mesh.close()

    except Exception:

        pass

    cv2.destroyAllWindows()

    try:

        pygame.mixer.quit()

    except Exception:

        pass


# ============================================================
# STOP MESSAGE
# ============================================================

print()

print(
    "=============================================="
)

print(
    "       DRIVER MONITORING STOPPED"
)

print(
    "=============================================="
)