# VisionGuard - AI Driver Monitoring System

Real-time driver drowsiness and heart rate monitoring system using computer vision.

## Features
- Drowsiness detection via Eye Aspect Ratio (EAR) + CNN model
- Remote Photoplethysmography (rPPG) for heart rate monitoring
- Automatic SMS alert via Twilio on sustained drowsiness
- GPS location included in alert SMS
- Color-coded status: GREEN / ORANGE / RED

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
Copy `.env.example` to `.env` and fill in your details:
```bash
cp .env.example .env
```

Edit `.env`:
```
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_NUMBER=your_twilio_number
ALERT_PHONE=your_phone_number
MODEL_PATH=path/to/drowsiness_model.h5
SOUND_PATH=path/to/alert.wav
```

### 3. Add model and sound files
Place your `drowsiness_model.h5` and `alert.wav` files in the project folder and update paths in `.env`.

### 4. Run
```bash
python "mini project second demo.py"
```

## Controls
| Key | Action |
|-----|--------|
| Q | Quit |
| A | Stop alarm |
| E | Send emergency SMS manually |

## Requirements
- Windows OS (uses Windows Camera API)
- Webcam
- Python 3.8+
