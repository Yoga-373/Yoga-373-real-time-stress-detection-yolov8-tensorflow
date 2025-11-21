Real-time Stress Detection System

A vision-based, non-invasive stress monitoring framework that processes live webcam feeds, screen captures, or uploaded images to detect facial stress cues in real time. Built with YOLOv8 for face detection and MobileNetV2 for stress classification, the system logs stress events, issues email alerts, and generates HTML reports with daily/hourly visualizations.

Features:
- Live Detection via webcam or desktop screen capture (≥ 15 FPS)
- Static Image Analysis through upload interface
- Face & Stress Classification using YOLOv8 + MobileNetV2
- Temporal Tracking of individual subjects with smooth bounding-box annotations
- Alerting: snapshot capture + structured JSON log + asynchronous email
- Reporting: auto-generated HTML report with charts and example images
- Modular, Cross-Platform: runs on Windows, macOS, or Linux (Python 3.8+)

Repository Structure:
detect_web.py           # Main real-time detection application
detect_screen.py        # Screen-capture detection variant
train.py                # Model training script (FER2013 → mobilenetv2_stress.h5)
report_web.py           # Report generation module (PNG/HTML output)
imageupload.py          # Gradio-based static image predictor
email_notification.py   # SMTP email helper (alerts)
app.py                  # Flask front-end for controlling detection & reports
requirements.txt        # Python dependencies
templates/              # HTML templates for Flask UI
logs/                   # Rotating logs & stress_events.json

Prerequisites:
- Python 3.8 or higher
- Git (to clone repository)
- A working SMTP account (Gmail or other) for email alerts

Install dependencies:
pip install -r requirements.txt

Configuration:
1. Environment Variables
   - EMAIL_PASSWORD – password or app-specific token for your sender email
2. config.json (example)
{
  "YOLO_MODEL_PATH": "/path/to/yolov8n.pt",
  "STRESS_MODEL_PATH": "/path/to/mobilenetv2_stress.h5",
  "CAMERA_ID": 0,
  "STRESS_THRESHOLD_PERCENT": 60,
  "STRESS_DURATION_THRESHOLD": 2,
  "COOLDOWN_PERIOD": 10,
  "MAX_ABSENCE_TIME": 5,
  "SMTP_SERVER": "smtp.gmail.com",
  "SMTP_PORT": 587,
  "SENDER_EMAIL": "you@example.com",
  "RECEIVER_EMAIL": "alert_recipient@example.com"
}

Usage:
1. Training the Stress Classifier
   python train.py

2. Real-Time Webcam Detection
   python detect_web.py --config config.json

3. Screen-Capture Detection
   python detect_screen.py --config config.json

4. Static Image Prediction (Gradio)
   python imageupload.py

5. Flask Web App
   python app.py

6. Report Generation
   python report_web.py --log logs/stress_events.json --images captured_stressed/ --output reports/

Logs & Outputs:
- logs/
  - stress_detection.log (INFO-level rotating file)
  - errors.log (ERROR-level daily file)
  - stress_events.json (structured events)
- captured_stressed/ – timestamped snapshots of stressed faces
- reports/ – generated HTML and PNG files


