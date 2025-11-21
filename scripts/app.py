# scripts/app.py
import os
import sys
import subprocess
import logging
import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory
)
from imageupload import predict_stress
from report_web import StressReportGenerator

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ─── Flask Setup ─────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")

# ─── Subprocess Handles & Paths ───────────────────────────────────────────────
webcam_proc = None
screen_proc = None
SCRIPT_DIR = os.path.dirname(__file__)

# ─── Launch / Stop Webcam Window ──────────────────────────────────────────────
@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_proc
    if webcam_proc and webcam_proc.poll() is None:
        return jsonify(status="already running")

    script = os.path.join(SCRIPT_DIR, "detect_web.py")
    plugin_path = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

    # Single bash command: export plugin path, run script, then wait for ENTER
    cmd = (
        f"export QT_QPA_PLATFORM_PLUGIN_PATH={plugin_path};"
        f" {sys.executable} {script};"
        " echo; echo 'Press ENTER to close…'; read"
    )

    webcam_proc = subprocess.Popen(
        ["gnome-terminal", "--", "bash", "-c", cmd],
        cwd=SCRIPT_DIR,
        env=os.environ.copy()
    )
    logger.info(f"Started detect_web.py (PID {webcam_proc.pid})")
    return jsonify(status="started")

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_proc
    if webcam_proc and webcam_proc.poll() is None:
        webcam_proc.terminate()
        webcam_proc.wait()
        logger.info("Stopped detect_web.py")
        return jsonify(status="stopped")
    return jsonify(status="not running")

# ─── Launch / Stop Screen Window ──────────────────────────────────────────────
@app.route('/start_screen', methods=['POST'])
def start_screen():
    global screen_proc
    if screen_proc and screen_proc.poll() is None:
        return jsonify(status="already running")

    script = os.path.join(SCRIPT_DIR, "detect_screen.py")
    plugin_path = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
    cmd = (
        f"export QT_QPA_PLATFORM_PLUGIN_PATH={plugin_path};"
        f" {sys.executable} {script};"
        " echo; echo 'Press ENTER to close…'; read"
    )

    screen_proc = subprocess.Popen(
        ["gnome-terminal", "--", "bash", "-c", cmd],
        cwd=SCRIPT_DIR,
        env=os.environ.copy()
    )
    logger.info(f"Started detect_screen.py (PID {screen_proc.pid})")
    return jsonify(status="started")

@app.route('/stop_screen', methods=['POST'])
def stop_screen():
    global screen_proc
    if screen_proc and screen_proc.poll() is None:
        screen_proc.terminate()
        screen_proc.wait()
        logger.info("Stopped detect_screen.py")
        return jsonify(status="stopped")
    return jsonify(status="not running")

# ─── Image Upload Endpoint ────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    try:
        data = file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = predict_stress(img_rgb)
        return jsonify(result=result)
    except Exception as e:
        return jsonify(error=str(e)), 500

# ─── Report Generation Endpoint ───────────────────────────────────────────────
@app.route('/api/report', methods=['GET'])
def api_report():
    rg = StressReportGenerator(
        log_file="/home/yoga/STRESS_DETECTION/STRESSED/logs/stress_events.json",
        image_dir="/home/yoga/STRESS_DETECTION/STRESSED/captured_stressed/",
        output_dir="/home/yoga/STRESS_DETECTION/STRESSED/reports/"
    )
    try:
        path = rg.generate_full_report()
        if not path:
            return jsonify(error="Report generation failed"), 500
        fname = os.path.basename(path)
        return jsonify(report_url=f"/reports/{fname}")
    except Exception as e:
        return jsonify(error=str(e)), 500

# ─── Serve Generated Reports ──────────────────────────────────────────────────
@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(
        "/home/yoga/STRESS_DETECTION/STRESSED/reports",
        filename
    )


# ─── Frontend ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ─── Run Flask ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
