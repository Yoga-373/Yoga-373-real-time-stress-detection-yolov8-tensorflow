import os

# point Qt at the real xcb plugin, not “offscreen”
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
os.environ["QT_QPA_PLATFORM"] = "xcb"


import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import mss
import time
import logging
from email_notification import send_email  
import threading

def send_email_async(subject, body, attachment_path=None):
    thread = threading.Thread(target=send_email, args=(subject, body, attachment_path))
    thread.start()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("screen_stress_detection.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load models
yolo_model = YOLO('/home/yoga/STRESS_DETECTION/STRESSED/models/yolov8n.pt')
stress_model = tf.keras.models.load_model('/home/yoga/STRESS_DETECTION/STRESSED/models/mobilenetv2_stress.h5')

# Configs
STRESS_THRESHOLD = 10    # Continuous stress duration to trigger email (in seconds)
COOLDOWN_PERIOD = 100    # Cooldown before sending another email (in seconds)

stress_tracker = {}  # Tracks when the last email was sent
stress_duration_tracker = {}  # Tracks how long a person remains stressed
total_emails_sent = 0  

# Draw vertical stress bar
def draw_vertical_stress_bar(face_crop, stress_level, bar_width=10):
    h, w, _ = face_crop.shape
    bar_height = h
    filled_height = int((stress_level / 100) * bar_height)

    red = int(255 * (stress_level / 100))
    green = int(255 * (1 - (stress_level / 100)))
    fill_color = (0, green, red)

    bar_x = w - bar_width - 5  
    cv2.rectangle(face_crop, (bar_x, 0), (bar_x + bar_width, bar_height), (255, 255, 255), 2)  
    cv2.rectangle(face_crop, (bar_x, bar_height - filled_height), (bar_x + bar_width, bar_height), fill_color, -1)

    return face_crop

# Determine stress label
def get_stress_label(stress_level):
    if stress_level <= 50:
        return "Relaxed", (0, 255, 0)
    elif stress_level <= 70:
        return "Mildly Stressed", (0, 255, 255)
    elif stress_level <= 85:
        return "Stressed", (0, 140, 255)
    else:
        return "Highly Stressed", (0, 0, 255)
if __name__ == "__main__":
    # Screen capture setup
    sct = mss.mss()
    monitor = sct.monitors[1]
    screen_width = monitor["width"]
    avoid_x_start = screen_width // 2  

    # OpenCV window setup
    window_name = "Stress Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  
    cv2.resizeWindow(window_name, 640, 480)
    cv2.moveWindow(window_name, avoid_x_start + 10, 50)

    logger.info("Screen stress detection system started successfully")
    start_time = time.time()

    try:
        while True:
            screen_frame = np.array(sct.grab({
                "top": monitor["top"],  
                "left": monitor["left"],  
                "width": avoid_x_start - 10,
                "height": monitor["height"]
            }))
            frame = cv2.cvtColor(screen_frame, cv2.COLOR_BGRA2BGR)
            current_time = time.time()
            
            results = yolo_model(frame)

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue  

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Preprocess for MobileNetV2
                    face_proc = cv2.resize(face, (224, 224))
                    face_proc = cv2.cvtColor(face_proc, cv2.COLOR_BGR2RGB)
                    face_proc = np.expand_dims(face_proc, axis=0) / 255.0

                    prediction = stress_model.predict(face_proc)
                    raw_conf = float(prediction[0][0])

                    stress_level = int((1 - raw_conf) * 100)  
                    display_label, box_color = get_stress_label(stress_level)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"{display_label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                    # Add stress bar
                    face_with_bar = draw_vertical_stress_bar(face, stress_level)

                    # Track stress for threshold check
                    person_key = (x1, y1, x2, y2)

                    if stress_level >= 60:  # Track any stress
                        if person_key not in stress_duration_tracker:
                            stress_duration_tracker[person_key] = current_time
                        else:
                            stressed_duration = current_time - stress_duration_tracker[person_key]

                            # If continuously stressed for 5 seconds
                            if stressed_duration >= STRESS_THRESHOLD:
                                if person_key not in stress_tracker:
                                    stress_tracker[person_key] = 0

                                # Check for cooldown
                                if current_time - stress_tracker[person_key] >= COOLDOWN_PERIOD:
                                    alert_type = display_label.upper()

                                    # Save image
                                    STRESS_IMAGE_DIR = "/home/yoga/STRESS_DETECTION/STRESSED/captured_stressed_image/"
                                    os.makedirs(STRESS_IMAGE_DIR, exist_ok=True)
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    image_filename = f"stress_{alert_type}_{timestamp}.jpg"
                                    image_path = os.path.join(STRESS_IMAGE_DIR, image_filename)
                                    cv2.imwrite(image_path, face_with_bar)

                                    # Send Email
                                    total_emails_sent += 1
                                    email_subject = f"ALERT: {alert_type} DETECTED"
                                    email_body = f"A person has been detected as {alert_type} with a stress level of {stress_level}.\n" \
                                                f"Stress detected for {stressed_duration:.1f} seconds."
                                    send_email_async(email_subject, email_body, attachment_path=image_path)

                                    logger.info(f"Email sent: {email_subject}, Stress Level: {stress_level}")
                                    stress_tracker[person_key] = current_time
                    else:
                        # Reset stress time if not stressed
                        stress_duration_tracker.pop(person_key, None)
                                
                    # Track previous email count to prevent false display updates
            previous_email_count = total_emails_sent

            # Inside the main loop, after email is sent:
            if total_emails_sent != previous_email_count:
                logger.info(f"Total Emails Sent Updated: {total_emails_sent}")
                previous_email_count = total_emails_sent

            # Display actual total emails sent
            cv2.putText(frame, f"Total Emails Sent: {total_emails_sent}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                                                                
            # Show frame
            cv2.imshow(window_name, frame)

            # Quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        logger.info("Shutting down screen stress detection system")
        cv2.destroyAllWindows()
