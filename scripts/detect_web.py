#this is detect_web.py
import os

# point Qt at the real xcb plugin, not “offscreen”
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
os.environ["QT_QPA_PLATFORM"] = "xcb"


import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time
import threading
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime



#######################
# 1) Enhanced Logging System
#######################
class StressDetectionLogger:
    """Enhanced logging system for stress detection application"""
    
    def __init__(self, log_dir="logs", enable_console=True):
        """Initialize the logging system"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("stress_detection")
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        standard_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        json_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for general logs (rotating by size - 5MB)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "stress_detection.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(standard_formatter)
        self.logger.addHandler(file_handler)
        
        # File handler for errors (daily rotation)
        error_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, "errors.log"),
            when="midnight",
            interval=1,
            backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(standard_formatter)
        self.logger.addHandler(error_handler)
        
        # File handler for alerts/events (in JSON format)
        self.event_logger = logging.getLogger("stress_events")
        self.event_logger.setLevel(logging.INFO)
        
        if self.event_logger.handlers:
            self.event_logger.handlers.clear()
            
        event_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, "stress_events.json"),
            when="midnight", 
            interval=1,
            backupCount=90
        )
        event_handler.setFormatter(json_formatter)
        self.event_logger.addHandler(event_handler)
        
        # Console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(standard_formatter)
            self.logger.addHandler(console_handler)
            
        self.logger.info("Logging system initialized")
    
    def log_event(self, event_type, data):
        """Log structured event data in JSON format"""
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data
        }
        self.event_logger.info(json.dumps(event_data))
    
    def log_stress_detected(self, person_id, stress_label, duration, stress_percentage, box_coords=None):
        """Log when stress is detected"""
        self.log_event("stress_detected", {
            "person_id": person_id,
            "stress_label": stress_label,
            "stress_percentage": stress_percentage,
            "duration": duration,
            "box_coords": box_coords
        })
        self.logger.info(f"Person {person_id} detected as {stress_label} ({stress_percentage:.1f}%) for {duration:.1f}s")

        
        
    
    def log_email_sent(self, person_id, stress_level, email_count, image_path=None):
        """Log when an email alert is sent"""
        self.log_event("email_sent", {
            "person_id": person_id,
            "stress_level": stress_level,
            "email_count": email_count,
            "image_path": image_path
        })
        self.logger.info(f"Email #{email_count} sent for Person {person_id} ({stress_level})")
    
    def log_system_stats(self, uptime, tracked_count, email_count):
        """Log system statistics periodically"""
        self.log_event("system_stats", {
            "uptime_minutes": uptime,
            "people_tracked": tracked_count,
            "total_emails_sent": email_count
        })
        self.logger.info(f"System stats: {uptime:.1f}min uptime, {tracked_count} people tracked, {email_count} emails sent")
    
    def log_startup(self, config):
        """Log system startup with configuration"""
        self.log_event("system_startup", {
            "configuration": config
        })
        self.logger.info(f"System started with config: {config}")
    
    def log_shutdown(self, uptime, email_count):
        """Log system shutdown"""
        self.log_event("system_shutdown", {
            "uptime_minutes": uptime,
            "total_emails_sent": email_count
        })
        self.logger.info(f"System shutdown after {uptime:.1f}min with {email_count} emails sent")
    
    def error(self, message, exc_info=False):
        """Log error messages"""
        self.logger.error(message, exc_info=exc_info)
    
    def warning(self, message):
        """Log warning messages"""
        self.logger.warning(message)
    
    def info(self, message):
        """Log info messages"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug messages"""
        self.logger.debug(message)

#######################
# 2) Email Helper
#######################
def send_email_async(send_email_func, subject, body, attachment_path=None):
    """Send email in a background thread"""
    thread = threading.Thread(target=send_email_func, args=(subject, body, attachment_path))
    thread.daemon = True  # Make thread daemon so it doesn't block program exit
    thread.start()
    return thread

#######################
# 3) Main Stress Detection System
#######################
class StressDetectionSystem:
    def __init__(self, config):
        """Initialize the stress detection system"""
        self.config = config
        
        # Set up logger
        self.logger = StressDetectionLogger(log_dir=config["LOG_DIR"])
        self.logger.log_startup(config)
        
        # Load models
        self.logger.info("Loading models...")
        self.yolo_model = YOLO(config["YOLO_MODEL_PATH"])
        self.stress_model = tf.keras.models.load_model(config["STRESS_MODEL_PATH"])
        self.logger.info("Models loaded successfully")
        
        # Initialize trackers
        self.people_tracker = {}  # Format: {person_id: person_data}
        self.total_emails_sent = 0
        self.start_time = time.time()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(config["CAMERA_ID"])
        if not self.cap.isOpened():
            self.logger.error("Failed to open camera")
            raise RuntimeError("Failed to open camera")
        
        # Import email function
        try:
            from email_notification import send_email
            self.send_email = send_email
        except ImportError:
            self.logger.error("Failed to import email_notification module")
            self.send_email = None
            
        self.logger.info("Stress detection system initialized")
    
    def _find_closest_person(self, face_center, max_distance=70):
        """Find the closest tracked person to the given face center"""
        person_id = None
        min_distance = max_distance
        
        for pid, data in self.people_tracker.items():
            prev_center = data["position"]
            distance = np.sqrt((prev_center[0] - face_center[0])**2 + (prev_center[1] - face_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                person_id = pid
                
        return person_id
    
    def _get_stress_label(self, raw_conf):
        """
        Determine stress label, color, and level based on model confidence
        raw_conf: sigmoid output from model (0-1)
        Lower raw_conf = higher stress
        """
        stress_level = (1 - raw_conf) * 100  # Convert to percentage (0-100)
        
        if stress_level <= 50:
            return "Relaxed", (0, 255, 0), stress_level
        elif stress_level <= 60:
            return "Mildly Stressed", (0, 255, 255), stress_level
        elif stress_level <= 85:
            return "Stressed", (0, 0, 255), stress_level
        else:
            return "Highly Stressed", (0, 0, 150), stress_level
    
    def _draw_horizontal_stress_bar(self, img, x1, y1, stress_indicator, bar_length=100, bar_height=10):
        """Draw a horizontal stress indicator bar"""
        filled_length = int(bar_length * stress_indicator)
        bar_x = x1
        bar_y = y1 - 40  # Move higher above the label (was y1 - 25)
        
        # Draw outline
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (255, 255, 255), 2)
        
        # Determine fill color (green to red)
        red = int(255 * stress_indicator)
        green = int(255 * (1 - stress_indicator))
        fill_color = (0, green, red)  # BGR
        
        # Fill bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled_length, bar_y + bar_height), fill_color, -1)
    
    def _send_stress_alert(self, person_id, stress_label, stress_level, stressed_duration, image_path):
        """Send an email alert about a stressed person"""
        if self.send_email is None:
            self.logger.warning("Email function not available, skipping alert")
            return False
            
        self.total_emails_sent += 1
        email_count = self.people_tracker[person_id]["email_count"] + 1
        self.people_tracker[person_id]["email_count"] = email_count
        
        email_subject = f"ALERT: {stress_label} DETECTED"
        email_body = (
            f"Person {person_id} has been detected as {stress_label} with a stress level of {stress_level:.0f}%.\n"
            f"Stress detected for {stressed_duration:.1f} seconds.\n"
            f"This is alert #{email_count} for this person."
        )
        
        try:
            send_email_async(self.send_email, email_subject, email_body, attachment_path=image_path)
            self.logger.log_email_sent(person_id, stress_label, email_count, image_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}", exc_info=True)
            return False
    
    def _save_face_image(self, frame, x1, y1, x2, y2, stress_label):
        """Save an image of the detected stressed face"""
        # Ensure the directory exists
        os.makedirs(self.config["STRESS_IMAGE_DIR"], exist_ok=True)
        
        # Save the face region with some padding
        face_height = y2 - y1
        padding_y = int(0.2 * face_height)
        padding_x = int(0.2 * (x2 - x1))
        
        # Calculate padded coordinates with bounds checking
        y1_pad = max(0, y1 - padding_y - 30)  # Extra space for stress bar
        y2_pad = min(frame.shape[0], y2 + padding_y)
        x1_pad = max(0, x1 - padding_x)
        x2_pad = min(frame.shape[1], x2 + padding_x)
        
        face_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Skip if image is empty
        if face_img.size == 0:
            self.logger.warning("Skipping image save: cropped region is empty")
            return None
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stress_type = stress_label.replace(" ", "_")
        image_filename = f"stress_{stress_type}_{timestamp}.jpg"
        image_path = os.path.join(self.config["STRESS_IMAGE_DIR"], image_filename)
        
        # Save image
        cv2.imwrite(image_path, face_img)
        self.logger.info(f"Saved stressed face image: {image_path}")
        return image_path
    
    def process_frame(self, frame):
        """Process a single frame from the camera"""
        current_time = time.time()
        uptime_minutes = (current_time - self.start_time) / 60
        
        # Run YOLOv8 detection
        results = self.yolo_model(frame)
        detected_person_ids = []
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # If not a person, mark as "Object" and continue
                if cls != 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
                    cv2.putText(frame, "Object", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    continue
                
                # Process person detection
                # Adjust face region within the bounding box
                box_height = y2 - y1
                face_y1 = y1 + int(0.15 * box_height)
                face_y2 = y2 - int(0.10 * box_height)
                face = frame[face_y1:face_y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                # Calculate face center for tracking
                face_center = ((x1 + x2) // 2, (face_y1 + face_y2) // 2)
                
                # Find or create person ID
                person_id = self._find_closest_person(face_center)
                if person_id is None:
                    person_id = len(self.people_tracker) + 1
                    self.logger.info(f"New person detected: ID {person_id}")
                    self.people_tracker[person_id] = {
                        "position": face_center,
                        "stress_start_time": 0,
                        "last_alert_time": 0,
                        "last_seen": current_time,
                        "stress_level": "Unknown",
                        "email_count": 0,
                        "previous_bounding_box": (x1, face_y1, x2, face_y2)
                        
                    }
                
                # Update tracking data
                prev_bbox = self.people_tracker[person_id]["previous_bounding_box"]
                # Smooth bounding box transition (30% current + 70% previous)
                smooth_x1 = int(0.3 * x1 + 0.7 * prev_bbox[0])
                smooth_y1 = int(0.3 * face_y1 + 0.7 * prev_bbox[1])
                smooth_x2 = int(0.3 * x2 + 0.7 * prev_bbox[2])
                smooth_y2 = int(0.3 * face_y2 + 0.7 * prev_bbox[3])
                
                # Update position and bounding box
                self.people_tracker[person_id]["position"] = face_center
                self.people_tracker[person_id]["last_seen"] = current_time
                self.people_tracker[person_id]["previous_bounding_box"] = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
                detected_person_ids.append(person_id)
                
                # Preprocess face for stress detection
                face_resized = cv2.resize(face, (224, 224))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_input = np.expand_dims(face_rgb, axis=0) / 255.0
                
                # Run stress detection
                prediction = self.stress_model.predict(face_input, verbose=0)
                raw_conf = float(prediction[0][0])
                
                # Get stress label and level
                stress_label, box_color, stress_level = self._get_stress_label(raw_conf)
                stress_indicator = stress_level / 100.0
                
                # Draw smoothed bounding box & label
                cv2.rectangle(frame, (smooth_x1, smooth_y1), (smooth_x2, smooth_y2), box_color, 2)
                cv2.putText(frame, f"{stress_label}", (smooth_x1, smooth_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                # Draw horizontal stress bar
                self._draw_horizontal_stress_bar(frame, smooth_x1, smooth_y1, stress_indicator)
                
                # Update person's stress information
                self.people_tracker[person_id]["stress_level"] = stress_label
                
                # Check if person is stressed enough to trigger an alert
                if stress_level >= self.config["STRESS_THRESHOLD_PERCENT"]:
                    if self.people_tracker[person_id].get("stress_start_time") == 0:
                        # Just became stressed, start timer
                        self.people_tracker[person_id]["stress_start_time"] = current_time
                    
                    # Calculate stress duration
                    stress_duration = current_time - self.people_tracker[person_id]["stress_start_time"]
                    time_since_last_alert = current_time - self.people_tracker[person_id]["last_alert_time"]
                    
                    # If stressed long enough and cooldown expired
                    if (stress_duration >= self.config["STRESS_DURATION_THRESHOLD"] and 
                            time_since_last_alert >= self.config["COOLDOWN_PERIOD"]):
                        
                        # Save image
                        image_path = self._save_face_image(frame, smooth_x1, smooth_y1, smooth_x2, smooth_y2, stress_label)
                        
                        if image_path:
                            # Send alert
                            self._send_stress_alert(
                                person_id, 
                                stress_label, 
                                stress_level, 
                                stress_duration, 
                                image_path
                            )
                            
                            # Update alert time for cooldown
                            self.people_tracker[person_id]["last_alert_time"] = current_time
                            self.logger.log_stress_detected(
                                person_id, 
                                stress_label, 
                                stress_duration,
                                stress_level,  # This is the numeric percentage
                                (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
                            )

                else:
                    # Reset stress timer if not stressed
                    self.people_tracker[person_id]["stress_start_time"] = 0
                    if self.people_tracker[person_id].get("stress_level") != stress_label:
                        self.logger.log_stress_detected(
                            person_id,
                            stress_label,
                            duration=0,
                            stress_percentage=stress_level,
                            box_coords=(smooth_x1, smooth_y1, smooth_x2, smooth_y2)
                        )
        
        
        
        # Add status display for each person
                # Remove people that haven't been seen recently
        for pid in list(self.people_tracker.keys()):
            if pid not in detected_person_ids:
                if current_time - self.people_tracker[pid]["last_seen"] > self.config["MAX_ABSENCE_TIME"]:
                    self.logger.info(f"Person {pid} removed from tracking (absent for {current_time - self.people_tracker[pid]['last_seen']:.1f}s)")
                    del self.people_tracker[pid]
        # Add status display for each person (only show for first 2 persons)
        y_pos = 30
        displayed = 0
        for pid, data in self.people_tracker.items():
            if displayed >= 2:
                break
            cooldown_remaining = max(0, self.config["COOLDOWN_PERIOD"] - (current_time - data["last_alert_time"]))
            status_text = f"Person {pid}: {data['stress_level']}, Emails: {data['email_count']}, Cooldown: {cooldown_remaining:.1f}s"
            cv2.putText(frame, status_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25
            displayed += 1

        
        # Add system status at bottom right
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status_text = f"Uptime: {uptime_minutes:.1f} min | {timestamp}"
        text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = frame.shape[0] - 10
        cv2.putText(frame, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Periodically log system stats
        if int(current_time) % 60 == 0 and int(current_time) != int(self.last_log_time) if hasattr(self, 'last_log_time') else True:
            self.logger.log_system_stats(
                uptime_minutes,
                len(self.people_tracker),
                self.total_emails_sent
            )
            self.last_log_time = current_time
            
        return frame
        
    def run(self):
        """Run the stress detection system"""
        self.logger.info("Starting stress detection system")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow("Stress Detection", processed_frame)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            self.logger.error(f"Unhandled exception: {e}", exc_info=True)
        finally:
            uptime_minutes = (time.time() - self.start_time) / 60
            self.logger.log_shutdown(uptime_minutes, self.total_emails_sent)
            self.cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Stress detection system shut down")

#######################
# 4) Main Application
#######################
if __name__ == "__main__":
    # Configuration
    config = {
        "YOLO_MODEL_PATH": "/home/yoga/STRESS_DETECTION/STRESSED/models/yolov8n.pt",
        "STRESS_MODEL_PATH": "/home/yoga/STRESS_DETECTION/STRESSED/models/mobilenetv2_stress.h5",
        "CAMERA_ID": 0,
        "STRESS_IMAGE_DIR": "/home/yoga/STRESS_DETECTION/STRESSED/captured_stressed/",
        "LOG_DIR": "/home/yoga/STRESS_DETECTION/STRESSED/logs/",
        "STRESS_THRESHOLD_PERCENT": 60,  # Stress percentage to start tracking
        "STRESS_DURATION_THRESHOLD": 2,   # Must be stressed for 5 seconds
        "COOLDOWN_PERIOD": 10,            # 10 seconds between alerts
        "MAX_ABSENCE_TIME": 5             # Remove tracking after 5 seconds absent
    }
    
    # Create and run the stress detection system
    system = StressDetectionSystem(config)
    system.run()