#this is imageupload.py
import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import time
from email_notification import send_email  # Import email function

# Load trained MobileNetV2 stress detection model
stress_model = tf.keras.models.load_model('/home/yoga/STRESS_DETECTION/STRESSED/models/mobilenetv2_stress.h5')

# Class labels
class_labels = ["Stressed", "Not Stressed"]

# Email Cooldown Configuration
COOLDOWN_PERIOD = 60  # Cooldown in seconds (1 minute)
last_email_time = 0  # Track last email timestamp

# Function to process image and predict stress
def predict_stress(image):
    global last_email_time  # Allow modifying global variable

    # Convert image to correct format (OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize and preprocess image for MobileNetV2
    face = cv2.resize(image, (224, 224))
    face = face / 255.0  # Normalize
    face = np.expand_dims(face, axis=0)  # Add batch dimension

    # Predict stress level
    prediction = stress_model.predict(face)
    confidence = float(prediction[0][0])
    
    label = "Not Stressed" if confidence >= 0.5 else "Stressed"
    confidence_percent = confidence * 100 if label == "Not Stressed" else (1 - confidence) * 100
    result = f"{label} ({confidence_percent:.2f}%)"

    # Send email if stressed and cooldown period has passed
    if label == "Stressed":
        current_time = time.time()
        if current_time - last_email_time >= COOLDOWN_PERIOD:
            send_email("ALERT: Stressed Person Detected", "A person has been detected as stressed in the uploaded image.")
            last_email_time = current_time  # Update last email timestamp

    return result

# Gradio Interface
if __name__ == "__main__":
    gr.Interface(
        fn=predict_stress,
        inputs=gr.Image(type="numpy"),
        outputs="text",
        title="Stress Detection from Image",
        description="Upload an image, and the model will predict whether the person is stressed or not.\nIf stress is detected, an email notification will be sent (with a cooldown).",
    ).launch()
