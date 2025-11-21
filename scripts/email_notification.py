#this is email_notification.py
import smtplib
import time
import os
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "yoganand373@gmail.com"
SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Get password from environment variable
RECEIVER_EMAIL = "bluecifer636@gmail.com"

# Cooldown mechanism to prevent spam
EMAIL_COOLDOWN = 10  # Send email only once every 5 minutes
last_email_time = 0  # Keeps track of last email sent time

def send_email(subject, body, attachment_path=None):
    global last_email_time
    current_time = time.time()
    
    # Check cooldown
    if current_time - last_email_time < EMAIL_COOLDOWN:
        print("Skipping email to prevent spam")
        return

    if not SENDER_PASSWORD:
        print("Error: EMAIL_PASSWORD environment variable not set")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Attach image if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
                msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        last_email_time = current_time  # Update last email sent time
        print("Email sent successfully with attachment" if attachment_path else "Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Example usage
if __name__ == "__main__":
    send_email("Stress Alert!", "A person has been detected as stressed for an extended period.", "path/to/image.jpg")
