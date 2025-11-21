import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# ─── Paths & Params ─────────────────────────────────────────────────────────────
BASE_DIR    = "/home/yoga/stressed/dataset/FER2013/"
TEST_DIR    = os.path.join(BASE_DIR, "test")
MODEL_PATH  = "/home/yoga/stressed/models/mobilenetv2_stress.h5"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

# ─── Prepare Test Generator ─────────────────────────────────────────────────────
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False  # Maintain order for correct labels
)

# ─── Load Model ─────────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

# ─── Inference & FPS Measurement ────────────────────────────────────────────────
total_samples = test_generator.samples
print(f"\nStarting inference on {total_samples} test images...")

# Time the prediction process
start_time = time.time()
y_prob = model.predict(test_generator, verbose=1)
end_time = time.time()

# Calculate performance metrics
inference_time = end_time - start_time
fps = total_samples / inference_time

# ─── Performance Report ──────────────────────────────────────────────────────────
print("\n╭─────────────────────────────────────────────╮")
print("│               Performance Metrics           │")
print("├─────────────────────────────────────────────┤")
print(f"│ Frames Per Second (FPS)  │ {fps:>15.2f} │")
print(f"│ Total Inference Time     │ {inference_time:>15.2f} sec │")
print(f"│ Processed Images         │ {total_samples:>15} │")
print(f"│ Batch Size               │ {BATCH_SIZE:>15} │")
print("╰─────────────────────────────────────────────╯")

# ─── Model Evaluation ───────────────────────────────────────────────────────────
# Convert probabilities to class predictions
y_pred = (y_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Generate classification report
class_names = list(test_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Generate confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nRows: True | Columns: Predicted")