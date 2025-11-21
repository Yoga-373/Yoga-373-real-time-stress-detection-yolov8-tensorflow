import os
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
    class_mode="binary",     # binary labels
    color_mode="rgb",
    shuffle=False            # important: keep order for labels ↔ predictions
)

# ─── Load Model ─────────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

# ─── Predict on Test Set ────────────────────────────────────────────────────────
# This yields an array of shape (num_samples, 1) with probabilities in [0,1]
y_prob = model.predict(test_generator, verbose=1)

# Binarize at 0.5 threshold
y_pred = (y_prob > 0.5).astype(int).flatten()

# True labels (0 or 1)
y_true = test_generator.classes

# ─── Map Indices ↔ Class Names ───────────────────────────────────────────────────
# e.g. {0: "not_stressed", 1: "stressed"}
inv_class_map = {v: k for k, v in test_generator.class_indices.items()}

# (Optional) convert numeric labels → names
y_true_names = [inv_class_map[i] for i in y_true]
y_pred_names = [inv_class_map[i] for i in y_pred]

# ─── Classification Report ──────────────────────────────────────────────────────
print("Classification Report:")
print(classification_report(
    y_true, 
    y_pred, 
    target_names=[inv_class_map[0], inv_class_map[1]]
))

# ─── Confusion Matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# You can pretty‐print it if you like:
#   [[TN, FP],
#    [FN, TP]]
