import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Paths
BASE_DIR = "/home/yoga/stressed/dataset/FER2013/"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Emotion to Stress Mapping
stressed_classes = ["angry", "sad", "fear", "disgust"]
not_stressed_classes = ["happy", "neutral", "surprise"]

# Parameters
IMG_SIZE = (224, 224)  # Resize for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",  # Change to binary classification
    color_mode="rgb",
    classes={"stressed": stressed_classes, "not_stressed": not_stressed_classes}
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb"  # MobileNetV2 expects 3 channels
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb"
)



# Load MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
output_layer = Dense(1, activation="sigmoid")(x)  # Binary classification

# Create model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    verbose=1
)

# Save trained model
model.save("/home/yoga/stressed/models/mobilenetv2_stress.h5")

print("Training complete. Model saved!")
