"""
retrain_alzheimer.py
====================
Improved Alzheimer MRI Classification training script.
Designed to run on Kaggle/Colab with GPU for ~2-3 hours.

Key improvements over original:
  1. Class weights to handle severe imbalance
  2. Strong data augmentation (rotation, flip, zoom, shift)
  3. Two-phase training: frozen base → fine-tune top layers
  4. ReduceLROnPlateau + EarlyStopping callbacks
  5. Full evaluation with confusion matrix + classification report

INSTRUCTIONS:
  - Upload this script to Kaggle or Colab
  - Make sure GPU is enabled
  - Upload your Alzheimer dataset to:
      /kaggle/input/alzheimer-dataset/Combined Dataset/
    with subfolders: train/ and test/
    each containing: Mild Impairment, Moderate Impairment,
                     No Impairment, Very Mild Impairment
  - Run all cells
  - Download the saved model: improved_alzheimer_model.keras
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — ADJUST PATHS IF NEEDED                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# --- FOR KAGGLE ---
# TRAIN_DIR = "/kaggle/input/alzheimer-dataset/Combined Dataset/train"
# TEST_DIR  = "/kaggle/input/alzheimer-dataset/Combined Dataset/test"

# --- FOR LOCAL ---
TRAIN_DIR = r"Datasets\Brain Tumor Datasets\Alzehimers Datasets\Alzeh datasets\Combined Dataset\train"
TEST_DIR  = r"Datasets\Brain Tumor Datasets\Alzehimers Datasets\Alzeh datasets\Combined Dataset\test"

IMG_SIZE     = 150          # Match your deployment input size
BATCH_SIZE   = 32
NUM_CLASSES  = 4
CLASSES      = ["Mild Impairment", "Moderate Impairment",
                "No Impairment", "Very Mild Impairment"]

MODEL_SAVE   = "improved_alzheimer_model.keras"

print("=" * 60)
print("  IMPROVED ALZHEIMER MODEL TRAINING")
print("=" * 60)

# Check GPU
gpus = tf.config.list_physical_devices("GPU")
print(f"\nGPU available: {len(gpus) > 0}")
if gpus:
    print(f"  GPU: {gpus[0].name}")
else:
    print("  WARNING: No GPU detected. Training will be slow (~6-8 hrs).")
    print("  Recommend: Run on Kaggle/Colab with GPU enabled.")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  1. DATA LOADING WITH HEAVY AUGMENTATION                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print("\n[1/5] Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest",
    validation_split=0.15,      # 15% of train for validation
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    subset="validation",
    shuffle=False,
    seed=42,
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    shuffle=False,
)

print(f"  Training samples  : {train_gen.samples}")
print(f"  Validation samples: {val_gen.samples}")
print(f"  Test samples      : {test_gen.samples}")
print(f"  Classes           : {CLASSES}")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  2. COMPUTE CLASS WEIGHTS                                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print("\n[2/5] Computing class weights...")

# Get class indices from training generator
train_labels = train_gen.classes
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights = dict(enumerate(weights))
print(f"  Class weights: {class_weights}")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  3. BUILD MODEL — VGG16 TRANSFER LEARNING                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print("\n[3/5] Building VGG16 transfer learning model...")

base_model = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# Freeze ALL base layers initially
base_model.trainable = False

# Build classification head
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)            # Better than Flatten for generalization
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trainable = sum(p.numpy().size for p in model.trainable_weights)
total     = sum(p.numpy().size for p in model.weights)
print(f"  Total params    : {total:,}")
print(f"  Trainable params: {trainable:,}")
print(f"  Frozen params   : {total - trainable:,}")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  4. TRAINING — PHASE 1 (FROZEN BASE) + PHASE 2 (FINE-TUNE)              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ── PHASE 1: Train head only (base frozen) ────────────────────────────────────
print("\n[4/5] PHASE 1 — Training classification head (base frozen)...")

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# ── PHASE 2: Unfreeze top layers of VGG16 and fine-tune ───────────────────────
print("\n[4/5] PHASE 2 — Fine-tuning top VGG16 layers...")

# Unfreeze the last 4 layers of VGG16 (block5_conv1 through block5_pool)
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with much lower LR for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),    # 100x smaller LR
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ft_trainable = sum(p.numpy().size for p in model.trainable_weights)
print(f"  Fine-tune trainable params: {ft_trainable:,}")

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  5. EVALUATION ON HELD-OUT TEST SET                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

print("\n[5/5] Evaluating on test set...")

# Load best saved model
best_model = keras.models.load_model(MODEL_SAVE)

# Predict
test_gen.reset()
y_pred_probs = best_model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

# Classification report
print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
print(report)

# Accuracy
acc = np.mean(y_pred == y_true)
print(f"  OVERALL TEST ACCURACY: {acc*100:.2f}%")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Alzheimer Classification - Confusion Matrix", fontsize=13)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("improved_alzheimer_confusion_matrix.png", dpi=300)
print("\nSaved -> improved_alzheimer_confusion_matrix.png")

# ── Training History Plot ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Combine histories
acc_hist = history1.history["accuracy"] + history2.history["accuracy"]
val_acc  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss_hist = history1.history["loss"] + history2.history["loss"]
val_loss  = history1.history["val_loss"] + history2.history["val_loss"]
phase1_epochs = len(history1.history["accuracy"])

ax1.plot(acc_hist, label="Train Accuracy")
ax1.plot(val_acc, label="Val Accuracy")
ax1.axvline(x=phase1_epochs, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(loss_hist, label="Train Loss")
ax2.plot(val_loss, label="Val Loss")
ax2.axvline(x=phase1_epochs, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("improved_alzheimer_training_history.png", dpi=300)
print("Saved -> improved_alzheimer_training_history.png")

print("\n" + "=" * 60)
print(f"  MODEL SAVED: {MODEL_SAVE}")
print(f"  TEST ACCURACY: {acc*100:.2f}%")
print("=" * 60)
print("\nDone! Copy improved_alzheimer_model.keras back to your project.")
