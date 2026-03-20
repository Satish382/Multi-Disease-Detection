"""
evaluate_model.py
-----------------
Evaluates the federated brain-tumor model on the local test set.

Steps:
  1. Load new_tumor_model.keras  (falls back to federated_model.keras)
  2. Run predictions on the Testing/ folder
  3. Print classification report (precision / recall / F1)
  4. Plot confusion matrix with matplotlib and save as confusion_matrix.png (300 DPI)

Usage:
  python evaluate_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


# ── Configuration ────────────────────────────────────────────────────────────

# Model priority: use whichever file exists first
MODEL_CANDIDATES = [
    "new_tumor_model.keras",
    "federated_model.keras",
    "federated_models.keras",
    "best_model.keras",
]

TEST_DIR = r"Datasets\Brain Tumor Datasets\Brain datasets\Testing"

IMAGE_SIZE = (299, 299)   # must match training resolution

# Class order must match training label encoding
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

OUTPUT_PNG = "confusion_matrix.png"
DPI = 300

# ── Load model ────────────────────────────────────────────────────────────────

model_path = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        model_path = candidate
        break

if model_path is None:
    raise FileNotFoundError(
        "No model file found. Looked for: " + ", ".join(MODEL_CANDIDATES)
    )

print(f"[INFO] Loading model from '{model_path}' ...")
model = tf.keras.models.load_model(model_path, compile=False)
print("[INFO] Model loaded.\n")

# ── Build test dataset paths ──────────────────────────────────────────────────

test_paths  = []
test_labels = []

for label in CLASS_NAMES:
    class_dir = os.path.join(TEST_DIR, label)
    if not os.path.isdir(class_dir):
        print(f"[WARN] Class folder not found: {class_dir}")
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            test_paths.append(os.path.join(class_dir, fname))
            test_labels.append(label)

print(f"[INFO] Found {len(test_paths)} test images across {len(CLASS_NAMES)} classes.")
for cls in CLASS_NAMES:
    count = test_labels.count(cls)
    print(f"       {cls:15s}: {count}")
print()

# ── Preprocessing (matches notebook logic) ───────────────────────────────────

def preprocess(path):
    """Load, resize, normalise → (1, 299, 299, 3)."""
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Predict ───────────────────────────────────────────────────────────────────

print("[INFO] Running predictions ...")
y_pred = []
y_true = []

total = len(test_paths)
for idx, (path, label) in enumerate(zip(test_paths, test_labels)):
    img_array = preprocess(path)
    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    y_pred.append(CLASS_NAMES[pred_idx])
    y_true.append(label)
    if (idx + 1) % 50 == 0 or (idx + 1) == total:
        print(f"  {idx + 1}/{total} images processed...", flush=True)

print("[INFO] Predictions complete.\n")

# ── Classification Report ─────────────────────────────────────────────────────

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ── Confusion Matrix ──────────────────────────────────────────────────────────

cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Tick labels
tick_positions = np.arange(len(CLASS_NAMES))
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(CLASS_NAMES, fontsize=10)

# Annotate cells with counts
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        colour = "white" if cm[i, j] > thresh else "black"
        ax.text(
            j, i,
            format(cm[i, j], "d"),
            ha="center", va="center",
            color=colour, fontsize=11, fontweight="bold"
        )

ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax.set_ylabel("True Label", fontsize=12, labelpad=10)
ax.set_title("Brain Tumor — Confusion Matrix", fontsize=13, pad=14)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=DPI)
plt.show()
print(f"\n[INFO] Confusion matrix saved → '{OUTPUT_PNG}' at {DPI} DPI")
