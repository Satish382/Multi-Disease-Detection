"""
evaluate_alzheimer.py
---------------------
Evaluates the Alzheimer classification model on the local test set.

Steps:
  1. Load alzheimer_best_model.keras (falls back to new_alzheimer_model.keras)
  2. Predict on the test/ folder
  3. Print classification report (precision / recall / F1 / accuracy)
  4. Plot confusion matrix with matplotlib and save as
     alzheimer_confusion_matrix.png at 300 DPI (IEEE column-width safe)

Usage:
  python evaluate_alzheimer.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CANDIDATES = [
    "new_alzheimer_model.keras",     # primary (150×150)
    "alzheimer_best_model.keras",    # secondary (299×299)
]

TEST_DIR = (
    r"Datasets\Brain Tumor Datasets"
    r"\Alzehimers Datasets\Alzeh datasets\Combined Dataset\test"
)

# Class names must match the sub-folder names exactly
CLASS_NAMES = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]

OUTPUT_PNG = "alzheimer_confusion_matrix.png"
DPI = 300

# ── Load model ────────────────────────────────────────────────────────────────

model_path = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        model_path = candidate
        break

if model_path is None:
    raise FileNotFoundError(
        "No Alzheimer model found. Looked for: " + ", ".join(MODEL_CANDIDATES)
    )

print(f"[INFO] Loading model from '{model_path}' ...")
model = tf.keras.models.load_model(model_path, compile=False)

# Infer input size from the model's first layer
input_shape = model.input_shape  # e.g. (None, 299, 299, 3) or (None, 150, 150, 3)
IMAGE_SIZE = (input_shape[1], input_shape[2])
print(f"[INFO] Model loaded. Input size: {IMAGE_SIZE}\n")

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
    print(f"       {cls:25s}: {test_labels.count(cls)}")
print()

if len(test_paths) == 0:
    raise RuntimeError("No test images found. Check TEST_DIR path.")

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(path, size):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Predict ───────────────────────────────────────────────────────────────────

print("[INFO] Running predictions ...")
y_pred = []
y_true = []
total = len(test_paths)

for idx, (path, label) in enumerate(zip(test_paths, test_labels)):
    arr = preprocess(path, IMAGE_SIZE)
    preds = model.predict(arr, verbose=0)
    pred_idx = np.argmax(preds[0])
    y_pred.append(CLASS_NAMES[pred_idx])
    y_true.append(label)
    if (idx + 1) % 50 == 0 or (idx + 1) == total:
        print(f"  {idx + 1}/{total} images processed ...", flush=True)

print("[INFO] Predictions complete.\n")

# ── Classification Report ─────────────────────────────────────────────────────

print("=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ── Confusion Matrix ──────────────────────────────────────────────────────────

cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

# IEEE single-column ≈ 3.5 in; double-column ≈ 7.16 in.
# 5×4.5 gives a clean square-ish plot that fits a double-column IEEE paper.
fig, ax = plt.subplots(figsize=(5, 4.5))

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)

tick_positions = np.arange(len(CLASS_NAMES))
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)

# Shorter tick labels for readability (remove " Impairment" suffix)
short_labels = [n.replace(" Impairment", "") for n in CLASS_NAMES]
ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
ax.set_yticklabels(short_labels, fontsize=8)

# Annotate each cell
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        colour = "white" if cm[i, j] > thresh else "black"
        ax.text(
            j, i,
            format(cm[i, j], "d"),
            ha="center", va="center",
            color=colour, fontsize=9, fontweight="bold",
        )

ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
ax.set_ylabel("True Label",      fontsize=10, labelpad=8)
ax.set_title("Alzheimer's Disease — Confusion Matrix", fontsize=11, pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
print(f"[INFO] Confusion matrix saved → '{OUTPUT_PNG}' ({DPI} DPI)")
plt.show()
