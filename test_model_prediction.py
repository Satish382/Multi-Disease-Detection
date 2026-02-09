"""
Test the brain tumor model with the user's MRI scan
"""
import tf_keras
import numpy as np
from PIL import Image
import json
import sys

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the model
print("Loading brain tumor model...")
model = tf_keras.models.load_model('best_model.keras', compile=False)
print(f"Model loaded successfully")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Load and preprocess the test image
print("\nLoading test image...")
image = Image.open('test_tumor_image.jpg')
print(f"Original image size: {image.size}")
print(f"Original image mode: {image.mode}")

# Preprocess exactly as app.py does
img = image.resize((299, 299))
img_array = np.array(img) / 255.0

print(f"\nAfter resize and normalization:")
print(f"Array shape: {img_array.shape}")
print(f"Array dtype: {img_array.dtype}")
print(f"Array min: {img_array.min()}, max: {img_array.max()}")

# Handle grayscale/RGBA
if img_array.ndim == 2:
    img_array = np.stack((img_array,) * 3, axis=-1)
    print("Converted grayscale to RGB")
elif img_array.shape[-1] == 4:
    img_array = img_array[..., :3]
    print("Removed alpha channel")

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)
print(f"Final input shape: {img_array.shape}")

# Make prediction
print("\nMaking prediction...")
predictions = model.predict(img_array)
print(f"Raw predictions shape: {predictions.shape}")

# Labels from app.py
brain_tumor_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']

# Get predicted class
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"\nPredicted class index: {predicted_class}")
print(f"Predicted label: {brain_tumor_labels[predicted_class]}")
print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

print("\nAll class probabilities:")
for idx, label in enumerate(brain_tumor_labels):
    prob = predictions[0][idx]
    print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")

# Check if this matches what the user reported
if brain_tumor_labels[predicted_class] == 'notumor' and confidence > 0.8:
    print("\nWARNING: Model is incorrectly predicting 'no tumor' with high confidence!")
    print("This confirms the user's report.")
else:
    print(f"\nModel prediction: {brain_tumor_labels[predicted_class]} with {confidence*100:.2f}% confidence")
