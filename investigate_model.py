"""
Deep investigation into model performance
This script will:
1. Check model architecture
2. Test with multiple tumor images
3. Analyze layer outputs
4. Check if model weights are properly loaded
"""
import tf_keras
import tensorflow as tf
import numpy as np
from PIL import Image
import os

print("="*60)
print("BRAIN TUMOR MODEL INVESTIGATION")
print("="*60)

# Load model
print("\n1. Loading model...")
model = tf_keras.models.load_model('best_model.keras', compile=False)
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Number of layers: {len(model.layers)}")

# Check model summary
print("\n2. Model architecture:")
model.summary()

# Check if model has been trained (weights should not be random)
print("\n3. Checking model weights...")
first_conv_layer = None
for layer in model.layers:
    if 'conv' in layer.name.lower():
        first_conv_layer = layer
        break

if first_conv_layer:
    weights = first_conv_layer.get_weights()[0]
    print(f"   First conv layer: {first_conv_layer.name}")
    print(f"   Weight shape: {weights.shape}")
    print(f"   Weight mean: {np.mean(weights):.6f}")
    print(f"   Weight std: {np.std(weights):.6f}")
    print(f"   Weight min: {np.min(weights):.6f}")
    print(f"   Weight max: {np.max(weights):.6f}")
    
    # Random weights would have very different statistics
    if abs(np.mean(weights)) < 0.01 and np.std(weights) < 0.5:
        print("   ⚠️ WARNING: Weights look suspiciously like random initialization!")

# Test with the user's image
print("\n4. Testing with user's tumor image...")
image = Image.open('test_tumor_image.jpg')
img = image.resize((299, 299))
img_array = np.array(img) / 255.0

if img_array.ndim == 2:
    img_array = np.stack((img_array,) * 3, axis=-1)
elif img_array.shape[-1] == 4:
    img_array = img_array[..., :3]

img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array, verbose=0)
labels = ['pituitary', 'notumor', 'meningioma', 'glioma']

print(f"\n   Predictions:")
for idx, label in enumerate(labels):
    prob = predictions[0][idx]
    marker = " ← PREDICTED" if idx == np.argmax(predictions[0]) else ""
    print(f"     {label:12s}: {prob:.4f} ({prob*100:5.2f}%){marker}")

predicted_label = labels[np.argmax(predictions[0])]
print(f"\n   Final prediction: {predicted_label}")

# Check if there are any sample images in the directory
print("\n5. Looking for other test images...")
image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"   Found {len(image_files)} image files")

print("\n" + "="*60)
print("INVESTIGATION COMPLETE")
print("="*60)
