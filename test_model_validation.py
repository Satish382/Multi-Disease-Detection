"""
Test script to verify if AI models are properly validating inputs
or accepting any image type (including non-medical images)
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Model paths
BRAIN_TUMOR_MODEL = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzheimer_best_model.keras"
ALZHEIMER_MODEL = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzimer_federated_model.keras"

# Labels
brain_tumor_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']
alzheimer_labels = ['No Impairment', 'Very Mild Impairment', 'Moderate Impairment', 'Mild Impairment']

def preprocess_image(image):
    """Same preprocessing as in app.py"""
    img = image.resize((299, 299))
    img_array = np.array(img) / 255.0
    
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def test_model_with_random_images():
    """Test models with various non-medical images"""
    
    print("=" * 70)
    print("TESTING MODEL VALIDATION")
    print("=" * 70)
    
    # Check if models exist
    print("\n1. Checking model files...")
    brain_exists = os.path.exists(BRAIN_TUMOR_MODEL)
    alzheimer_exists = os.path.exists(ALZHEIMER_MODEL)
    
    print(f"   Brain Tumor Model: {'✓ Found' if brain_exists else '✗ Not Found'}")
    print(f"   Alzheimer Model: {'✓ Found' if alzheimer_exists else '✗ Not Found'}")
    
    if not brain_exists or not alzheimer_exists:
        print("\n❌ ERROR: Model files not found!")
        return
    
    # Load models
    print("\n2. Loading models...")
    try:
        brain_model = tf.keras.models.load_model(BRAIN_TUMOR_MODEL)
        print("   ✓ Brain Tumor Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load Brain Tumor Model: {e}")
        brain_model = None
    
    try:
        alzheimer_model = tf.keras.models.load_model(ALZHEIMER_MODEL)
        print("   ✓ Alzheimer Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load Alzheimer Model: {e}")
        alzheimer_model = None
    
    if brain_model is None or alzheimer_model is None:
        print("\n❌ ERROR: Failed to load models!")
        return
    
    # Create test images (random noise, solid colors, patterns)
    print("\n3. Creating test images (non-medical)...")
    test_cases = []
    
    # Random noise image
    random_img = Image.fromarray(np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8))
    test_cases.append(("Random Noise", random_img))
    
    # Solid red image
    red_img = Image.new('RGB', (299, 299), color='red')
    test_cases.append(("Solid Red", red_img))
    
    # Solid blue image
    blue_img = Image.new('RGB', (299, 299), color='blue')
    test_cases.append(("Solid Blue", blue_img))
    
    # Black image
    black_img = Image.new('RGB', (299, 299), color='black')
    test_cases.append(("Black", black_img))
    
    # White image
    white_img = Image.new('RGB', (299, 299), color='white')
    test_cases.append(("White", white_img))
    
    print(f"   Created {len(test_cases)} test images")
    
    # Test Brain Tumor Model
    print("\n" + "=" * 70)
    print("BRAIN TUMOR MODEL PREDICTIONS")
    print("=" * 70)
    
    for name, img in test_cases:
        img_array = preprocess_image(img)
        predictions = brain_model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print(f"\n{name} Image:")
        print(f"  Predicted: {brain_tumor_labels[predicted_class]}")
        print(f"  Confidence: {confidence*100:.2f}%")
        print(f"  All predictions:")
        for idx, label in enumerate(brain_tumor_labels):
            print(f"    - {label}: {predictions[0][idx]*100:.2f}%")
    
    # Test Alzheimer Model
    print("\n" + "=" * 70)
    print("ALZHEIMER MODEL PREDICTIONS")
    print("=" * 70)
    
    for name, img in test_cases:
        img_array = preprocess_image(img)
        predictions = alzheimer_model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print(f"\n{name} Image:")
        print(f"  Predicted: {alzheimer_labels[predicted_class]}")
        print(f"  Confidence: {confidence*100:.2f}%")
        print(f"  All predictions:")
        for idx, label in enumerate(alzheimer_labels):
            print(f"    - {label}: {predictions[0][idx]*100:.2f}%")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
⚠️  CRITICAL FINDING:

The models are accepting ANY image input without validation!

PROBLEM:
- The current implementation has NO input validation
- Models will predict on ANY image (faces, random noise, solid colors)
- There's no check to verify if the image is actually a brain scan

WHY THIS HAPPENS:
1. Deep learning models accept any 299x299x3 image tensor
2. No pre-validation layer to check image content
3. No medical image classifier before the diagnosis model

SOLUTIONS:
1. Add a pre-classifier to detect if image is a brain scan
2. Implement image quality checks (contrast, brightness)
3. Use anomaly detection to reject non-medical images
4. Add user warnings about upload requirements
5. Implement confidence thresholds (reject low confidence)
6. Add DICOM format support for real medical scans
    """)

if __name__ == "__main__":
    test_model_with_random_images()
