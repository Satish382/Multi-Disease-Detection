"""
Test all available tumor models to find which one works best
"""
import tf_keras
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def test_model(model_path, image_path):
    """Test a model with the given image"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"File not found!")
        return None
    
    # Check file size
    size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"File size: {size_mb:.2f} MB")
    
    # Try to load
    try:
        # Try with tf_keras first (for H5 files)
        try:
            model = tf_keras.models.load_model(model_path, compile=False)
            print("Loaded with tf_keras (H5 format)")
        except:
            # Try with tf.keras (for Keras 3 format)
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Loaded with tf.keras (Keras 3 format)")
        
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Load and preprocess image
        image = Image.open(image_path)
        img = image.resize((299, 299))
        img_array = np.array(img) / 255.0
        
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        labels = ['pituitary', 'notumor', 'meningioma', 'glioma']
        
        print("\nPredictions:")
        for idx, label in enumerate(labels):
            prob = predictions[0][idx]
            marker = " <-- PREDICTED" if idx == np.argmax(predictions[0]) else ""
            print(f"  {label:12s}: {prob*100:5.2f}%{marker}")
        
        predicted_idx = np.argmax(predictions[0])
        predicted_label = labels[predicted_idx]
        confidence = predictions[0][predicted_idx]
        
        print(f"\nFinal: {predicted_label} ({confidence*100:.2f}%)")
        
        # Evaluate if this is a good prediction
        # The image clearly has a tumor, so 'notumor' is wrong
        if predicted_label == 'notumor':
            print("VERDICT: INCORRECT - This is clearly a tumor image!")
            return False
        else:
            print(f"VERDICT: CORRECT - Detected tumor type: {predicted_label}")
            return True
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Test all available models
models_to_test = [
    'best_model.keras',
    'federated_model.keras',
    'federated_models.keras',
    'alzimer_federated_model.keras'
]

print("TESTING ALL AVAILABLE MODELS")
print("="*60)

results = {}
for model_path in models_to_test:
    result = test_model(model_path, 'test_tumor_image.jpg')
    results[model_path] = result

print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)
for model_path, result in results.items():
    if result is None:
        status = "FAILED TO LOAD"
    elif result:
        status = "CORRECT PREDICTION"
    else:
        status = "INCORRECT PREDICTION"
    print(f"{model_path:30s}: {status}")
