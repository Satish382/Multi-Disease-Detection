"""
Test if both models load correctly with the current configuration
"""
import tensorflow as tf
import tf_keras

print("Testing model loading configuration...")
print("="*60)

# Test Brain Tumor Model (federated_models.keras)
print("\n1. Testing Brain Tumor Model (federated_models.keras)...")
brain_tumor_path = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\federated_models.keras"

try:
    print(f"   Loading with tf.keras...")
    model = tf.keras.models.load_model(brain_tumor_path, compile=False)
    print(f"   ✓ SUCCESS with tf.keras")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"   ✗ FAILED with tf.keras: {str(e)}")

# Test Alzheimer Model (alzheimer_best_model.keras)
print("\n2. Testing Alzheimer Model (alzheimer_best_model.keras)...")
alzheimer_path = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzheimer_best_model.keras"

try:
    print(f"   Loading with tf.keras...")
    model = tf.keras.models.load_model(alzheimer_path, compile=False)
    print(f"   ✓ SUCCESS with tf.keras")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"   ✗ FAILED with tf.keras: {str(e)}")

print("\n" + "="*60)
print("CONCLUSION: Both models should load with tf.keras")
