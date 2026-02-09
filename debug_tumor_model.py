import tensorflow as tf
import numpy as np
import os

model_path = "test.h5"

print(f"Loading {model_path}...")
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded.")
    input_shape = model.input_shape
    print(f"Model Input Shape: {input_shape}")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()

