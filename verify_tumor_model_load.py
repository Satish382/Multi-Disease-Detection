import os
import tensorflow as tf
import numpy as np

models_to_check = ["best_model.keras", "alzheimer_best_model.keras"]

for model_path in models_to_check:
    print(f"\n--- Verifying {model_path} ---")
    if not os.path.exists(model_path):
        print("File not found.")
        continue
    
    try:
        # Check header
        with open(model_path, 'rb') as f:
            header = f.read(4)
            print(f"Header: {header}")
            
        print("Attempting to load model with compile=False...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        # Check output shape
        output_shape = model.output_shape
        print(f"Output shape: {output_shape}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
