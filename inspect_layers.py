import tensorflow as tf
import os

# Model paths
BRAIN_TUMOR_MODEL = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzheimer_best_model.keras"
ALZHEIMER_MODEL = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzimer_federated_model.keras"

def inspect_model(model_path, name):
    print(f"\n{'='*20} Inspecting {name} {'='*20}")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        
        # Print summary (abbreviated)
        # model.summary()
        
        print("\nLast 10 Layers:")
        for i, layer in enumerate(model.layers[-15:]):
            print(f"{i}. Name: {layer.name}, Type: {layer.__class__.__name__}")
            
            # Check if it's a conv layer
            if 'conv' in layer.name or 'pool' in layer.name:
                output_shape = layer.output.shape
                print(f"   Shape: {output_shape}")

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model(BRAIN_TUMOR_MODEL, "Brain Tumor Model")
    inspect_model(ALZHEIMER_MODEL, "Alzheimer Model")
