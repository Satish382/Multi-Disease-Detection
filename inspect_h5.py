import h5py
import os

model_path = r"c:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzheimer_best_model.keras"

def inspect_h5(path):
    print(f"Inspecting HDF5: {path}")
    try:
        with h5py.File(path, 'r') as f:
            if 'model_weights' in f:
                print("Found 'model_weights' group. Listing layers:")
                # In HDF5 saved models, layer names are usually keys in model_weights
                layers = list(f['model_weights'].keys())
                for l in layers[-20:]: # Print last 20 layers
                    print(l)
            else:
                print("No 'model_weights' group found.")
                print("Keys:", list(f.keys()))
    except Exception as e:
        print(f"Error reading with h5py: {e}")

if __name__ == "__main__":
    if os.path.exists(model_path):
        inspect_h5(model_path)
    else:
        print("File not found.")
