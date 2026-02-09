import os

files = [
    "alzheimer_best_model.keras",
    "alzimer_federated_model.keras"
]

for f_path in files:
    print(f"\n--- Checking {f_path} ---")
    if not os.path.exists(f_path):
        print("File DOES NOT EXIST.")
        continue
    
    size = os.path.getsize(f_path)
    print(f"Size: {size / (1024*1024):.2f} MB")
    
    try:
        with open(f_path, 'rb') as f:
            header = f.read(50)
            print(f"Header (hex): {header.hex()}")
            print(f"Header (text): {header}")
    except Exception as e:
        print(f"Error reading header: {e}")
