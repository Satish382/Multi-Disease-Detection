def check_file_header(filename):
    try:
        with open(filename, 'rb') as f:
            header = f.read(8)
        
        print(f"\nChecking {filename}...")
        print(f"First 8 bytes: {header}")
        
        if header.startswith(b'\x89HDF'):
            print("✅ Format: HDF5 (Old Keras format).")
        elif header.startswith(b'PK'):
            print("✅ Format: ZIP (New Keras v3 format).")
        elif b'html' in header or b'<!DOC' in header:
            print("❌ Format: HTML (Corrupted download). You downloaded the webpage, not the file.")
        else:
            print("❌ Format: Unknown/Corrupted.")
            
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")

# Check your specific files
check_file_header('federated_model.keras')
check_file_header('alzimer_federated_model.keras')