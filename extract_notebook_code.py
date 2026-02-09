import json
import os

def extract_code(ipynb_path):
    print(f"\n--- Extracting code from: {os.path.basename(ipynb_path)} ---")
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for cell in data['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'keras.applications' in source or 'Model(' in source or 'Sequential(' in source or 'Conv2D' in source:
                    print("Found relevant code block:")
                    print("-" * 40)
                    print(source[:500]) # Print first 500 chars of relevant cells
                    print("-" * 40)
    except Exception as e:
        print(f"Error reading notebook: {e}")

notebooks = [
    r"c:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\federated-learning-based-transfer-learning-vgg-16 (3).ipynb",
    r"c:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\notebook9956e00078 (2).ipynb"
]

for nb in notebooks:
    if os.path.exists(nb):
        extract_code(nb)
    else:
        print(f"File not found: {nb}")
