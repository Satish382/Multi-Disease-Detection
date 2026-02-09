"""
Verification script to test the implemented image validation system.
This script simulates sending requests to the Flask app's predict endpoint
to verify that invalid images are correctly rejected.
"""
import unittest
import io
import os
import sys
import numpy as np
from PIL import Image
from flask import Flask, session
import json

# Add current directory to path to import app
sys.path.append(os.getcwd())

# Import the app (assuming app.py is in the current directory)
# We need to mock some parts because models might fail to load in this test environment
# without full GPU support, but we primarily want to test the validation logic which 
# runs BEFORE model prediction.
try:
    from app import app, validate_medical_image
except ImportError:
    print("Could not import app. Make sure you are in the correct directory.")
    sys.exit(1)

app.config['TESTING'] = True
app.secret_key = 'test_key'

class TestImageValidation(unittest.TestCase):
    
    def setUp(self):
        self.client = app.test_client()
        # Mock user login
        with self.client.session_transaction() as sess:
            sess['user_email'] = 'test@example.com'
            sess['user_name'] = 'Test User'

    def create_test_image(self, type='random', size=(299, 299)):
        """Helper to create different types of test images"""
        if type == 'random':
            # Random noise
            arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            img = Image.fromarray(arr)
        elif type == 'solid_color':
            # Solid red
            img = Image.new('RGB', size, color='red')
        elif type == 'small':
            # Too small
            img = Image.new('L', (50, 50), color=128)
        elif type == 'grayscale_good':
            # Synthetic "good" looking grayscale image (gradient)
            # This attempts to pass the basic layout checks
            arr = np.zeros((size[1], size[0]), dtype=np.uint8)
            # Create some structure (center circle)
            y, x = np.ogrid[:size[1], :size[0]]
            center = (size[0]//2, size[1]//2)
            mask = ((x - center[0])**2 + (y - center[1])**2) < (100**2)
            arr[mask] = 200
            arr[~mask] = 50
            # Add some noise/texture
            noise = np.random.randint(0, 50, (size[1], size[0]), dtype=np.uint8)
            arr = arr + noise
            img = Image.fromarray(arr)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    def test_validation_function_logic(self):
        """Test the validate_medical_image function directly"""
        print("\nTesting validation function logic...")
        
        # 1. Test Small Image
        img = Image.new('RGB', (50, 50))
        is_valid, msg = validate_medical_image(img)
        print(f"Small Image: Valid={is_valid}, Msg={msg}")
        self.assertFalse(is_valid)
        self.assertIn("too small", msg)

        # 2. Test Solid Color (Red) - Should fail color check
        img = Image.new('RGB', (299, 299), color=(255, 0, 0))
        is_valid, msg = validate_medical_image(img)
        print(f"Solid Red Image: Valid={is_valid}, Msg={msg}")
        self.assertFalse(is_valid)
        self.assertIn("color photo", msg)
        
        # 3. Test Blank/Dark Image
        img = Image.new('L', (299, 299), color=0)
        is_valid, msg = validate_medical_image(img)
        print(f"Black Image: Valid={is_valid}, Msg={msg}")
        self.assertFalse(is_valid)
        self.assertIn("too dark", msg)
        
    def test_predict_endpoint_validation(self):
        """Test the predict endpoint with invalid images"""
        print("\nTesting predict endpoint...")
        
        # Test 1: Upload random noise
        img_bytes = self.create_test_image('random')
        data = {
            'file': (img_bytes, 'random.png'),
            'detection_type': 'brain_tumor'
        }
        resp = self.client.post('/predict', data=data, content_type='multipart/form-data')
        
        print(f"Endpoint Response (Random Noise): {resp.status_code}")
        json_resp = resp.get_json()
        print(f"Response JSON: {json_resp}")
        
        # Should be 400 Bad Request due to validation failure
        # Note: Depending on the randomness, it might fail color check or texture check
        self.assertEqual(resp.status_code, 400)
        self.assertIn('error', json_resp)
        self.assertEqual(json_resp.get('type'), 'validation_error')

if __name__ == '__main__':
    unittest.main()
