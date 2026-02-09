"""
IMPROVED app.py with Image Validation
This version adds validation to reject non-medical images
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import io
import base64
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

USER_DATA_FILE = 'user.json'

# Load users from JSON file
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

# Save users to JSON file
def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

import tf_keras

def load_models():
    # FIXED: Using federated_models.keras instead of best_model.keras
    # The best_model.keras file is broken and incorrectly predicts "no tumor" for actual tumors
    # federated_models.keras correctly identifies tumor types
    brain_tumor_model_path = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\federated_models.keras"
    # Corrected path for Alzheimer as well (was using federated model which failed)
    alzheimer_model_path = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzheimer_best_model.keras"

    models = {}
    
    # Load Brain Tumor Model using tf.keras (Keras 3 Zip format)
    # federated_models.keras is in the modern Keras 3 format
    try:
        print(f"Loading Brain Tumor model from {brain_tumor_model_path}...")
        models["brain_tumor"] = tf.keras.models.load_model(brain_tumor_model_path, compile=False)
        print("Brain Tumor model loaded successfully.")
    except Exception as e:
        print(f"Error loading Brain Tumor model: {str(e)}")
        models["brain_tumor"] = None

    # Load Alzheimer Model using tf.keras (Standard Keras 3 Zip support)
    try:
        print(f"Loading Alzheimer model from {alzheimer_model_path}...")
        models["alzheimer"] = tf.keras.models.load_model(alzheimer_model_path, compile=False)
        print("Alzheimer model loaded successfully.")
    except Exception as e:
        print(f"Error loading Alzheimer model: {str(e)}")
        models["alzheimer"] = None

    return models

# ✅ NEW: Image validation function
def validate_medical_image(image):
    """
    Validate if the uploaded image appears to be a medical brain scan
    Returns: (is_valid, error_message)
    """
    try:
        img_array = np.array(image)
        width, height = image.size
        
        # Check 1: Minimum size requirement
        if width < 100 or height < 100:
            return False, "Image is too small. Brain scans should be at least 100x100 pixels."
        
        # Check 2: Aspect ratio (brain scans are usually square-ish)
        aspect_ratio = width / height
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            return False, "Image aspect ratio suggests this is not a brain scan. Please upload a square-ish medical image."
        
        # Check 3: Color check (brain scans are typically grayscale or low color variance)
        if image.mode == 'RGB':
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            # Calculate color variance
            color_diff = np.mean(np.abs(r.astype(float) - g.astype(float))) + \
                        np.mean(np.abs(g.astype(float) - b.astype(float)))
            
            if color_diff > 15:  # Too colorful
                return False, "Image appears to be a color photo. Brain scans are typically grayscale. Please upload a medical scan."
        
        # Check 4: Brightness distribution
        gray = np.array(image.convert('L'))
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 15:
            return False, "Image is too dark. Please upload a properly exposed brain scan."
        
        if mean_brightness > 240:
            return False, "Image is too bright or blank. Please upload a valid brain scan."
        
        # Check 5: Contrast (medical images should have good contrast)
        contrast = np.std(gray)
        if contrast < 25:
            return False, "Image has very low contrast. Brain scans should show clear tissue differentiation."
        
        # Check 6: Edge detection (medical scans have distinct structures)
        # Simple edge detection using gradient
        edges_x = np.abs(np.diff(gray, axis=1))
        edges_y = np.abs(np.diff(gray, axis=0))
        edge_density = (np.mean(edges_x) + np.mean(edges_y)) / 2
        
        if edge_density < 5:
            return False, "Image lacks structural features typical of brain scans."
        
        return True, "Image passed validation"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def preprocess_image(image):
    img = image.resize((299, 299))
    img_array = np.array(img) / 255.0
    
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def predict_class(model, image, labels):
    try:
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        all_predictions = []
        for idx, label in enumerate(labels):
            all_predictions.append({
                'label': label,
                'confidence': float(predictions[0][idx])
            })
        
        return labels[predicted_class], float(confidence), all_predictions
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None, None

models = load_models()

brain_tumor_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']
alzheimer_labels = ['No Impairment', 'Very Mild Impairment', 'Moderate Impairment', 'Mild Impairment']

@app.route('/')
def index():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    users = load_users()
    
    if email in users:
        return jsonify({'error': 'Email already exists'}), 400
    
    users[email] = {
        'name': name,
        'password': password,  # In production, use proper password hashing
        'created_at': datetime.now().isoformat(),
        'scan_history': []
    }
    
    save_users(users)
    
    return jsonify({'message': 'Account created successfully'}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    users = load_users()
    
    if email not in users:
        return jsonify({'error': 'Invalid email or password'}), 401
    
    if users[email]['password'] != password:
        return jsonify({'error': 'Invalid email or password'}), 401
    
    session['user_email'] = email
    session['user_name'] = users[email]['name']
    
    return jsonify({'message': 'Login successful'}), 200

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/get_user')
@login_required
def get_user():
    return jsonify({
        'name': session.get('user_name'),
        'email': session.get('user_email')
    })

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    detection_type = request.form.get('detection_type', 'brain_tumor')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image = Image.open(file.stream)
        
        # ✅ NEW: Validate image before processing
        is_valid, validation_message = validate_medical_image(image)
        if not is_valid:
            return jsonify({
                'error': validation_message,
                'type': 'validation_error'
            }), 400
        
        img_array = preprocess_image(image)
        
        if detection_type == 'brain_tumor':
            model = models['brain_tumor']
            labels = brain_tumor_labels
        else:
            model = models['alzheimer']
            labels = alzheimer_labels
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        predicted_label, confidence, all_predictions = predict_class(model, img_array, labels)
        
        if predicted_label is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # ✅ NEW: Confidence threshold check
        MIN_CONFIDENCE = 0.40  # 40% minimum confidence
        if confidence < MIN_CONFIDENCE:
            return jsonify({
                'error': f'Prediction confidence too low ({confidence*100:.1f}%). This image may not be a valid brain scan.',
                'confidence': confidence,
                'type': 'low_confidence'
            }), 400
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save scan to user history
        users = load_users()
        user_email = session.get('user_email')
        if user_email in users:
            scan_entry = {
                'timestamp': datetime.now().isoformat(),
                'detection_type': detection_type,
                'predicted_label': predicted_label,
                'confidence': confidence
            }
            if 'scan_history' not in users[user_email]:
                users[user_email]['scan_history'] = []
            users[user_email]['scan_history'].append(scan_entry)
            save_users(users)
        
        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)