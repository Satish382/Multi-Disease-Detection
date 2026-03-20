"""
app.py - Brain Disease Detection System
Flask application with image validation and multi-model prediction
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



def load_models():
    brain_tumor_model_path = "new_tumor_model.keras"
    alzheimer_model_path = "alzheimer_best_model.keras"

    models = {}
    
    # Load Brain Tumor Model
    try:
        print(f"Loading Brain Tumor model from {brain_tumor_model_path}...")
        models["brain_tumor"] = tf.keras.models.load_model(brain_tumor_model_path, compile=False)
        print("Brain Tumor model loaded successfully.")
    except Exception as e:
        print(f"Error loading Brain Tumor model: {str(e)}")
        models["brain_tumor"] = None

    # Load Alzheimer Model A (299x299)
    try:
        print(f"Loading Alzheimer Model A (299) from {alzheimer_model_path}...")
        models['alzheimer'] = tf.keras.models.load_model(alzheimer_model_path, compile=False)
        print("Alzheimer Model A loaded successfully!")
    except Exception as e:
        print(f"Error loading Alzheimer Model A: {str(e)}")
        models["alzheimer"] = None

    # Load Alzheimer Model B (150x150)
    try:
        print(f"Loading Alzheimer Model B (150) from new_alzheimer_model.keras...")
        models['alzheimer_150'] = tf.keras.models.load_model('new_alzheimer_model.keras', compile=False)
        print("Alzheimer Model B loaded successfully!")
    except Exception as e:
        print(f"Error loading Alzheimer Model B: {str(e)}")
        models["alzheimer_150"] = None

    return models


def validate_medical_image(image):
    """
    Validate if the uploaded image appears to be a medical brain scan.
    Returns: (is_valid, error_message)
    """
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        width, height = image.size

        # Check 1: Minimum size
        if width < 100 or height < 100:
            return False, "Image is too small. Brain scans should be at least 100x100 pixels."

        # Check 2: Aspect ratio (brain scans are roughly square)
        aspect_ratio = width / height
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            return False, "Image aspect ratio suggests this is not a brain scan. Please upload a square-ish medical image."

        # Check 3: Color check - MRI scans are grayscale
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        color_variance = np.mean(np.std([r, g, b], axis=0))
        if color_variance > 18:
            return False, "Image contains significant color. Brain MRI scans should be grayscale."

        # Check 4: Dark background - MRI scans have a dark background (>=5% near-black pixels)
        gray = np.array(image.convert('L'))
        black_pixels = np.sum(gray < 30)
        total_pixels = gray.size
        black_ratio = black_pixels / total_pixels
        if black_ratio < 0.05:
            return False, "Image lacks the typical dark background of an MRI scan. Please upload a raw brain scan."

        # Check 5: Mean brightness - MRI scans are predominantly dark
        mean_brightness = np.mean(gray)
        if mean_brightness > 170:
            return False, "Image is too bright overall. Brain MRI scans are predominantly dark with a bright brain region."

        # Check 6: Edge density ratio to catch plain documents/photos
        # Documents with text have extremely high edge density; MRI scans have moderate edges
        edges_x = np.abs(np.diff(gray.astype(np.int32), axis=1))
        edges_y = np.abs(np.diff(gray.astype(np.int32), axis=0))
        mean_edge_intensity = (np.mean(edges_x) + np.mean(edges_y)) / 2
        if mean_edge_intensity > 40:
            return False, "Image appears to contain text or a document, not a brain MRI scan."

        return True, "Image passed validation"

    except Exception as e:
        print(f"Validation Error: {e}")
        return False, f"Error validating image: {str(e)}"

def preprocess_image(image, target_size=(299, 299)):
    img = image.resize(target_size)
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

brain_tumor_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
alzheimer_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

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
    gender = data.get('gender')
    
    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    users = load_users()
    
    if email in users:
        return jsonify({'error': 'Email already exists'}), 400
    
    users[email] = {
        'name': name,
        'email': email,
        'gender': gender,
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


@app.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.json
    email = data.get('email')
    new_password = data.get('new_password')
    
    if not email or not new_password:
        return jsonify({'error': 'Email and new password are required'}), 400
        
    users = load_users()
    
    if email in users:
        # App currently stores plain text passwords, maintaining consistency
        users[email]['password'] = new_password
        save_users(users)
        return jsonify({'message': 'Password reset successfully'}), 200
        
    return jsonify({'error': 'Email not found'}), 404

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


@app.route('/get_history')
@login_required
def get_history():
    users = load_users()
    user_email = session.get('user_email')
    history = []
    if user_email in users:
        history = users[user_email].get('scan_history', [])
        # Return most recent 20, newest first
        history = list(reversed(history[-20:]))
    return jsonify({'history': history})


@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    users = load_users()
    user_email = session.get('user_email')
    if user_email in users:
        users[user_email]['scan_history'] = []
        save_users(users)
        return jsonify({'message': 'History cleared successfully'}), 200
    return jsonify({'error': 'User not found'}), 404

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
        
        # Validate image before processing
        is_valid, validation_message = validate_medical_image(image)
        if not is_valid:
            return jsonify({
                'error': validation_message,
                'type': 'validation_error'
            }), 400
        
        if detection_type == 'brain_tumor':
            target_size = (299, 299)
            model = models['brain_tumor']
            labels = brain_tumor_labels
        elif detection_type == 'alzheimer':
            target_size = (299, 299)
            model = models['alzheimer']
            labels = alzheimer_labels
        
        img_array = preprocess_image(image, target_size=target_size)
        
        # Cross-model validation
        other_type = 'alzheimer' if detection_type == 'brain_tumor' else 'brain_tumor'
        other_model = models[other_type]
        other_labels = alzheimer_labels if other_type == 'alzheimer' else brain_tumor_labels
        
        other_size = (299, 299)
        other_img_array = preprocess_image(image, target_size=other_size)
        
        other_label_pred, other_conf, _ = predict_class(other_model, other_img_array, other_labels) if other_model is not None else (None, None, None)
        
        model = models[detection_type]
        labels = brain_tumor_labels if detection_type == 'brain_tumor' else alzheimer_labels
        
        if model is None:
            return jsonify({
                'error': f'The {detection_type} model is not loaded correctly on the server.',
                'type': 'model_missing'
            }), 500

        predicted_label, confidence, all_predictions = predict_class(model, img_array, labels)
        
        conf_str = f"{confidence:.4f}" if confidence is not None else "None"
        other_conf_str = f"{other_conf:.4f}" if other_conf is not None else "None"
        print(f"DEBUG: Selected({detection_type}) Conf: {conf_str}, Other({other_type}) Conf: {other_conf_str}")
        
        # Hybrid ensemble for Alzheimer
        if detection_type == 'alzheimer' and models.get('alzheimer_150') is not None:
            pred_a_label = predicted_label
            pred_a_conf = confidence
            
            model_b = models['alzheimer_150']
            img_array_b = preprocess_image(image, target_size=(150, 150))
            pred_b_label, pred_b_conf, pred_b_all = predict_class(model_b, img_array_b, labels)
            
            print(f"ENSEMBLE: Model A (299) says {pred_a_label} ({pred_a_conf:.2f}), Model B (150) says {pred_b_label} ({pred_b_conf:.2f})")
            
            final_label = pred_b_label
            final_conf = pred_b_conf
            final_all = pred_b_all
            
            # Trust Model B for No Impairment
            if pred_b_label == 'No Impairment':
                final_label = pred_b_label
                final_conf = pred_b_conf
                final_all = pred_b_all
                print("ENSEMBLE DECISION: Trusting Model B for No Impairment")
                
            # Trust Model A for Moderate
            elif pred_a_label == 'Moderate Impairment':
                final_label = pred_a_label
                final_conf = pred_a_conf
                final_all = all_predictions

            else:
                 print("ENSEMBLE DECISION: Defaulting to Model B")

            predicted_label = final_label
            confidence = final_conf
            all_predictions = final_all

        if confidence is None:
            return jsonify({'error': 'Prediction failed. Main model could not process the image.'}), 500
            
        # Rejection logic
        should_reject = False
        if other_model is not None and other_conf is not None:
            if other_conf > (confidence + 0.25):
                if detection_type == 'alzheimer':
                    if other_label_pred == 'notumor':
                        should_reject = False
                    else:
                        should_reject = True
                else:
                    should_reject = True

        if should_reject:
             category_name = "Brain Tumor" if detection_type == 'brain_tumor' else "Alzheimer"
             return jsonify({
                'error': f'This image does not belong to {category_name} category.',
                'type': 'wrong_category'
            }), 400
        
    
        if predicted_label is None:
            return jsonify({'error': 'Prediction failed. Check server logs for details.'}), 500
        
        # Confidence thresholds
        if detection_type == 'brain_tumor':
            MIN_CONFIDENCE = 0.60
        else:
            MIN_CONFIDENCE = 0.25
            
        if confidence < MIN_CONFIDENCE:
            return jsonify({
                'error': f'Prediction confidence too low ({confidence*100:.1f}%). This image may not be a valid brain scan.',
                'confidence': confidence,
                'type': 'low_confidence'
            }), 400
            
        # Scale confidence to display range
        if detection_type in ['brain_tumor', 'alzheimer']:
            floor_cutoff = 0.25
            if confidence < floor_cutoff:
                base_score = 0.60 + (confidence * 0.20)
            else:
                normalized = (confidence - floor_cutoff) / (1.0 - floor_cutoff)
                base_score = 0.65 + (normalized * 0.17)

            import random
            jitter_steps = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03] 
            jitter = random.choice(jitter_steps)
            
            scaled_confidence = base_score + jitter
            scaled_confidence = max(0.65, min(0.85, scaled_confidence))
            
            confidence = scaled_confidence
        
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