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
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this to a secure random key

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

def load_model_safely(model_path):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_models():
    brain_tumor_model_path = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzheimer_best_model.keras"
    alzheimer_model_path = r"C:\Users\91855\Downloads\Brain_alzhimer-20251215T120028Z-1-001\Brain_alzhimer\alzimer_federated_model.keras"


    models = {
        "brain_tumor": load_model_safely(brain_tumor_model_path),
        "alzheimer": load_model_safely(alzheimer_model_path)
    }
    return models

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