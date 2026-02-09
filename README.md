# Multi-Disease Detection System

A Flask-based web application that uses deep learning models for medical image analysis, specifically for Brain Tumor and Alzheimer's Disease detection.

## 🧠 Features

- **Dual AI Models**
  - Brain Tumor Detection (4 classes: Pituitary, No Tumor, Meningioma, Glioma)
  - Alzheimer's Disease Detection (4 severity levels: No Impairment, Very Mild, Moderate, Mild)

- **Robust Image Validation**
  - Size and aspect ratio checks
  - Color variance analysis
  - Brightness and contrast validation
  - Edge detection for structural features
  - Confidence threshold filtering (40% minimum)

- **User Management**
  - Secure authentication system
  - Session-based access control
  - Scan history tracking

## 🚀 Technology Stack

- **Backend**: Python 3.x, Flask 3.1.0
- **AI/ML**: TensorFlow 2.17.1, Keras 3.5.0, tf_keras 2.18.0
- **Image Processing**: Pillow 10.4.0, OpenCV 4.10.0
- **Frontend**: HTML5, CSS3, JavaScript

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Satish382/Multi-Disease-Detection.git
cd Multi-Disease-Detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure model files are present:
   - `federated_models.keras` (Brain Tumor Model)
   - `alzheimer_best_model.keras` (Alzheimer Model)

## 🎯 Usage

1. Run the application:
```bash
python app.py
```
Or use the batch file on Windows:
```bash
run.bat
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Create an account or login

4. Upload brain scan images for analysis

## 📁 Project Structure

```
Brain_alzhimer/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── run.bat                         # Windows startup script
├── user.json                       # User database
├── templates/
│   ├── login.html                  # Login/Signup page
│   ├── dashboard.html              # Main dashboard
│   └── index.html                  # Landing page
├── static/
│   ├── style.css                   # Styling
│   ├── script.js                   # Client-side logic
│   └── brain_hero.png              # UI assets
└── models/
    ├── federated_models.keras      # Brain Tumor Model
    └── alzheimer_best_model.keras  # Alzheimer Model
```

## 🔒 Security Notes

- This is a demonstration application
- In production, implement proper password hashing (bcrypt, argon2)
- Use environment variables for secret keys
- Implement HTTPS
- Add rate limiting and CSRF protection

## 📊 Model Information

- **Architecture**: VGG-16 Transfer Learning
- **Input Size**: 299x299x3
- **Model Format**: Keras 3 (.keras files)
- **Model Size**: ~87 MB each
- **Minimum Confidence**: 40%

## 🎓 Use Cases

- Medical research and education
- Preliminary screening tool
- Healthcare AI demonstrations
- Deep learning portfolio projects

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Satish382

## 🙏 Acknowledgments

- VGG-16 architecture by Visual Geometry Group, Oxford
- TensorFlow and Keras teams
- Medical imaging datasets contributors
