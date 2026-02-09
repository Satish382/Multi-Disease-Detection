// Global variables
let currentDetectionType = 'brain_tumor';
let selectedFile = null;

// Page titles for different detection types
const pageTitles = {
    brain: {
        title: 'Brain Tumor Detection',
        icon: 'fa-brain'
    },
    alzheimer: {
        title: 'Alzheimer Detection',
        icon: 'fa-head-side-brain'
    }
};

// DOM elements
const navButtons = document.querySelectorAll('.nav-btn');
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const removeBtn = document.getElementById('remove-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const uploadForm = document.getElementById('upload-form');
const loading = document.getElementById('loading');
const emptyState = document.getElementById('empty-state');
const resultsContent = document.getElementById('results-content');
const pageTitle = document.getElementById('page-title');

// Navigation button click handler
navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all buttons
        navButtons.forEach(b => b.classList.remove('active'));
        
        // Add active class to clicked button
        btn.classList.add('active');
        
        // Get page type
        const page = btn.getAttribute('data-page');
        
        // Update detection type
        currentDetectionType = page === 'brain' ? 'brain_tumor' : 'alzheimer';
        
        // Update page title
        const titleData = pageTitles[page];
        pageTitle.innerHTML = `<i class="fas ${titleData.icon}"></i> ${titleData.title}`;
        
        // Reset form
        resetForm();
    });
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    }
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Handle file selection
function handleFileSelect(file) {
    if (!file) return;
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Remove button handler
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetForm();
});

// Form submit handler
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedFile) return;
    
    // Hide results and show loading
    resultsContent.style.display = 'none';
    emptyState.style.display = 'none';
    loading.style.display = 'block';
    analyzeBtn.disabled = true;
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('detection_type', currentDetectionType);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            // Show specific error message from validation
            let errorMsg = data.error || 'Prediction failed';
            
            // Add helpful styling for validation errors
            if (data.type === 'validation_error') {
                alert('⚠️ Image Validation Failed\n\n' + errorMsg + '\n\nPlease upload a brain MRI or CT scan image.');
            } else if (data.type === 'low_confidence') {
                alert('⚠️ Low Confidence Prediction\n\n' + errorMsg + '\n\nThe image may not be a valid brain scan.');
            } else {
                alert('Error: ' + errorMsg);
            }
            
            loading.style.display = 'none';
            emptyState.style.display = 'block';
            analyzeBtn.disabled = false;
            return;
        }
        
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
        loading.style.display = 'none';
        emptyState.style.display = 'block';
        analyzeBtn.disabled = false;
    }
});


// Display results
function displayResults(data) {
    loading.style.display = 'none';
    resultsContent.style.display = 'block';
    
    // Update predicted condition
    document.getElementById('predicted-condition').textContent = data.predicted_label;
    
    // Update confidence
    const confidencePercent = (data.confidence * 100).toFixed(1);
    document.getElementById('confidence-value').textContent = `${confidencePercent}%`;
    
    // Update confidence bar
    const confidenceFill = document.getElementById('confidence-fill');
    setTimeout(() => {
        confidenceFill.style.width = `${confidencePercent}%`;
    }, 100);
    
    // Update predictions list
    const predictionsList = document.getElementById('predictions-list');
    predictionsList.innerHTML = '';
    
    // Sort predictions by confidence
    const sortedPredictions = [...data.all_predictions].sort((a, b) => b.confidence - a.confidence);
    
    sortedPredictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const label = document.createElement('span');
        label.className = 'prediction-label';
        label.textContent = pred.label;
        
        const confidence = document.createElement('span');
        confidence.className = 'prediction-confidence';
        confidence.textContent = `${(pred.confidence * 100).toFixed(1)}%`;
        
        item.appendChild(label);
        item.appendChild(confidence);
        predictionsList.appendChild(item);
    });
    
    // Enable analyze button for new analysis
    analyzeBtn.disabled = false;
    
    // Update uploaded image
    const uploadedImage = document.getElementById('uploadedImage');
    if (data.image) {
        uploadedImage.src = "data:image/png;base64," + data.image;
    }

    // Enable analyze button for new analysis
    analyzeBtn.disabled = false;
}

// Reset form
function resetForm() {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsContent.style.display = 'none';
    emptyState.style.display = 'block';
    loading.style.display = 'none';
}

// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Add animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.card, .info-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
});