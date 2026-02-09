"""
Quick test to show which model works correctly
"""
import tf_keras
import numpy as np
from PIL import Image

def quick_test(model_path):
    try:
        model = tf_keras.models.load_model(model_path, compile=False)
    except:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
    
    image = Image.open('test_tumor_image.jpg')
    img = image.resize((299, 299))
    img_array = np.array(img) / 255.0
    
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    labels = ['pituitary', 'notumor', 'meningioma', 'glioma']
    
    print(f"\n{model_path}:")
    for idx, label in enumerate(labels):
        prob = predictions[0][idx]
        marker = " <--" if idx == np.argmax(predictions[0]) else ""
        print(f"  {label:12s}: {prob*100:5.2f}%{marker}")

print("COMPARISON OF MODELS:")
print("="*50)
quick_test('best_model.keras')
quick_test('federated_models.keras')
quick_test('alzimer_federated_model.keras')
