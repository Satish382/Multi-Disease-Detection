import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

def find_last_conv_layer(model):
    """
    Finds the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        # Check if layer output is 4D (batch, height, width, channels)
        if len(layer.output_shape) == 4 and 'conv' in layer.name.lower():
            return layer.name
        # Also check for activation layers that follow conv
        if len(layer.output_shape) == 4 and 'activation' in layer.name.lower():
             return layer.name
        # Also check for specific names like 'block14_sepconv2_act' (Xception)
        if 'block14_sepconv2_act' in layer.name:
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates Grad-CAM heatmap for a given image and model.
    """
    try:
        # Create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    Overlays heatmap on the original image.
    original_img: PIL Image
    heatmap: numpy array (2D)
    """
    try:
        img_array = np.array(original_img)
        
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        
        # We use RGB values of the colormap
        # heatmap is 0-1 float, we skip uint8 scaling for map here to use float indices? 
        # No, colormap takes 0-1 float or 0-255 int.
        jet_colors = jet(heatmap)[:, :3] # Get RGB from RGBA
        
        # jet_colors is (H, W, 3) float 0-1
        jet_heatmap = np.uint8(255 * jet_colors)
        
        # Resize heatmap to match original image size
        jet_heatmap_img = Image.fromarray(jet_heatmap)
        jet_heatmap_img = jet_heatmap_img.resize((img_array.shape[1], img_array.shape[0]), resample=Image.BILINEAR)
        jet_heatmap = np.array(jet_heatmap_img)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img_array
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8') # Ensure valid range

        return Image.fromarray(superimposed_img)
    except Exception as e:
        print(f"Error overlaying heatmap: {e}")
        return original_img
