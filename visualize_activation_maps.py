import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import cv2
import os

# Load the pre-trained model
model_path = 'model/my_model.h5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Load and preprocess the input image
def load_and_preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to match model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Visualize activation maps from a specific convolutional layer
def visualize_activation_maps(model, img_path, layer_name, num_filters_to_show=8):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path)
    
    # Display the original image
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    # Ensure the layer exists in the model
    if layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Layer {layer_name} not found in the model.")
    
    # Get the output of the specified layer
    layer_output = model.get_layer(layer_name).output

    # Ensure the model has been built by running a dummy input through it
    _ = model.predict(np.zeros((1, 224, 224, 3)))

    # Create a new model that outputs the activations of the selected layer
    activation_model = Model(inputs=model.input, outputs=layer_output)

    # Get the activations for the input image
    activations = activation_model.predict(img_array)

    # Visualize activations for the first few filters
    fig, ax = plt.subplots(1, num_filters_to_show, figsize=(20, 20))
    for i in range(num_filters_to_show):
        activation = activations[0, :, :, i]
        ax[i].imshow(activation, cmap='viridis')
        ax[i].axis('off')

    plt.suptitle(f'Activation Maps from Layer: {layer_name}')
    plt.show()

if __name__ == "__main__":
    # Path to the image
    img_path = 'images/test_image.jpg'

    # Layer to visualize activations from (replace with your chosen layer)
    layer_name = 'conv2d_1'  # Modify according to your model's layer names

    # Visualize activation maps
    visualize_activation_maps(model, img_path, layer_name, num_filters_to_show=8)
