import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import os
from tkinter import Tk, Button, Label, filedialog, messagebox

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def visualize_activation_maps(model, img_path, layer_name, num_filters_to_show=8):
    img_array = load_and_preprocess_image(img_path)
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.axis('off')
    plt.show()
    if layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Layer {layer_name} not found in the model.")
    layer_output = model.get_layer(layer_name).output
    activation_model = Model(inputs=model.inputs, outputs=layer_output)
    activations = activation_model.predict(img_array)
    fig, ax = plt.subplots(1, num_filters_to_show, figsize=(20, 20))
    for i in range(num_filters_to_show):
        activation = activations[0, :, :, i]
        ax[i].imshow(activation, cmap='viridis')
        ax[i].axis('off')
    plt.suptitle(f'Activation Maps from Layer: {layer_name}')
    plt.show()

def select_image_for_activation_map(model, layer_name):
    img_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if img_path:
        visualize_activation_maps(model, img_path, layer_name, num_filters_to_show=8)
    else:
        messagebox.showwarning("No Image", "No image was selected!")

def create_gui(model, layer_name):
    root = Tk()
    root.title("Image Upload for Activation Maps")
    root.geometry("400x200")
    label = Label(root, text="Click the button to upload an image.")
    label.pack(pady=20)
    upload_button = Button(root, text="Select Image", command=lambda: select_image_for_activation_map(model, layer_name))
    upload_button.pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    model_path = r'C:\Users\ankit\OneDrive\Desktop\Data Science\first task\emotion_detection_activation_maps\model\emotion_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    layer_name = 'conv2d_1'
    create_gui(model, layer_name)
