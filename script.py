import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from matplotlib import pyplot as plt

# Define the image dimensions
IMG_WIDTH = 448
IMG_HEIGHT = 448

# Load the trained model
model_dir = 'modelcp'
checkpoint_path = model_dir + '/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

def create_model(summary=True):
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Add global average pooling layer
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Add L2 regularization
    x = Dropout(0.5)(x)  # Add Dropout layer
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Optional additional Dense layer with L2
    x = Dropout(0.5)(x)  # Add Dropout layer
    predictions = Dense(4, activation='softmax')(x)  # Two output neurons for "defect" and "no defect"

    # Combine the base model and the custom layers into a final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Keep all layers trainable
    for layer in base_model.layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-5)  # Reduce the learning rate
    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if summary:
        print(model.summary())
    return model

latest = tf.train.latest_checkpoint(checkpoint_dir)
loaded_model = create_model(summary=True)
loaded_model.load_weights(latest)
loaded_model.summary()

# Define a function to display images
def show_img(img_array, title):
    plt.title(title)
    plt.imshow(img_array)
    plt.axis("off")
    plt.show()

# Function to create Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    model.layers[-1].activation = None
    grad_model = tf.keras.models.Model([model.inputs], 
                                       [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / max_val.numpy()

    return heatmap.numpy()

# Function to superimpose heatmap onto the image and save it
def save_and_display_gradcam(img_path, heatmap, save_path, alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Colorize the heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create heatmap image
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    # Save the image with the heatmap
    superimposed_img.save(save_path)

# Prediction and saving defect images with Grad-CAM visualization
def run_gradcam_on_images():
    test_images_path = 'test_run/images'
    defects_save_path = 'test_run/defects'
    
    if not os.path.exists(defects_save_path):
        os.makedirs(defects_save_path)
    
    # Iterate through images in the test folder
    for img_name in os.listdir(test_images_path):
        img_path = os.path.join(test_images_path, img_name)
        
        # Load and preprocess the image
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_rescaled = img_resized / 255.0
        img_array = np.expand_dims(img_rescaled, axis=0)
        
        # Predict using the model
        pred = np.argmax(loaded_model.predict(img_array)[0])
        
        # If it's a defect (replace this with your actual label index for defects)
        if pred in [1, 2]:  # Assuming 1, 2 are defect labels
            print(f'Defect detected in: {img_name}')
            
            # Create Grad-CAM heatmap
            last_conv_layer_name = 'conv5_block3_out'  # Adjust this layer name as needed
            heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name)
            
            # Save the defect image with Grad-CAM visualization
            save_path = os.path.join(defects_save_path, f'defect_{img_name}')
            save_and_display_gradcam(img_path, heatmap, save_path)

# Run the function
run_gradcam_on_images()
