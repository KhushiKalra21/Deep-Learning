import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
from PIL import Image
import os

# Google Drive model file ID
file_id = "1a6rYW589ZueR9Mq1GCD_LZxuSSCmakXr"
model_path = "mnist_cnn_model.h5"

# Download the model if not already downloaded
if not os.path.exists(model_path):
    with st.spinner("Downloading trained model... Please wait."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to MNIST size
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (28,28,1)
    return image

# Streamlit UI
st.title("📌 Handwritten & Typed Digit Recognition")
st.write("Upload a handwritten or typed digit image to classify it using a CNN model trained on MNIST.")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert file to PIL Image
    image = Image.open(uploaded_file)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Convert confidence to percentage

    # Display result
    st.write(f"### 🔢 Predicted Digit: {predicted_digit}")
    st.write(f"**Confidence: {confidence:.2f}%**")
