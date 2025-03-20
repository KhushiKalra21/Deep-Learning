import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import gdown
import os

# Download the trained model if not present
MODEL_PATH = "mnist_cnn_model.h5"
MODEL_DRIVE_LINK = "https://drive.google.com/uc?id=1a6rYW589ZueR9Mq1GCD_LZxuSSCmakXr"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model..."):
        gdown.download(MODEL_DRIVE_LINK, MODEL_PATH, quiet=False)

# Load the trained model
model = load_model(MODEL_PATH)

# Title and description
st.title("🔢 Handwritten & Typed Digit Recognition")
st.write("""
Upload an image of a handwritten or typed digit, and the trained CNN model will predict the number!
""")

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    image = np.array(image.convert("L"))

    # Resize to MNIST size (28x28)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert colors if background is white
    if np.mean(image) > 127:
        image = 255 - image

    # Normalize pixel values (0-1)
    image = image.astype('float32') / 255.0

    # Expand dimensions for model input
    image = np.expand_dims(image, axis=0)  # Batch dimension
    image = np.expand_dims(image, axis=-1)  # Channel dimension (28,28,1)
    
    return image

# File uploader
uploaded_file = st.file_uploader("Upload a digit image (JPG, PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict using the trained model
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Convert to percentage

    # Display prediction results
    st.write(f"## 🔢 Predicted Digit: **{predicted_digit}**")
    st.write(f"**Confidence: {confidence:.2f}%**")

    # Show an alert if confidence is low
    if confidence < 50:
        st.warning("⚠️ The
