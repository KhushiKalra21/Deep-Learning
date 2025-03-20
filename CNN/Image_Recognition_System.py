import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os
from PIL import Image

# Define Google Drive file ID and destination path
file_id = "1a6rYW589ZueR9Mq1GCD_LZxuSSCmakXr"
model_path = "mnist_cnn_model.keras"

# Download model if not already downloaded
if not os.path.exists(model_path):
    with st.spinner("Downloading model... Please wait."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image.convert('L'))  # Convert to grayscale
    image = cv2.resize(image, (28, 28))   # Resize to 28x28
    image = image.astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Reshape for model
    return image

# Streamlit UI
st.title("📝 Handwritten & Typed Digit Recognition")
st.write("Upload an image of a digit (handwritten or typed) to predict the number.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Display result
    st.subheader(f"Predicted Digit: {predicted_digit}")
    st.write("### Confidence Scores:")
    for i, score in enumerate(prediction[0]):
        st.write(f"Digit {i}: {score:.4f}")
