import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
import cv2
from PIL import Image

# ---------------------- MODEL DOWNLOAD & LOADING ----------------------

MODEL_URL = "https://drive.google.com/uc?id=1a6rYW589ZueR9Mq1GCD_LZxuSSCmakXr"
MODEL_PATH = "mnist_cnn_model.h5"

# Download model if not found
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading pre-trained model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------------------- IMAGE PREPROCESSING ----------------------

def preprocess_image(image):
    """ Convert image to grayscale, threshold, resize, and normalize for MNIST format. """
    image = np.array(image.convert('L'))  # Convert to grayscale
    image = cv2.resize(image, (28, 28))   # Resize to 28x28 (MNIST format)

    # Adaptive thresholding for better contrast (MNIST has white digits on black background)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Normalize pixel values (0-1)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1) # Add channel dimension
    return image

# ---------------------- STREAMLIT UI ----------------------

st.title("🔢 Handwritten & Typed Digit Recognition")
st.write("Upload an image of a handwritten or typed digit, and the model will predict the number.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Perform prediction
    if model:
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display result
        st.success(f"🧠 Model Prediction: **{predicted_digit}**")
        st.info(f"📊 Confidence Level: **{confidence:.2f}%**")
    else:
        st.error("Model not loaded properly. Please check the model file.")
