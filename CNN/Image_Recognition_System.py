import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load Trained CNN Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()

# Image Preprocessing Function
def preprocess_image(image):
    """Converts uploaded image into a format suitable for the CNN model."""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 (same as MNIST)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (28, 28, 1)
    return image

# Streamlit UI
st.title("🖊️ Handwritten & Typed Digit Recognition")
st.write("Upload an image of a handwritten or typed digit, and the model will predict it.")

# Upload Image
uploaded_file = st.file_uploader("Upload a digit image (JPG, PNG)", type=["jpg", "png"])

if uploaded_file:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    processed_image = preprocess_image(image)

    # Predict with Model
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)

    # Display Prediction
    st.subheader(f"🔢 Predicted Digit: {predicted_digit}")

    # Show Confidence Scores
    st.write("**Confidence Scores:**")
    confidence_scores = {f"Digit {i}": round(float(prediction[0][i]) * 100, 2) for i in range(10)}
    st.json(confidence_scores)
