import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import gdown
import zipfile
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Google Drive Links
MODEL_URL = "https://drive.google.com/uc?id=1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"
DATASET_ZIP_URL = "https://drive.google.com/uc?id=1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"

# File Paths
MODEL_PATH = "customer_churn_model.h5"
DATASET_ZIP_PATH = "dataset.zip"
EXTRACTED_FOLDER = "extracted_data"

# Download and load pre-trained model
@st.cache_resource
def load_model():
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Download, extract, and sample dataset
@st.cache_data
def load_dataset():
    gdown.download(DATASET_ZIP_URL, DATASET_ZIP_PATH, quiet=False)

    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)

    extracted_files = os.listdir(EXTRACTED_FOLDER)
    csv_file = [file for file in extracted_files if file.endswith(".csv")][0]

    full_data = pd.read_csv(os.path.join(EXTRACTED_FOLDER, csv_file))

    sampled_data = full_data.sample(n=50000, random_state=random.randint(1, 10000))
    return sampled_data

# Sidebar - Hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 1, 50, 10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 0.2])
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Dense Layers", 2, 5, 3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer (2^n)", 5, 10, 7)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.05)

# Main UI
st.title("🚀 Customer Churn Prediction Dashboard")
st.markdown("### 📊 Interactive AI Model Training & Visualization")

# Load dataset preview
st.subheader("📥 Loading Dataset...")
dataset = load_dataset()
st.write(dataset.head())
st.success("✅ Dataset Loaded Successfully!")

# Train the Model Button
if st.button("🚀 Train the Model"):
    st.subheader("🔄 Training the Model...")
    
    # Prepare features & target
    X = dataset.drop(columns=["churned"])
    y = dataset["churned"]

    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build ANN Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(X.shape[1],)))

    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(2**neurons_per_layer, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train Model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    st.success("✅ Model Training Completed!")

    # Plot Training Accuracy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    ax.plot(history.history["val_accuracy"], label="Test Accuracy", marker="s")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("Model Accuracy Over Epochs")
    st.pyplot(fig)

    # Save model & download link
    model.save("trained_model.h5")
    with open("trained_model.h5", "rb") as f:
        st.download_button("📥 Download Trained Model", f, file_name="trained_model.h5")

    # Feature Importance (Correlation Heatmap)
    st.subheader("🔥 Feature Importance Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(fig)

    # Churn Probability Distribution
    st.subheader("📊 Churn Probability Distribution")
    predicted_probs = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(predicted_probs, bins=20, kde=True, color="blue")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Churn Predictions")
    st.pyplot(fig)

st.info("🎯 Select hyperparameters from the sidebar and click 'Train the Model' to begin!")
