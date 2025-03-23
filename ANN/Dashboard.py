import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import gdown
import zipfile
import os
import random
from io import BytesIO

# Google Drive links
MODEL_URL = "https://drive.google.com/uc?id=1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"
DATASET_ZIP_URL = "https://drive.google.com/uc?id=1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"

# File paths
MODEL_PATH = "customer_churn_model.h5"
DATASET_ZIP_PATH = "dataset.zip"
EXTRACTED_FOLDER = "extracted_data"

# Download and load model
@st.cache_resource
def load_model():
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Download, extract, and sample dataset
@st.cache_data
def load_dataset():
    gdown.download(DATASET_ZIP_URL, DATASET_ZIP_PATH, quiet=False)
    
    # Extract ZIP
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)
    
    # Find CSV file inside extracted folder
    extracted_files = os.listdir(EXTRACTED_FOLDER)
    csv_file = [file for file in extracted_files if file.endswith(".csv")][0]
    
    # Load CSV
    full_data = pd.read_csv(os.path.join(EXTRACTED_FOLDER, csv_file))
    
    # Sample 50,000 random records
    sampled_data = full_data.sample(n=50000, random_state=random.randint(1, 10000))
    
    return sampled_data

# Streamlit UI
st.title("🔍 Customer Churn Prediction - ANN Dashboard")

# Sidebar - Hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 0.2])
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Dense Layers", min_value=2, max_value=5, value=3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer (2^n)", min_value=5, max_value=10, value=7)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, step=0.05)

# Load model and dataset
st.subheader("📥 Loading Model & Dataset...")
model = load_model()
dataset = load_dataset()
st.write(dataset.head())

st.success("✅ Model and Dataset Loaded Successfully!")
