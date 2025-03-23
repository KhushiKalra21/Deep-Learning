import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import zipfile
import requests
from io import BytesIO

# Google Drive File IDs
MODEL_FILE_ID = "1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"
ZIP_FILE_ID = "1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"

# Function to download files from Google Drive
@st.cache_resource
def download_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    return BytesIO(response.content)

# Load Model from Google Drive
@st.cache_resource
def load_model():
    model_file = download_from_drive(MODEL_FILE_ID)
    model = tf.keras.models.load_model(model_file)
    return model

# Extract 50,000 Random Data Points from ZIP
@st.cache_data
def extract_random_data():
    zip_file = download_from_drive(ZIP_FILE_ID)

    with zipfile.ZipFile(zip_file, 'r') as z:
        file_names = z.namelist()
        csv_file_name = file_names[0]  # Assuming first file is the dataset
        with z.open(csv_file_name) as f:
            df = pd.read_csv(f)

    return df.sample(n=50000, random_state=random.randint(1, 1000))  # Random state changes every time

# Load Model and Data
model = load_model()
df = extract_random_data()

# Sidebar - Hyperparameter Slicers
st.sidebar.title("🔧 Hyperparameter Tuning")

epochs = st.sidebar.slider("Epochs", 1, 100, 10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 0.5])
activation_fn = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
backprop_method = st.sidebar.radio("Backpropagation Method", ["Batch Gradient", "Stochastic Gradient", "Mini-Batch Gradient"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Dense Layers", 2, 5, 3)
neurons_per_layer = st.sidebar.selectbox("Neurons per Layer", [2**n for n in range(5, 11)])
dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)

# Prediction Button
if st.sidebar.button("Run Prediction"):
    X = df.drop(columns=["churned"])  # Assuming 'churned' is the target variable
    y = df["churned"]

    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)

    df["Churn Probability"] = y_pred_prob
    df["Predicted Churn"] = y_pred

    st.write("### 📊 Prediction Results on 50,000 Customers")
    st.write(df.head(20))

    st.write("### 📈 Churn Distribution")
    st.bar_chart(df["Predicted Churn"].value_counts())

    st.write("### 🔍 Retention Strategy")
    st.write(
        "👥 **High Churn Risk Customers:**\n"
        "- Offer loyalty discounts.\n"
        "- Provide personalized email campaigns.\n"
        "- Improve customer service interactions.\n"
        "- Introduce exclusive benefits for long-term retention."
    )

st.write("📌 **Note:** Every time you run the dashboard, a new random 50,000 customers are selected from the dataset.")

