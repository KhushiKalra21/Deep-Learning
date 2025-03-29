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
from sklearn.preprocessing import StandardScaler

# Set Streamlit Page Layout
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Google Drive File Links (Use the Correct Ones You Provided)
MODEL_URL = "https://drive.google.com/uc?id=1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"
DATASET_ZIP_URL = "https://drive.google.com/uc?id=1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"

# File Paths
MODEL_PATH = "customer_churn_model.h5"
DATASET_ZIP_PATH = "customer_data.zip"
EXTRACTED_FOLDER = "customer_dataset"

# 🔹 Download & Load Model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# 🔹 Download & Extract Dataset
@st.cache_data
def load_dataset():
    if not os.path.exists(EXTRACTED_FOLDER):
        gdown.download(DATASET_ZIP_URL, DATASET_ZIP_PATH, quiet=False)
        with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_FOLDER)
    
    # Locate CSV File Inside Extracted Folder
    for file in os.listdir(EXTRACTED_FOLDER):
        if file.endswith(".csv"):
            csv_path = os.path.join(EXTRACTED_FOLDER, file)
            break
    
    # Load Dataset
    df = pd.read_csv(csv_path)

    # 🔹 Drop Unwanted Columns & Null Values
    columns_to_remove = ["CustomerID", "Name", "Email"]  # Modify as needed
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
    df.dropna(inplace=True)  # Remove all null rows

    return df

# Load Model & Dataset
model = load_model()
df = load_dataset()

# 🔹 Extract 50,000 Random Data Points Each Time
def get_random_sample():
    return df.sample(50000, replace=False, random_state=random.randint(1, 10000))

# 🔹 Sidebar: Hyperparameter Tuning
st.sidebar.header("Hyperparameter Tuning")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 0.0001])
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32)

# 🔹 Main UI Layout
st.title("📊 Customer Churn Prediction Dashboard")

# 🔹 "Train Model" Button (Outside Sidebar)
if st.button("Train the Model 🚀"):
    # Load Random Sample Data
    df_sample = get_random_sample()

    # Convert Categorical to Numeric
    df_sample = pd.get_dummies(df_sample)

    # Split Data
    X = df_sample.drop(columns=["churned"])
    y = df_sample["churned"]

    # Normalize Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss="binary_crossentropy", metrics=["accuracy"])
    
    history = model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # 🎯 Show Training Progress
    st.success("Model Training Completed! 🎉")

    # 🔹 Plot Training Loss & Accuracy
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Plot
    ax[0].plot(history.history["loss"], label="Loss", color="red")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Accuracy Plot
    ax[1].plot(history.history["accuracy"], label="Accuracy", color="green")
    ax[1].set_title("Training Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    st.pyplot(fig)

    # 🔹 Feature Importance (Correlation Heatmap)
    st.subheader("🔍 Feature Importance - Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_sample.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # 🔹 Churn Rate Insights
    st.subheader("📈 Churn Rate Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="churned", data=df_sample, palette="pastel", ax=ax)
    ax.set_title("Churned vs Retained Customers")
    st.pyplot(fig)

    # 🔹 Age vs Churn Analysis
    if "age" in df_sample.columns:
        st.subheader("📊 Churn by Age Group")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="churned", y="age", data=df_sample, palette="Set2", ax=ax)
        ax.set_title("Age vs Churn Analysis")
        st.pyplot(fig)

    st.balloons()

