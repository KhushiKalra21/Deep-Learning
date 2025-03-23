import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import gdown
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# ============================
# 📌 GOOGLE DRIVE FILE LINKS
# ============================
DATASET_URL = "https://drive.google.com/uc?id=1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_URL = "https://drive.google.com/uc?id=1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"
DATASET_PATH = "customer_dataset.zip"
MODEL_PATH = "customer_churn_model.h5"

# ============================
# 🔽 DOWNLOAD FILES IF NOT EXIST
# ============================

# Download Dataset
if not os.path.exists(DATASET_PATH):
    st.info("Downloading dataset...")
    gdown.download(DATASET_URL, DATASET_PATH, quiet=False)

# Extract Dataset
if os.path.exists(DATASET_PATH):
    with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
        zip_ref.extractall("data")  # Extract inside 'data' folder
    st.success("Dataset extracted successfully!")

# Download Trained Model
if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load Model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("Trained model loaded successfully!")

# ============================
# 📊 LOAD DATA & PICK 50,000 RANDOM RECORDS
# ============================

DATA_FILE = "data/customer_data.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)

    # 🟢 Random 50,000 records, har baar alag
    df = df.sample(n=50000, random_state=np.random.randint(1, 10000))

    st.write("### 🔍 Preview of 50,000 Sampled Customers", df.head())

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # Splitting Data
    X = df.drop(columns=["churned"])
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# ============================
# 🎛 STREAMLIT SIDEBAR CONTROLS
# ============================

st.sidebar.title("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=100, step=10, value=50)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"], index=0)
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd"], index=0)
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4], index=1)
neurons = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256], index=1)

# ============================
# 🏋️ TRAIN MODEL BUTTON (OUTSIDE SIDEBAR)
# ============================

if st.button("🚀 Train the Model"):
    with st.spinner("Training the model..."):
        # Define New Model Architecture
        new_model = tf.keras.models.Sequential()
        new_model.add(tf.keras.layers.Dense(neurons, activation=activation, input_shape=(X_train.shape[1],)))
        for _ in range(dense_layers - 1):
            new_model.add(tf.keras.layers.Dense(neurons, activation=activation))
        new_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer

        # Compile Model
        new_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        # Train Model
        history = new_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

        # Save the Model
        new_model.save("new_trained_model.h5")
        st.success("Model training complete! ✅")

        # ============================
        # 📈 VISUALIZATIONS
        # ============================

        st.write("## 📊 Training Progress")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(history.history["accuracy"], label="Train Accuracy")
        ax[0].plot(history.history["val_accuracy"], label="Test Accuracy")
        ax[0].set_title("📈 Accuracy over Epochs")
        ax[0].legend()

        ax[1].plot(history.history["loss"], label="Train Loss")
        ax[1].plot(history.history["val_loss"], label="Test Loss")
        ax[1].set_title("📉 Loss over Epochs")
        ax[1].legend()

        st.pyplot(fig)

# ============================
# 📊 DASHBOARD VISUALS
# ============================

st.write("## 🔍 Data Insights")

# Age vs Churned Customers
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df[df['churned'] == 1]['age'], bins=30, kde=True, color="red", label="Churned")
sns.histplot(df[df['churned'] == 0]['age'], bins=30, kde=True, color="blue", label="Retained")
ax.set_title("Age Distribution of Churned vs Retained Customers")
plt.legend()
st.pyplot(fig)

# Churn Rate by Loyalty Program
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=df["loyalty_program"], y=df["churned"], ci=None, palette="coolwarm")
ax.set_title("📊 Churn Rate by Loyalty Program")
st.pyplot(fig)

st.success("Dashboard loaded successfully! 🚀")
