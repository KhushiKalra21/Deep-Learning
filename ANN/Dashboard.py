import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import os
import random
import gdown
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile  # Import the tempfile module

# Google Drive File IDs
DATASET_FILE_ID = "1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_FILE_ID = "1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"

# Function to download and extract dataset
@st.cache_data
def load_dataset():
    dataset_url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
    output_zip = "customer_data.zip"

    # Use a temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, output_zip)
        gdown.download(dataset_url, output_path, quiet=True) # Changed quiet=False to True

        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Assuming dataset contains a single CSV file
        csv_filename = [f for f in os.listdir(temp_dir) if f.endswith(".csv")][0]
        df = pd.read_csv(os.path.join(temp_dir, csv_filename))
    return df

# Function to download and load model
@st.cache_resource
def load_model():
    model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    output_model = "customer_churn_model.h5"
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, output_model)
        gdown.download(model_url, output_path, quiet=True) # Changed quiet=False to True
        model = tf.keras.models.load_model(output_path)
    return model

# Load dataset
df = load_dataset()

# Randomly select 50,000 data points each time the model is trained
def get_random_sample(df):
    return df.sample(n=50000, random_state=random.randint(1, 1000))

# Preprocessing: Convert categorical variables to numerical
def preprocess_data(df):
    categorical_columns = ['gender', 'marital_status', 'education_level', 'occupation', 'preferred_store', 'payment_method',
                                        'store_city', 'store_state', 'season', 'product_color', 'product_material', 'promotion_channel',
                                        'promotion_type', 'promotion_target_audience']

    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# Sidebar: Model Hyperparameters
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=5)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05])
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Dense Layers", min_value=2, max_value=5, value=3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", min_value=32, max_value=512, step=32, value=128)

# "Train the Model" Button (Outside the Sidebar)
if st.button("🚀 Train the Model"):
    st.subheader("📥 Extracting 50,000 Random Data Points")
    df_sample = get_random_sample(df)
    df_sample = preprocess_data(df_sample)

    X = df_sample.drop(columns=['churned'])  # Features
    y = df_sample['churned']  # Target Variable

    # Define ANN Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))

    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.get(optimizer_choice)  # Get optimizer instance
    if optimizer_choice == "adam":
        optimizer.learning_rate = learning_rate
    elif optimizer_choice == "sgd":
        optimizer.learning_rate = learning_rate
    elif optimizer_choice == "rmsprop":
        optimizer.learning_rate = learning_rate

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train Model
    history = model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

    # Display Model Performance
    st.subheader("📊 Training Progress")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss Over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Plot Accuracy
    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title("Accuracy Over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    st.pyplot(fig)

    # Save trained model
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_model_file:
        model.save(tmp_model_file.name)
        st.download_button(
            label="📥 Download Updated Model",
            data=open(tmp_model_file.name, "rb"),
            file_name="updated_customer_churn_model.h5",
        )
    # Clean up the temporary file
    # os.remove(tmp_model_file.name) # Removed this line

st.subheader("📈 Data Insights & Visualization")

# Churn Distribution Plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["churned"], palette="coolwarm", ax=ax)
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Churn (0 = No, 1 = Yes)")
st.pyplot(fig)

# Churn vs Age Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df, x="age", hue="churned", kde=True, element="step", palette="coolwarm", ax=ax)
ax.set_title("Churn Distribution by Age")
st.pyplot(fig)

# Income Bracket vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="churned", y="income_bracket", data=df, palette="coolwarm", ax=ax)
ax.set_title("Income Bracket vs Churn")
st.pyplot(fig)

st.success("✅ Dashboard Ready!")
