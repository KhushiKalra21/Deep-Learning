import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import zipfile
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from io import BytesIO

# Set Streamlit Page Configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Download Dataset & Model from Google Drive
DATASET_URL = "https://drive.google.com/uc?id=1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_URL = "https://drive.google.com/uc?id=1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"

DATASET_PATH = "customer_data.zip"
MODEL_PATH = "customer_churn_model.h5"

# Function to download files
def download_file(url, output):
    gdown.download(url, output, quiet=False)

# Download dataset if not present
if not os.path.exists(DATASET_PATH):
    with st.spinner("Downloading dataset..."):
        download_file(DATASET_URL, DATASET_PATH)

# Extract ZIP file
with zipfile.ZipFile(DATASET_PATH, "r") as zip_ref:
    zip_ref.extractall("dataset")

csv_file = [f for f in os.listdir("dataset") if f.endswith(".csv")][0]  # Get CSV filename
df = pd.read_csv(os.path.join("dataset", csv_file))

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model..."):
        download_file(MODEL_URL, MODEL_PATH)

# Load Pre-trained Model
model = tf.keras.models.load_model(MODEL_PATH)

# Select 50,000 random samples each time
df = df.sample(n=50000, random_state=np.random.randint(1000))

# Data Preprocessing
def preprocess_data(df):
    df = df.copy()

    # Encode categorical features
    label_encoders = {}
    for col in ["gender", "loyalty_program"]:  
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["age", "income_bracket", "membership_years", "purchase_frequency",
                      "total_transactions", "days_since_last_purchase", "social_media_engagement",
                      "customer_support_calls"]
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, label_encoders, scaler

df, label_encoders, scaler = preprocess_data(df)

# Sidebar Hyperparameter Selection
st.sidebar.header("Model Hyperparameters")

epochs = st.sidebar.slider("Epochs", 10, 100, 50, 10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
num_layers = st.sidebar.slider("Hidden Layers", 1, 4, 2)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", 32, 512, 128, 32)

# Main Dashboard Layout
st.title("📊 Customer Churn Prediction Dashboard")

# Train Button
if st.button("🚀 Train the Model"):
    with st.spinner("Training the model..."):

        # Build ANN Model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(df.shape[1] - 1,)))

        for _ in range(num_layers):
            model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Compile Model
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        # Train Model
        history = model.fit(df.drop(columns=["churned"]), df["churned"], epochs=epochs, batch_size=32, verbose=0)

        # Plot Training Performance
        st.subheader("📈 Training Performance")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history.history["loss"], label="Loss")
        ax[0].set_title("Loss Curve")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].plot(history.history["accuracy"], label="Accuracy", color="green")
        ax[1].set_title("Accuracy Curve")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        st.pyplot(fig)

# Churn Insights Visualization
st.subheader("📊 Churn Insights")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Churn Distribution
sns.countplot(x="churned", data=df, palette="pastel", ax=ax[0])
ax[0].set_title("Churned vs Retained Customers")

# Churn vs Age
sns.histplot(df[df["churned"] == 1]["age"], kde=True, color="red", label="Churned", ax=ax[1])
sns.histplot(df[df["churned"] == 0]["age"], kde=True, color="blue", label="Retained", ax=ax[1])
ax[1].set_title("Churn Based on Age")
ax[1].legend()

st.pyplot(fig)

# Loyalty Program Analysis
st.subheader("🔍 Impact of Loyalty Program on Churn")
loyalty_churn = df.groupby("loyalty_program")["churned"].mean().reset_index()
st.bar_chart(loyalty_churn.set_index("loyalty_program"))

# Show Churn Probability Table
st.subheader("🔢 Churn Probability for Sample Customers")
df_sample = df.sample(10)
df_sample["Churn Probability"] = model.predict(df_sample.drop(columns=["churned"])).flatten()
st.write(df_sample[["age", "income_bracket", "loyalty_program", "Churn Probability"]])

# Recommendations
st.subheader("📌 Business Recommendations")
st.markdown("""
- **High-risk customers (Churn Probability > 0.7)** should receive personalized retention campaigns.
- **Loyalty program members** have lower churn rates; promote benefits to non-members.
- **Older customers** tend to churn less, while younger customers need engagement strategies.
""")
