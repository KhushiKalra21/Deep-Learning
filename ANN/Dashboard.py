import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import io
import requests
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title
st.title("📊 Customer Churn Prediction Dashboard")

# Sidebar for Hyperparameter Selection
st.sidebar.header("🔧 Hyperparameter Tuning")

# Hyperparameter Slicers
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10, step=1)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
activation_fn = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.select_slider("Number of Dense Layers", options=[2, 3, 4, 5], value=3)
neurons = st.sidebar.select_slider("Neurons per Layer", options=[2**n for n in range(5, 11)], value=32)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)

# Load Dataset from Google Drive ZIP
@st.cache_data
def load_dataset():
    dataset_url = "https://drive.google.com/uc?export=download&id=1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
    response = requests.get(dataset_url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    
    # Extract CSV file inside ZIP
    csv_filename = zip_file.namelist()[0]  # Assume first file is dataset
    df = pd.read_csv(zip_file.open(csv_filename))

    # Select 50,000 random rows
    df_sampled = df.sample(n=50000, random_state=random.randint(1, 10000))
    return df_sampled

# Load dataset
st.subheader("📂 Loading Dataset...")
df = load_dataset()
st.write(df.head())

# Preprocess Data
def preprocess_data(df):
    categorical_cols = ["gender", "loyalty_program", "income_bracket"]
    numerical_cols = ["age", "membership_years", "purchase_frequency", "total_transactions",
                      "days_since_last_purchase", "social_media_engagement", "customer_support_calls"]

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical variables
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=["churned"])
    y = df["churned"]

    return X, y

# Preprocess dataset
X, y = preprocess_data(df)

# Load Trained Model from Google Drive
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?export=download&id=1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"
    response = requests.get(model_url)
    
    with open("customer_churn_model.h5", "wb") as f:
        f.write(response.content)
    
    return tf.keras.models.load_model("customer_churn_model.h5")

# Load model
st.subheader("📥 Loading Pre-trained Model...")
model = load_model()

# Train Model Button (Outside Sidebar)
if st.button("🚀 Train Model"):
    st.subheader("🔄 Training Model...")

    # Define Model
    new_model = tf.keras.models.Sequential()
    new_model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
    
    for _ in range(dense_layers):
        new_model.add(tf.keras.layers.Dense(neurons, activation=activation_fn))
        new_model.add(tf.keras.layers.Dropout(dropout_rate))
    
    new_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Compile Model
    new_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train Model
    history = new_model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save New Model
    new_model.save("customer_churn_trained_model.h5")
    st.success("✅ Model Training Completed & Saved as `customer_churn_trained_model.h5`!")

    # Display Training Accuracy
    st.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    st.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Plot Training & Validation Accuracy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["accuracy"], label="Train Accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    st.pyplot(fig)

# Predictions on Sample Data
st.subheader("🔍 Predicting Customer Churn")
random_sample = X.sample(10)
predictions = model.predict(random_sample)

df_pred = random_sample.copy()
df_pred["Predicted Churn Probability"] = predictions
df_pred["Predicted Churn"] = (predictions > 0.5).astype(int)

st.write(df_pred)

# Visualizations
st.subheader("📈 Visualizations")

# Churn Distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["churned"], palette="coolwarm", ax=ax)
ax.set_title("Churn Distribution")
st.pyplot(fig)

# Churn vs Age
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df, x="age", hue="churned", kde=True, palette="coolwarm", ax=ax)
ax.set_title("Age Distribution by Churn")
st.pyplot(fig)

# Churn vs Purchase Frequency
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x="churned", y="purchase_frequency", data=df, palette="coolwarm", ax=ax)
ax.set_title("Purchase Frequency by Churn Status")
st.pyplot(fig)
