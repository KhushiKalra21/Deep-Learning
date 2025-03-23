import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import os
import random
import gdown
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Google Drive File IDs
DATASET_FILE_ID = "1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_FILE_ID = "1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"

# Function to download and extract dataset
@st.cache_data
def load_dataset():
    dataset_url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
    output_zip = "customer_data.zip"

    try:
        # Use a temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, output_zip)
            gdown.download(dataset_url, output_path, quiet=True)

            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Assuming dataset contains a single CSV file
            csv_filename = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            if not csv_filename:
                raise FileNotFoundError("No CSV file found in the extracted archive.")
            csv_filename = csv_filename[0]
            df = pd.read_csv(os.path.join(temp_dir, csv_filename))
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to download and load model
@st.cache_resource
def load_model():
    model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    output_model = "customer_churn_model.h5"
    try:
        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, output_model)
            gdown.download(model_url, output_path, quiet=True)
            model = tf.keras.models.load_model(output_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dataset
df = load_dataset()
if df is None:
    st.stop()  # Stop if dataset loading failed.

# 🎯 Feature Selection (Adjusted for the churn dataset)
features = ['age', 'gender', 'income_bracket', 'num_of_purchases', 'total_spend', 'customer_rating',
            'days_since_last_purchase', 'marital_status', 'education_level', 'occupation',
            'preferred_store', 'payment_method', 'store_city', 'store_state', 'season',
            'product_color', 'product_material', 'promotion_channel', 'promotion_type',
            'promotion_target_audience']
target = 'churned'

# Preprocessing: Convert categorical variables to numerical and handle missing values
def preprocess_data(df):
    categorical_columns = ['gender', 'marital_status', 'education_level', 'occupation', 'preferred_store', 'payment_method',
                                        'store_city', 'store_state', 'season', 'product_color', 'product_material', 'promotion_channel',
                                        'promotion_type', 'promotion_target_audience']
    try:
        # Fill missing values with the mode for categorical columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        # Fill missing values of numerical columns with mean
        numerical_cols = ['age', 'income_bracket', 'num_of_purchases', 'total_spend', 'customer_rating', 'days_since_last_purchase']
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].mean())

        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()

df = preprocess_data(df)

# Handle Class Imbalance with SMOTE
X = df[features]
y = df[target]

smote = SMOTE(random_state=552627)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize Data
scaler = StandardScaler()
X_resampled[X.columns] = scaler.fit_transform(X_resampled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=552627)

# Compute Class Weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 ANN Model Dashboard - Customer Churn Prediction")

# Sidebar: Model Hyperparameters
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 5, 100, 5, 5)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5])
neurons_per_layer = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256, 512, 1024])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.1, 0.3)

# Select Optimizer
optimizers = {"adam": tf.keras.optimizers.Adam(learning_rate),
              "sgd": tf.keras.optimizers.SGD(learning_rate),
              "rmsprop": tf.keras.optimizers.RMSprop(learning_rate)}
optimizer = optimizers[optimizer_choice]

# Train Model Button
if st.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

        for _ in range(dense_layers):
            model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, class_weight=class_weight_dict, verbose=0)

    st.success("🎉 Model training complete!")

    # Model Performance
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.subheader("📊 Model Performance")
    st.metric(label="Test Accuracy", value=f"{accuracy:.4f}")
    st.metric(label="Test Loss", value=f"{loss:.4f}")

    # Training Performance Plots
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax[0].plot(history.history['accuracy'], label="Train Accuracy", color="blue")
    ax[0].plot(history.history['val_accuracy'], label="Validation Accuracy", color="orange")
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid()

    # Loss Plot
    ax[1].plot(history.history['loss'], label="Train Loss", color="blue")
    ax[1].plot(history.history['val_loss'], label="Validation Loss", color="orange")
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid()

    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=["Not Churned", "Churned"],
                yticklabels=["Not Churned", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📜 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    #  Feature Importance using SHAP
    st.subheader("🔍 Feature Importance")
    try:
        explainer = shap.Explainer(model, X_train[:100])
        shap_values = explainer(X_test[:100])

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:100], show=False)
        st.pyplot(fig)

        # Feature Importance Stats
        st.subheader("📌 Feature Importance Stats")
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': mean_abs_shap_values})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        st.dataframe(importance_df)
    except Exception as e:
        st.error(f"Error calculating SHAP values: {e}")

# 🔗 Follow Me on GitHub Button
st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://github.com/Rushil-K" target="_blank">
            <button style="background-color: #24292e; color: white; padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">
                ⭐ Follow Me on GitHub
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
