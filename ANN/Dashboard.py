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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import time  # Import time for measuring execution
from typing import Tuple, List, Dict, Optional
from collections.abc import Callable

# Google Drive File IDs
DATASET_FILE_ID = "1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_FILE_ID = "1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"

# --- Constants ---
RANDOM_STATE = 552134
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# --- Helper Functions ---

def download_file(url: str, output_path: str, quiet: bool = True) -> bool:
    """Downloads a file from a URL with retries.

    Args:
        url: The URL of the file to download.
        output_path: The path to save the downloaded file.
        quiet: Whether to suppress console output.

    Returns:
        True if the download was successful, False otherwise.
    """
    for attempt in range(MAX_RETRIES):
        try:
            gdown.download(url, output_path, quiet=quiet)
            return True  # Success
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Download failed (attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)  # Wait before retrying
            else:
                st.error(f"Download failed after {MAX_RETRIES} attempts: {e}")
                return False  # Failure
    return False

def extract_zip(zip_path: str, extract_path: str) -> bool:
    """Extracts a ZIP file with error handling.

    Args:
        zip_path: Path to the ZIP file.
        extract_path: Path to extract the contents.

    Returns:
        True if extraction was successful, False otherwise.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return True
    except zipfile.BadZipFile:
        st.error(f"Error: {zip_path} is not a valid ZIP file.")
        return False
    except Exception as e:
        st.error(f"Error extracting ZIP file: {e}")
        return False



# Function to download and extract dataset
@st.cache_data
def load_dataset() -> Optional[pd.DataFrame]:
    """Downloads and extracts the dataset, handling potential errors.

    Returns:
        The loaded Pandas DataFrame, or None on error.
    """
    dataset_url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
    output_zip = "customer_data.zip"

    try:
        # Use a temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, output_zip)
            if not download_file(dataset_url, output_path, quiet=True):
                raise Exception(f"Failed to download dataset from {dataset_url}")

            if not extract_zip(output_path, temp_dir):
                raise Exception(f"Failed to extract dataset from {output_zip}")

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
def load_model() -> Optional[tf.keras.Model]:
    """Downloads and loads the TensorFlow model, handling errors.

    Returns:
        The loaded TensorFlow model, or None on error.
    """
    model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    output_model = "customer_churn_model.h5"
    try:
        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, output_model)
            if not download_file(model_url, output_path, quiet=True):
                raise Exception(f"Failed to download model from {model_url}")
            model = tf.keras.models.load_model(output_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dataset
df = load_dataset()
if df is None:
    st.stop()  # Stop if dataset loading failed.

# --- Preprocessing ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values and encodes categorical features.

    Args:
        df: The input Pandas DataFrame.

    Returns:
        The preprocessed Pandas DataFrame.
    """
    categorical_columns = ['gender', 'marital_status', 'education_level', 'occupation',
                           'preferred_store', 'payment_method', 'store_city', 'store_state',
                           'season', 'product_color', 'product_material', 'promotion_channel',
                           'promotion_type', 'promotion_target_audience']
    numerical_cols = ['age', 'income_bracket', 'num_of_purchases', 'total_spend',
                      'customer_rating', 'days_since_last_purchase']

    try:
        # Impute missing values
        for col in categorical_columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        for col in numerical_cols:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

        # Encode categorical features
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

# --- Feature Selection ---
features = ['age', 'gender', 'income_bracket', 'num_of_purchases', 'total_spend',
            'customer_rating', 'days_since_last_purchase', 'marital_status',
            'education_level', 'occupation', 'preferred_store', 'payment_method',
            'store_city', 'store_state', 'season', 'product_color', 'product_material',
            'promotion_channel', 'promotion_type', 'promotion_target_audience']
target = 'churned'

X = df[features]
y = df[target]

# --- SMOTE and Train-Test Split ---
smote = SMOTE(random_state=RANDOM_STATE)
X_resampled, y_resampled = smote.fit_resample(X, y)

scaler = StandardScaler()
X_resampled[X.columns] = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=RANDOM_STATE
)

# --- Class Weights ---
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# --- Model Definition ---
def create_model(input_shape: Tuple[int],
                 optimizer: tf.keras.optimizers.Optimizer,
                 activation_function: str,
                 dense_layers: int,
                 neurons_per_layer: int,
                 dropout_rate: float) -> tf.keras.Model:
    """Creates and compiles the TensorFlow model.

    Args:
        input_shape: Shape of the input data.
        optimizer: The optimizer to use.
        activation_function: The activation function for the dense layers.
        dense_layers: The number of dense layers.
        neurons_per_layer: The number of neurons per dense layer.
        dropout_rate: The dropout rate.

    Returns:
        The compiled TensorFlow model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_and_evaluate_model(model: tf.keras.Model,
                           x_train: np.ndarray,
                           y_train: np.ndarray,
                           x_test: np.ndarray,
                           y_test: np.ndarray,
                           epochs: int,
                           class_weight: Dict[int, float],
                           validation_split: float = 0.2,
                           batch_size: int = 32) -> Tuple[tf.keras.Model, Dict[str, List[float]], float, float]:
    """Trains and evaluates the model, returning history and metrics.

        Args:
            model: The TensorFlow model to train.
            x_train: Training data features.
            y_train: Training data labels.
            x_test: Testing data features.
            y_test: Testing data labels.
            epochs: Number of epochs to train for.
            class_weight: Class weights dictionary.
            validation_split: The fraction of the training data to be used as validation data.
            batch_size: Batch size for training.

        Returns:
            A tuple containing:
            - The trained TensorFlow model.
            - The training history.
            - The test loss
            - The test accuracy
    """
    start_time = time.time()
    history = model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=validation_split, class_weight=class_weight, verbose=0
    ).history
    end_time = time.time()
    training_time = end_time - start_time

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return model, history, loss, accuracy, training_time



# --- Streamlit App ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 ANN Model Dashboard - Customer Churn Prediction")

# --- Sidebar ---
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 5, 100, 5, 5)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox(
    "Activation Function", ["relu", "sigmoid", "tanh", "softmax"]
)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5])
neurons_per_layer = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256, 512, 1024])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.1, 0.3)

# Select Optimizer
optimizers = {
    "adam": Adam(learning_rate=learning_rate),
    "sgd": SGD(learning_rate=learning_rate),
    "rmsprop": RMSprop(learning_rate=learning_rate),
}
optimizer = optimizers[optimizer_choice]

# --- Train Model Button ---
if st.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        model = create_model(
            X_train.shape[1], optimizer, activation_function, dense_layers,
            neurons_per_layer, dropout_rate
        )
        model, history, loss, accuracy, training_time = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, epochs, class_weight_dict,
            VALIDATION_SPLIT, BATCH_SIZE
        )

    st.success("🎉 Model training complete!")
    st.write(f"Training time: {training_time:.2f} seconds")  # Display training time

    # Model Performance
    st.subheader("📊 Model Performance")
    st.metric(label="Test Accuracy", value=f"{accuracy:.4f}")
    st.metric(label="Test Loss", value=f"{loss:.4f}")

    # Training Performance Plots
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))  # Add space for ROC curve

    # Accuracy Plot
    ax[0].plot(history['accuracy'], label="Train Accuracy", color="blue")
    ax[0].plot(
        history['val_accuracy'], label="Validation Accuracy", color="orange"
    )
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid()

    # Loss Plot
    ax[1].plot(history['loss'], label="Train Loss", color="blue")
    ax[1].plot(history['val_loss'], label="Validation Loss", color="orange")
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid()

    # ROC Curve
    y_pred_proba = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ANN').plot(ax=ax[2])
    ax[2].set_title('ROC Curve')
    ax[2].grid()

    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="coolwarm",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
    )
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
