import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import os
import random
import gdown
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

# Set page configuration at the top
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Google Drive File IDs
DATASET_FILE_ID = "1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_FILE_ID = "1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"

# Function to download and extract dataset
@st.cache_data
def load_dataset():
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        output_zip = os.path.join(temp_dir, "customer_data.zip")
        dataset_url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
        
        with st.spinner("Downloading dataset..."):
            gdown.download(dataset_url, output_zip, quiet=True)

        # Create extraction directory
        extract_dir = os.path.join(temp_dir, "dataset")
        os.makedirs(extract_dir, exist_ok=True)
        
        with st.spinner("Extracting dataset..."):
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        # Find CSV file
        csv_files = [f for f in os.listdir(extract_dir) if f.endswith(".csv")]
        if not csv_files:
            st.error("No CSV file found in the dataset")
            return None
            
        csv_filename = csv_files[0]
        df = pd.read_csv(os.path.join(extract_dir, csv_filename))
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Function to download and load model
@st.cache_resource
def load_model():
    try:
        temp_dir = tempfile.mkdtemp()
        output_model = os.path.join(temp_dir, "customer_churn_model.h5")
        model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        
        with st.spinner("Downloading pre-trained model..."):
            gdown.download(model_url, output_model, quiet=True)
        
        model = tf.keras.models.load_model(output_model)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Add a title banner
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# Load dataset with error handling
with st.spinner("Loading data..."):
    df = load_dataset()

if df is None:
    st.error("Failed to load dataset. Please check the Google Drive file ID or your internet connection.")
    st.stop()

# Randomly select data points for model training
def get_random_sample(df, sample_size=50000):
    if len(df) < sample_size:
        st.warning(f"Dataset contains fewer than {sample_size} records. Using entire dataset.")
        return df
    return df.sample(n=sample_size, random_state=random.randint(1, 1000))

# Preprocessing: Convert categorical variables to numerical
def preprocess_data(df):
    df_copy = df.copy()
    
    categorical_columns = ['gender', 'marital_status', 'education_level', 'occupation', 
                          'preferred_store', 'payment_method', 'store_city', 'store_state', 
                          'season', 'product_color', 'product_material', 'promotion_channel', 
                          'promotion_type', 'promotion_target_audience']
    
    # Only use columns that exist in the dataframe
    categorical_columns = [col for col in categorical_columns if col in df_copy.columns]
    
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        label_encoders[col] = le
    
    # Handle null values if any
    df_copy = df_copy.fillna(0)
    
    return df_copy, label_encoders

# Sidebar: Model Hyperparameters
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=5)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05])
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Dense Layers", min_value=2, max_value=5, value=3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", min_value=32, max_value=512, step=32, value=128)
sample_size = st.sidebar.slider("Sample Size", min_value=1000, max_value=100000, step=1000, value=50000)

# "Train the Model" Button (Outside the Sidebar)
train_col, _ = st.columns([1, 2])
with train_col:
    train_button = st.button("🚀 Train the Model", use_container_width=True)

if train_button:
    st.subheader("📥 Extracting Random Data Points")
    
    # Progress bar for training
    progress_bar = st.progress(0)
    
    # Get and preprocess sample data
    df_sample = get_random_sample(df, sample_size)
    df_sample, label_encoders = preprocess_data(df_sample)
    
    # Verify 'churned' column exists
    if 'churned' not in df_sample.columns:
        st.error("Target column 'churned' not found in dataset!")
        st.stop()
    
    # Prepare features and target
    X = df_sample.drop(columns=['churned'])  # Features
    y = df_sample['churned']  # Target Variable
    
    # Define optimizer with learning rate
    if optimizer_choice == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    # Define ANN Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
    
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))
        model.add(tf.keras.layers.Dropout(0.2))  # Add dropout for regularization
    
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    # Custom callback to update progress bar
    class ProgressBarCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / epochs)
    
    # Train Model
    with st.spinner(f"Training model (epoch 0/{epochs})..."):
        history = model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=32, 
            validation_split=0.2, 
            verbose=1,
            callbacks=[ProgressBarCallback()]
        )
    
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
    
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        model_path = tmp.name
        model.save(model_path)
        
        # Offer download
        with open(model_path, "rb") as f:
            model_bytes = f.read()
            
        st.download_button(
            label="📥 Download Updated Model",
            data=model_bytes,
            file_name="updated_customer_churn_model.h5",
            mime="application/octet-stream"
        )
        
    st.success("✅ Model training complete!")

# Data Insights & Visualization
st.markdown("---")
st.subheader("📈 Data Insights & Visualization")

# Handle if df is empty
if df.empty:
    st.warning("Dataset is empty, cannot generate visualizations.")
else:
    # Check if churned column exists
    if 'churned' not in df.columns:
        st.error("Target column 'churned' not found in dataset! Cannot generate churn visualizations.")
    else:
        # Display basic stats
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn Distribution Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            churned_counts = df["churned"].value_counts()
            sns.countplot(x=df["churned"], palette="coolwarm", ax=ax)
            ax.set_title("Customer Churn Distribution")
            ax.set_xlabel("Churn (0 = No, 1 = Yes)")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)
            
            # Add percentage
            churn_percentage = round((churned_counts.get(1, 0) / len(df)) * 100, 2)
            st.metric("Churn Rate", f"{churn_percentage}%")
            
        with col2:
            # Check if age column exists
            if 'age' in df.columns:
                # Churn vs Age Distribution
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df, x="age", hue="churned", kde=True, element="step", palette="coolwarm", ax=ax)
                ax.set_title("Churn Distribution by Age")
                st.pyplot(fig)
            else:
                st.warning("Age column not found in dataset.")
                
        # Check if income_bracket column exists
        if 'income_bracket' in df.columns:
            # Income Bracket vs Churn
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x="churned", y="income_bracket", data=df, palette="coolwarm", ax=ax)
            ax.set_title("Income Bracket vs Churn")
            st.pyplot(fig)
        else:
            st.warning("Income bracket column not found in dataset.")

st.markdown("---")
st.success("✅ Dashboard Ready!")
