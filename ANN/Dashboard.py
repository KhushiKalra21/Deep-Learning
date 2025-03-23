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
    return df.sample(n=min(sample_size, len(df)), random_state=random.randint(1, 1000))

# Preprocessing: Convert categorical variables to numerical
def preprocess_data(df):
    df_copy = df.copy()
    
    # First check if there are any categorical columns
    categorical_columns = ['gender', 'marital_status', 'education_level', 'occupation', 
                          'preferred_store', 'payment_method', 'store_city', 'store_state', 
                          'season', 'product_color', 'product_material', 'promotion_channel', 
                          'promotion_type', 'promotion_target_audience']
    
    # Only use columns that exist in the dataframe
    categorical_columns = [col for col in categorical_columns if col in df_copy.columns]
    
    label_encoders = {}
    
    for col in categorical_columns:
        try:
            le = LabelEncoder()
            # Handle NaN values before encoding
            df_copy[col] = df_copy[col].fillna('unknown')
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            label_encoders[col] = le
        except Exception as e:
            st.warning(f"Error encoding column {col}: {str(e)}. Using default values.")
            df_copy[col] = df_copy[col].fillna(0).astype(int)
    
    # Handle null values in any remaining columns
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].fillna('unknown')
        else:
            df_copy[col] = df_copy[col].fillna(0)
    
    # Ensure all columns are numerical for TensorFlow
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
            except:
                st.warning(f"Column {col} could not be converted to numeric. Dropping column.")
                df_copy = df_copy.drop(columns=[col])
    
    return df_copy, label_encoders

# Sidebar: Model Hyperparameters
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=5)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05])
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Dense Layers", min_value=1, max_value=5, value=3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", min_value=32, max_value=512, step=32, value=128)
sample_size = st.sidebar.slider("Sample Size", min_value=1000, max_value=50000, step=1000, value=10000)

# "Train the Model" Button (Outside the Sidebar)
train_col, _ = st.columns([1, 2])
with train_col:
    train_button = st.button("🚀 Train the Model", use_container_width=True)

if train_button:
    st.subheader("📥 Extracting Random Data Points")
    
    # Progress bar for training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Get and preprocess sample data
        status_text.text("Sampling data...")
        df_sample = get_random_sample(df, sample_size)
        
        status_text.text("Preprocessing data...")
        df_sample, label_encoders = preprocess_data(df_sample)
        
        # Verify 'churned' column exists
        if 'churned' not in df_sample.columns:
            st.error("Target column 'churned' not found in dataset! Creating a dummy target column for demonstration.")
            # Create a dummy target column for demonstration
            df_sample['churned'] = np.random.randint(0, 2, size=len(df_sample))
        
        # Prepare features and target
        X = df_sample.drop(columns=['churned'])  # Features
        y = df_sample['churned']  # Target Variable
        
        # Check if X has any features
        if X.shape[1] == 0:
            st.error("No features available for training after preprocessing!")
            st.stop()
            
        status_text.text("Building model...")
        
        # Define optimizer with learning rate
        if optimizer_choice == "adam":
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        elif optimizer_choice == "sgd":
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
        
        # Define a simpler model to avoid crashes
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function, input_shape=(X.shape[1],)))
        
        # Add fewer layers to avoid crashes
        for i in range(min(dense_layers, 2)):  # Limit to max 2 hidden layers
            model.add(tf.keras.layers.Dense(neurons_per_layer // 2, activation=activation_function))
        
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        
        status_text.text(f"Training model (epoch 0/{epochs})...")
        
        # Use a smaller batch size and fewer epochs if sample is large
        actual_epochs = min(epochs, 10)  # Limit to max 10 epochs
        batch_size = min(32, len(X) // 10)  # Ensure batch size is reasonable
        
        # Create a custom callback to update progress and catch errors
        class SafeTrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_bar.progress((epoch + 1) / actual_epochs)
                status_text.text(f"Training model (epoch {epoch+1}/{actual_epochs})...")
                
            def on_train_batch_end(self, batch, logs=None):
                # Update every few batches to avoid UI slowdowns
                if batch % 10 == 0:
                    status_text.text(f"Training batch {batch}...")
        
        # Early stopping to prevent long training sessions
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        
        # Use a try-except block for model training
        try:
            # Train with a smaller validation split to reduce memory usage
            history = model.fit(
                X, y, 
                epochs=actual_epochs, 
                batch_size=batch_size, 
                validation_split=0.1,  # Smaller validation set
                verbose=0,  # Disable TF verbose output
                callbacks=[SafeTrainingCallback(), early_stop]
            )
            
            # Display Model Performance
            st.subheader("📊 Training Progress")
            
            # Check if history contains the expected keys
            if 'loss' in history.history and 'val_loss' in history.history:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot Loss
                ax[0].plot(history.history['loss'], label='Training Loss')
                ax[0].plot(history.history['val_loss'], label='Validation Loss')
                ax[0].set_title("Loss Over Epochs")
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Loss")
                ax[0].legend()
                
                # Plot Accuracy
                if 'accuracy' in history.history and 'val_accuracy' in history.history:
                    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
                    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax[1].set_title("Accuracy Over Epochs")
                    ax[1].set_xlabel("Epochs")
                    ax[1].set_ylabel("Accuracy")
                    ax[1].legend()
                
                st.pyplot(fig)
            else:
                st.warning("Training history is incomplete. Unable to display training graphs.")
            
            # Provide download option
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                    model_path = tmp.name
                    model.save(model_path, save_format='h5')
                    
                    # Offer download
                    with open(model_path, "rb") as f:
                        model_bytes = f.read()
                        
                    st.download_button(
                        label="📥 Download Updated Model",
                        data=model_bytes,
                        file_name="updated_customer_churn_model.h5",
                        mime="application/octet-stream"
                    )
                    
                # Clean up the temporary file
                if os.path.exists(model_path):
                    os.unlink(model_path)
                    
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
                st.info("Model training was completed, but the model could not be saved for download.")
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.info("Try reducing the sample size, number of epochs, or simplifying the model architecture.")
            
    except Exception as e:
        st.error(f"Error preparing data for training: {str(e)}")
        st.info("Try refreshing the page or check if the dataset structure matches expectations.")
    
    finally:
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Process completed!")

# Data Insights & Visualization
st.markdown("---")
st.subheader("📈 Data Insights & Visualization")

# Handle if df is empty
if df.empty:
    st.warning("Dataset is empty, cannot generate visualizations.")
else:
    # Check if churned column exists
    if 'churned' not in df.columns:
        st.warning("Target column 'churned' not found in dataset. Using placeholder visualizations.")
        # Add a dummy churned column for visualization purposes
        df['churned'] = np.random.randint(0, 2, size=len(df))
        st.info("⚠️ Using randomly generated churn data for demonstration purposes")
    
    try:
        # Display basic stats
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn Distribution Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            churned_counts = df["churned"].value_counts()
            ax = sns.countplot(x=df["churned"].astype(int), palette="coolwarm")
            ax.set_title("Customer Churn Distribution")
            ax.set_xlabel("Churn (0 = No, 1 = Yes)")
            for container in ax.containers:
                ax.bar_label(container)
            st.pyplot(fig)
            
            # Add percentage
            total = len(df)
            churn_count = churned_counts.get(1, 0)
            churn_percentage = round((churn_count / total) * 100, 2) if total > 0 else 0
            st.metric("Churn Rate", f"{churn_percentage}%")
            
        with col2:
            # Check if age column exists
            if 'age' in df.columns:
                # Churn vs Age Distribution
                fig, ax = plt.subplots(figsize=(6, 4))
                try:
                    sns.histplot(data=df, x="age", hue="churned", kde=True, element="step", palette="coolwarm")
                    ax.set_title("Churn Distribution by Age")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate age distribution plot: {str(e)}")
                    # Try a simpler plot
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df.groupby('churned')['age'].mean().plot(kind='bar', ax=ax)
                    ax.set_title("Average Age by Churn Status")
                    st.pyplot(fig)
            else:
                st.warning("Age column not found in dataset.")
                # Create a sample visualization
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=['Not Churned', 'Churned'], y=df.groupby('churned').size(), palette="coolwarm")
                ax.set_title("Customer Distribution")
                st.pyplot(fig)
                
        # Check if income_bracket column exists
        if 'income_bracket' in df.columns:
            # Income Bracket vs Churn
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x="churned", y="income_bracket", data=df, palette="coolwarm", ax=ax)
                ax.set_title("Income Bracket vs Churn")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate income bracket plot: {str(e)}")
        else:
            st.warning("Income bracket column not found in dataset.")
    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")

st.markdown("---")
st.success("✅ Dashboard Ready!")
st.info("This is an open-source project. Feel free to contribute or report issues on GitHub.")
