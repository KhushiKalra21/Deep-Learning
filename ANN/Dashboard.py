import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import zipfile
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit Page Configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Google Drive File IDs
MODEL_FILE_ID = "11zVuiwTzPKBk5mvu7Ma2vDVXOs3t5y89"
DATA_FILE_ID = "1FVnEkoFQJqufn_65_Euc8PrPyGBWAyLy"

# Filenames
MODEL_PATH = "subset_model.h5"
DATA_PATH = "subset_data.csv"

# Download files if they don't exist
def download_file(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Download Model & Dataset
download_file(MODEL_FILE_ID, MODEL_PATH)
download_file(DATA_FILE_ID, DATA_PATH)

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)

# Load Data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("Dataset file not found!")

# Drop completely null columns
df.dropna(axis=1, how="all", inplace=True)

# Select 50,000 random rows each time
df_sample = df.sample(n=50000, random_state=random.randint(1, 1000))

# Sidebar Controls
st.sidebar.header("Hyperparameter Settings")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1])
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])

# Train Model Button
if st.sidebar.button("Train Model"):
    st.sidebar.write("Training... (Mock Implementation)")
    # Here you should implement model training with hyperparameters

# Dashboard Title
st.title("Customer Churn Prediction Dashboard")

# Display Data Sample
st.subheader("📊 Sample Data (Random 50,000 Rows)")
st.dataframe(df_sample.head())

# **Visualizations**
st.subheader("📈 Data Insights")

# Churn Count Plot
fig, ax = plt.subplots()
sns.countplot(x=df_sample["churned"], palette="viridis", ax=ax)
st.pyplot(fig)

# Age Distribution
fig, ax = plt.subplots()
sns.histplot(df_sample["age"], bins=30, kde=True, ax=ax, color="blue")
st.pyplot(fig)

# Purchase Frequency vs. Churn
fig, ax = plt.subplots()
sns.boxplot(x="churned", y="purchase_frequency", data=df_sample, ax=ax, palette="coolwarm")
st.pyplot(fig)

# **Predictions**
st.subheader("🔮 Predict Churn for a New Customer")

# Input Fields
age = st.number_input("Age", min_value=18, max_value=80, value=30)
income = st.number_input("Income Bracket", min_value=1, max_value=10, value=5)
loyalty = st.selectbox("Loyalty Program", [0, 1])
transactions = st.number_input("Total Transactions", min_value=0, max_value=500, value=50)

# Predict Button
if st.button("Predict Churn"):
    input_data = np.array([[age, income, loyalty, transactions]])
    prediction = model.predict(input_data)
    st.write(f"Churn Probability: {prediction[0][0]:.2f}")

# Footer
st.markdown("**Project by [Your Name] | AI-Powered Churn Prediction**")
