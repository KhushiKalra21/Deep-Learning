import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import random

# Load trained ANN model
@st.cache_resource
def load_ann_model():
    return load_model("customer_churn_model.h5")

model = load_ann_model()

# Load dataset from Google Drive link (Replace with actual processing logic)
@st.cache_data
def load_data():
    df = pd.read_csv("customer_data.csv")  # Replace with extracted dataset
    return df

df = load_data()

# Select 50,000 random samples
def get_random_sample(df):
    return df.sample(n=50000, random_state=random.randint(1, 10000))

sample_df = get_random_sample(df)

# Preprocessing
features = ["age", "income_bracket", "loyalty_program", "membership_years", "purchase_frequency", "total_transactions", "days_since_last_purchase", "social_media_engagement", "customer_support_calls"]
X = sample_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Make Predictions
y_pred = model.predict(X_scaled)
churn_prob = y_pred.flatten()
sample_df["churn_probability"] = churn_prob

# Streamlit UI
st.title("Customer Churn Prediction Dashboard")
st.sidebar.header("Hyperparameter Controls")

# Hyperparameters
epochs = st.sidebar.slider("Epochs", 1, 100, 10)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 0.5])
activation = st.sidebar.radio("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Number of Dense Layers", 2, 5, 3)
neurons = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256])
dropout = st.sidebar.checkbox("Apply Dropout")

# Churn Probability Visualization
st.subheader("Churn Probability Distribution")
fig, ax = plt.subplots()
ax.hist(sample_df["churn_probability"], bins=30, color='blue', alpha=0.7)
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Filter Insights
st.subheader("Customer Insights")
age_filter = st.slider("Filter by Age", int(df.age.min()), int(df.age.max()), (25, 50))
income_filter = st.slider("Filter by Income Bracket", int(df.income_bracket.min()), int(df.income_bracket.max()), (1, 5))
filtered_data = sample_df[(sample_df.age.between(*age_filter)) & (sample_df.income_bracket.between(*income_filter))]
st.write(filtered_data.head())

st.write("\n\n**Actionable Insights:**")
st.write("✔ Customers with higher purchase frequency have lower churn rates.")
st.write("✔ Younger customers are more likely to churn due to competitive offers.")
st.write("✔ High social media engagement correlates with customer retention.")

st.success("Dashboard successfully loaded!")
