import os
import random
import gdown
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 🚀 Fix GPU Issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.keras.backend.clear_session()

# 🚀 Define Correct File Paths (Subset Model & Data)
subset_model_url = "https://drive.google.com/uc?id=11zVuiwTzPKBk5mvu7Ma2vDVXOs3t5y89"
subset_data_url = "https://drive.google.com/uc?id=1FVnEkoFQJqufn_65_Euc8PrPyGBWAyLy"

subset_model_path = "subset_customer_churn_model.h5"
subset_data_path = "subset_customer_data.csv"

# 🚀 Download Model & Data if Not Exists
if not os.path.exists(subset_model_path):
    gdown.download(subset_model_url, subset_model_path, quiet=False)

if not os.path.exists(subset_data_path):
    gdown.download(subset_data_url, subset_data_path, quiet=False)

# 🚀 Load Data
df = pd.read_csv(subset_data_path)

# 🚀 Remove Completely Empty Columns
df.dropna(axis=1, how='all', inplace=True)

# 🚀 Extract 50,000 Random Samples Each Time
df_sampled = df.sample(n=50000, random_state=random.randint(1, 1000))

# 🚀 Streamlit UI
st.title("📊 ANN-Based Customer Churn Prediction Dashboard (Subset)")

# Sidebar Controls
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 1, 100, 50)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001])
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])

# 🚀 Train Model Button
if st.button("🚀 Train Model"):
    model = tf.keras.models.load_model(subset_model_path)
    model.compile(optimizer=optimizer_choice, loss='binary_crossentropy', metrics=['accuracy'])
    
    X = df_sampled.drop(columns=['churned'])
    y = df_sampled['churned']

    model.fit(X, y, epochs=epochs, verbose=1)
    
    st.success("✅ Model Trained Successfully!")

# 🚀 Data Visualization
st.subheader("🔍 Data Insights")

# 📈 Churn Distribution Pie Chart
fig, ax = plt.subplots()
df_sampled["churned"].value_counts().plot.pie(autopct="%1.1f%%", colors=["green", "red"], startangle=90, ax=ax)
ax.set_ylabel("")
ax.set_title("Churned vs. Retained Customers")
st.pyplot(fig)

# 📊 Churn Rate by Income Bracket
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x="income_bracket", y="churned", data=df_sampled, ax=ax, palette="coolwarm")
ax.set_title("Churn Rate by Income Bracket")
st.pyplot(fig)

# 📊 Membership Years vs. Churn
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_sampled, x="membership_years", hue="churned", multiple="stack", bins=30, palette="coolwarm", ax=ax)
ax.set_title("Churn Trend by Membership Years")
st.pyplot(fig)

# 📉 Purchase Frequency vs. Churn
fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(df_sampled[df_sampled["churned"] == 1]["purchase_frequency"], shade=True, color="red", label="Churned", ax=ax)
sns.kdeplot(df_sampled[df_sampled["churned"] == 0]["purchase_frequency"], shade=True, color="green", label="Retained", ax=ax)
ax.set_title("Purchase Frequency Density by Churn")
ax.legend()
st.pyplot(fig)

# 📊 Total Transactions vs. Churn (Boxplot)
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x="churned", y="total_transactions", data=df_sampled, ax=ax, palette="coolwarm")
ax.set_title("Total Transactions vs. Churn")
st.pyplot(fig)

# 📈 Heatmap of Feature Correlations
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df_sampled.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)
