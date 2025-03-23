import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("customer_data.csv")  # Ensure this file exists
    return df

# Preprocess Data
def preprocess_data(df):
    df = df.dropna()  # Remove missing values
    label_enc = LabelEncoder()
    
    # Convert categorical features to numerical
    categorical_cols = ['gender', 'loyalty_program']  # Modify as per dataset
    for col in categorical_cols:
        df[col] = label_enc.fit_transform(df[col])
    
    X = df.drop(columns=["churned"])
    y = df["churned"]
    
    # Standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Create ANN Model
def build_model(input_dim, layers, neurons, activation, optimizer, dropout_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

    for _ in range(layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # Output layer
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# Load Data
df = load_data()
X, y = preprocess_data(df)
input_dim = X.shape[1]

# Streamlit UI
st.title("Customer Churn Prediction - ANN Dashboard")

# Sidebar: Hyperparameters
st.sidebar.header("Hyperparameters")

epochs = st.sidebar.slider("Epochs", min_value=10, max_value=100, step=10, value=50)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"], index=0)
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0)
layers = st.sidebar.slider("Number of Layers", min_value=2, max_value=5, value=3)
neurons = st.sidebar.slider("Neurons per Layer", min_value=32, max_value=256, step=32, value=128)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, step=0.1, value=0.2)

# Main UI: Train Model Button
if st.button("Train the Model"):
    model = build_model(input_dim, layers, neurons, activation, optimizer, dropout_rate)
    
    history = model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=0)
    
    st.success("Model Training Complete!")

    # Accuracy Plot
    st.subheader("Training Progress")
    fig, ax = plt.subplots()
    ax.plot(history.history["accuracy"], label="Train Accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    # Churn Distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["churned"], palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = abs(model.layers[0].get_weights()[0]).sum(axis=1)
    feature_names = df.drop(columns=["churned"]).columns
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(y=fi_df["Feature"], x=fi_df["Importance"], palette="viridis", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
