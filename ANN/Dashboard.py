import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Function to generate synthetic customer data
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    # Generate features
    age = np.random.normal(40, 10, n_samples).astype(int)
    age = np.clip(age, 18, 80)  # Clip age to realistic values
    
    income_bracket = np.random.normal(50000, 20000, n_samples).astype(int)
    income_bracket = np.clip(income_bracket, 20000, 150000)
    
    # Generate categorical features
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    marital_status = np.random.choice(
        ['Single', 'Married', 'Divorced', 'Widowed'], 
        n_samples, 
        p=[0.3, 0.5, 0.15, 0.05]
    )
    
    education_level = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'], 
        n_samples,
        p=[0.3, 0.4, 0.2, 0.1]
    )
    
    tenure = np.random.gamma(2, 2, n_samples).astype(int)
    tenure = np.clip(tenure, 0, 10)  # Years as customer
    
    products_owned = np.random.poisson(2, n_samples)
    products_owned = np.clip(products_owned, 1, 6)
    
    monthly_spend = np.random.gamma(shape=5, scale=20, size=n_samples).astype(int)
    monthly_spend = np.clip(monthly_spend, 20, 500)
    
    # Generate interaction features that affect churn
    service_issues = np.random.randint(0, 5, n_samples)
    satisfaction_score = np.random.normal(7, 2, n_samples)
    satisfaction_score = np.clip(satisfaction_score, 0, 10)
    
    # Create a churn probability model
    churn_probability = 0.1 + \
                        0.2 * (service_issues > 2) + \
                        0.3 * (satisfaction_score < 5) + \
                        0.1 * (tenure < 2) + \
                        0.1 * (monthly_spend < 50) - \
                        0.1 * (products_owned > 3)
    
    churn_probability = np.clip(churn_probability, 0.01, 0.99)
    
    # Generate churn outcome
    churned = np.random.binomial(1, churn_probability)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income_bracket': income_bracket,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education_level,
        'tenure': tenure,
        'products_owned': products_owned,
        'monthly_spend': monthly_spend,
        'service_issues': service_issues,
        'satisfaction_score': satisfaction_score.round(1),
        'churned': churned
    })
    
    return df

# Cache the training function to improve performance
@st.cache_resource
def train_model(_df, test_size=0.2, n_estimators=100):
    # Convert categorical variables
    df_encoded = pd.get_dummies(_df, columns=['gender', 'marital_status', 'education_level'], drop_first=True)
    
    # Split features and target
    X = df_encoded.drop('churned', axis=1)
    y = df_encoded['churned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create and train a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'feature_importance': dict(zip(X.columns, pipeline.named_steps['classifier'].feature_importances_))
    }
    
    return pipeline, metrics

# Title and introduction
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard demonstrates customer churn prediction using machine learning.
The app uses synthetically generated customer data that mimics real-world patterns.
""")
st.markdown("---")

# Sidebar for data and model parameters
st.sidebar.header("🔧 Settings")

# Data parameters
data_tab, model_tab = st.sidebar.tabs(["Data Settings", "Model Settings"])

with data_tab:
    data_size = st.slider("Sample Size", min_value=500, max_value=5000, value=1000, step=500)
    include_noise = st.checkbox("Include Random Noise", value=False)

with model_tab:
    n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=50, step=10)
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)

# Generate or load data
data_container = st.container()
with data_container:
    st.subheader("📋 Customer Data")
    
    # Generate synthetic data
    with st.spinner("Generating synthetic customer data..."):
        df = generate_synthetic_data(data_size)
        
        # Add noise if requested
        if include_noise:
            df['random_feature'] = np.random.normal(0, 1, len(df))
        
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing 10 of {len(df)} records")

# Train model button
train_col, _ = st.columns([1, 2])
with train_col:
    train_button = st.button("🚀 Train Churn Prediction Model", use_container_width=True)

# Model training section
if train_button:
    with st.spinner("Training model... Please wait"):
        # Train the model
        try:
            model, metrics = train_model(df, test_size, n_estimators)
            
            # Display metrics
            st.subheader("📈 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['precision']:.2%}")
            col3.metric("Recall", f"{metrics['recall']:.2%}")
            
            # Feature importance plot
            st.subheader("🔍 Feature Importance")
            
            # Sort feature importance
            importance_df = pd.DataFrame({
                'Feature': list(metrics['feature_importance'].keys()),
                'Importance': list(metrics['feature_importance'].values())
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
            ax.set_title("Top 10 Features for Predicting Churn")
            st.pyplot(fig)
            
            # Save model option
            if st.button("💾 Save Model"):
                # Save the model to a file
                model_filename = "churn_prediction_model.joblib"
                joblib.dump(model, model_filename)
                
                # Provide download link
                with open(model_filename, "rb") as f:
                    model_bytes = f.read()
                    
                st.download_button(
                    label="📥 Download Model",
                    data=model_bytes,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )
                
                # Clean up
                if os.path.exists(model_filename):
                    os.remove(model_filename)
        
        except Exception as e:
            st.error(f"An error occurred during model training: {str(e)}")
            st.info("Try reducing the sample size or number of trees.")

# Data visualizations
st.markdown("---")
st.subheader("📊 Data Insights & Visualization")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Churn distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    churn_counts = df['churned'].value_counts()
    ax = sns.countplot(x='churned', data=df, palette='coolwarm')
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn (0 = No, 1 = Yes)")
    
    # Add count labels
    for i, count in enumerate(churn_counts):
        ax.text(i, count + 5, str(count), ha='center')
    
    st.pyplot(fig)
    
    # Calculate churn rate
    churn_rate = df['churned'].mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.2f}%")

with viz_col2:
    # Satisfaction vs Churn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='churned', y='satisfaction_score', data=df, palette='coolwarm')
    ax.set_title("Customer Satisfaction vs Churn")
    ax.set_xlabel("Churned")
    ax.set_ylabel("Satisfaction Score (0-10)")
    st.pyplot(fig)

# Additional visualizations
st.subheader("📈 Additional Insights")

viz_col3, viz_col4 = st.columns(2)

with viz_col3:
    # Tenure vs Churn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='churned', y='tenure', data=df, palette='coolwarm')
    ax.set_title("Customer Tenure vs Churn")
    ax.set_xlabel("Churned")
    ax.set_ylabel("Tenure (Years)")
    st.pyplot(fig)

with viz_col4:
    # Service issues vs Churn
    fig, ax = plt.subplots(figsize=(10, 6))
    service_pivot = pd.crosstab(df['service_issues'], df['churned'], normalize='index') * 100
    service_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
    ax.set_title("Service Issues vs Churn Rate")
    ax.set_xlabel("Number of Service Issues")
    ax.set_ylabel("Percentage")
    ax.legend(title="Churned", labels=["No", "Yes"])
    st.pyplot(fig)

# Correlation heatmap
st.subheader("🔄 Feature Correlations")

# Get numerical columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[num_cols].corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title("Correlation Between Features")
st.pyplot(fig)

st.markdown("---")
st.success("✅ Dashboard Ready!")
st.info("This open-source project uses synthetic data to demonstrate churn prediction. The model uses RandomForest instead of TensorFlow for better performance.")
