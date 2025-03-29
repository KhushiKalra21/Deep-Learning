# Customer Churn Prediction Dashboard  

## **Live Streamlit Dashboard**  
[Customer Churn Prediction Dashboard](https://deep-learning-mzmchluyqbaoxvetuhzmma.streamlit.app/)  

## **Project Overview**  
This project focuses on **customer churn prediction** using an **Artificial Neural Network (ANN)**. The model is trained to classify customers as **likely to churn or stay retained**, based on key customer attributes such as engagement, purchase behavior, and loyalty program participation.  

The model has been trained on a dataset containing over **500,000 customer records**, with features covering **purchase frequency, membership years, social media engagement, and customer support interactions**. The goal is to provide businesses with actionable insights to reduce churn and improve customer retention strategies.  

## **Features**  
- Predicts whether a customer will churn based on historical data.  
- Interactive **Streamlit dashboard** with real-time predictions.  
- Hyperparameter tuning options for ANN optimization.  
- Visualizations for **model performance** and **customer trends**.  
- Supports large datasets (50,000 records at a time).  
- Business-oriented insights to improve customer retention.  

## **How It Works**  
1. Users upload or analyze customer data through the **Streamlit dashboard**.  
2. The system processes customer attributes and applies the **trained ANN model**.  
3. The model predicts the probability of churn for each customer.  
4. Results are displayed with confidence scores, trends, and actionable insights.  

## **Model & Dataset Details**  
- **Training Data:** Real-world customer churn dataset with over **500,000 records**.  
- **Testing Data:** A random subset of 50,000 customers is used for evaluation.  
- **Model Architecture:** ANN with **three fully connected layers** using **ReLU and Sigmoid activations**.  
- **Hyperparameter Controls:**  
  - Learning rate, batch size, and optimizer selection.  
  - Activation functions (**ReLU, Sigmoid, Tanh, Softmax**).  
  - Gradient descent variations (**Batch, Mini-Batch, Stochastic**).  

## **Setup & Installation**  
To run the project locally, follow these steps:  

```bash
# Clone the repository
git clone <your-repository-link>
cd <your-project-folder>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
