
# Handwritten & Typed Digit Recognition - README

## Live Streamlit App  
[Streamlit App Link](https://deep-learning-xrucdcrsre5nng4ysaw6n8.streamlit.app/)

#### REPORT can be viewed here : [REPORT](https://colab.research.google.com/github/KhushiKalra21/Deep-Learning/blob/main/CNN/kkpg2134_CNN_Report.ipynb)

---

## Project Overview  
This project focuses on digit classification using Convolutional Neural Networks (CNNs). Unlike traditional handwritten digit recognition systems, this model is trained on the MNIST dataset and tested on:

- Handwritten numbers collected from various individuals outside college premises.  
- Typed numbers generated in multiple fonts using MS Word.  

The goal is to improve digit classification accuracy across diverse writing styles and printed fonts, making it adaptable for real-world applications.

---

## Features  
- Handwritten and Typed Digit Recognition  
- Automatic Preprocessing (Grayscale, Thresholding, Resizing, Inversion Fix)  
- Confidence Score for Predictions  
- Supports Multiple Image Formats (JPG, PNG, JPEG)  
- Simple and Interactive Web Interface via Streamlit  

---

## How It Works  
1. Upload an image of a digit (handwritten or typed).  
2. The system processes the image to match MNIST-style input.  
3. The trained CNN model predicts the digit.  
4. The app displays the predicted number and confidence level.  

---

## Model & Dataset Details  
- **Training Data:** MNIST (70,000 grayscale handwritten digits).  
- **Testing Data:**  
  - Typed numbers (different fonts from MS Word).  
  - Handwritten numbers (collected from individuals).  
- **Model Architecture:** CNN with three convolutional layers, max pooling, and fully connected layers.  
- **Preprocessing Enhancements:** Adaptive thresholding, noise reduction, resizing, and inversion correction.  

---

