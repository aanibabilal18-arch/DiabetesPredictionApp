# 🩺 Diabetes Prediction System

An end-to-end machine learning project that turns diabetes data into an interactive prediction system using **K-Nearest Neighbors (KNN) classification**.

### 🚀 Live Demo

[Try the Diabetes Prediction System](https://diabetespredictionapp-c7nyfrevvex9ho3g9z7fyn.streamlit.app/)

### 🎯 Why KNN?

KNN was selected because diabetes prediction is a **classification problem**, and patient records can be compared based on the similarity of their health measurements. Since KNN is distance-based, the features were **preprocessed and standardized using StandardScaler** before training.

Rather than focusing only on a prediction, this project demonstrates the complete workflow from **data preparation → model training → evaluation → deployment**.

### 📊 Result

**KNN Accuracy: 71%**

The deployed application provides a predicted class along with the model's estimated probabilities for both outcomes.

### 🛠️ Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Streamlit`

### 📁 Core Files

- `app.py` — Interactive prediction interface
- `diabetes.csv` — Dataset
- `knn_model.pkl` — Trained KNN classifier
- `scaler.pkl` — Fitted feature scaler
- `requirements.txt` — Dependencies

### ⚠️ Disclaimer

This project is for educational purposes only and is not intended to provide medical diagnosis or professional medical advice.
