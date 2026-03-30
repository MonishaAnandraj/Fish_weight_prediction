# 🐟 Fish Weight Prediction App

A Machine Learning-powered web application that predicts the **weight of a fish** based on its physical measurements like length, height, width, and girth.

Built using **Streamlit** for an interactive UI and a trained regression model for prediction.

---

## 📌 Project Overview

This project demonstrates a complete ML workflow:

* 📊 Data preprocessing & feature engineering
* 🤖 Model training using regression
* 🌐 Deployment using Streamlit

Users can input fish measurements and instantly get the predicted weight.

---

## ✨ Features

* 🐟 Predict fish weight in real-time
* 🎨 Attractive UI with background image & styling
* 📥 User-friendly input fields
* ⚙️ Feature engineering (volume, log transformations, polynomial features)
* 🚀 Ready for deployment on Streamlit Cloud

---

## 🧠 Machine Learning Details

* Model Used: Linear Regression
* Feature Engineering:

  * Volume proxy
  * Log transformation
  * Polynomial features (length², length³)
  * Interaction features
* Encoding:

  * One-hot encoding for fish species

---

## 📂 Project Structure

```bash
📁 Fish-Weight-Prediction
│
├── app.py                  # Streamlit application
├── model.pkl              # Trained ML model
├── model_columns.pkl      # Feature columns used in training
├── fish-weight-prediction.ipynb  # Jupyter notebook
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/fish-weight-prediction.git
cd fish-weight-prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

You can try the live version of this Food Bill Predictor here:  
[🍔 Fish Weight Predictor - Streamlit App](https://fishweightprediction-jnbepk3punkngglhzjmplq.streamlit.app/) 



