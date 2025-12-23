# 🌦 Weather Prediction Using Time Series Analysis

This project focuses on predicting weather conditions using historical time series data. Multiple statistical, machine learning, and deep learning models are implemented and compared to identify the most accurate approach for weather forecasting.

---

## 📌 Project Overview

Weather prediction is essential for agriculture, transportation, disaster management, and daily planning. Traditional forecasting methods often struggle with non-linear patterns and seasonal variations. This project applies time series analysis and machine learning techniques to improve forecasting accuracy.

Models used in this project:
- SARIMA
- Linear Regression
- LSTM
- Support Vector Regression (SVR)

---

## 🎯 Objectives

- Analyze historical weather data using time series techniques
- Implement statistical, machine learning, and deep learning models
- Compare model performance using numerical metrics
- Visualize predictions and errors using graphs
- Identify the best-performing weather prediction model

---

## 🗂 Dataset Description

- **Dataset Name:** DailyDelhiClimate.csv
- **Source:** Kaggle
- **Location:** Delhi, India
- **Time Period:** 2013 – 2017

### Features Used
- Mean Temperature
- Humidity
- Wind Speed
- Mean Pressure

**Note : This Dataset contains some incorrect reading ("meanpressure")and outliers**

---

## 🧠 Models Used

### SARIMA
- Seasonal extension of ARIMA
- Captures trend and seasonality in time series data

### Linear Regression
- Baseline statistical model
- Assumes linear relationship between past and future values

### LSTM (Long Short-Term Memory)
- Deep learning model for time series
- Learns long-term dependencies and non-linear patterns

### Support Vector Regression (SVR)
- Kernel-based regression technique
- Performs well on non-linear data

---

## 🔬 Methodology

1. Data collection and cleaning
2. Exploratory Data Analysis (EDA)
3. Feature engineering and normalization
4. Train-test data splitting
5. Model training and prediction
6. Performance evaluation using error metrics
7. Graphical and numerical comparison

---

## 📐 Evaluation Metrics

- **Mean Absolute Error (MAE)**  
  `MAE = (1/n) Σ |yᵢ − ŷᵢ|`

- **Root Mean Square Error (RMSE)**  
  `RMSE = √[(1/n) Σ (yᵢ − ŷᵢ)²]`

- **R² Score**  
  `R² = 1 − (SS_res / SS_tot)`

---

## 📊 Results Summary

| Model             | MAE  | RMSE | R² Score |
|-------------------|------|------|----------|
| SARIMA            | Low  | Moderate | Poor |
| Linear Regression | 1.09 | 1.39 | 0.95 |
| LSTM              | 1.53 | 1.92 | 0.91 |
| Tuned SVR         | **1.10** | **1.39** | **0.95** |

**Best Performing Model:** Support Vector Regression (SVR)

---

## 📈 Visualization

- Actual vs Predicted line plots
- MAE and RMSE comparison bar charts
- R² score comparison graphs

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Statsmodels

---

## ▶ How to Run the Project

1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow statsmodels
