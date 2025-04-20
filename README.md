# Sales-Predictor
# Sales Forecasting Using Linear Regression

## Project Overview
This project focuses on forecasting monthly sales using a supervised learning approach with a linear regression model. By transforming a time series dataset into a supervised learning format, we train a regression model to predict future sales based on past trends.

---

##  How It Works

### 1. Loading & Preparing Data
- Load the `monthly_sales.csv` dataset.
- Parse the `date` column as datetime and set it as the DataFrame index.
- Compute `sales_diff`, the month-to-month difference in sales.

### 2. Creating Supervised Learning Format
- Use a custom `create_supervised()` function with a lag of 12 months.
- This creates feature sets of the past 12 months to predict the current month’s sales difference.

### 3. Train-Test Split
- Use the last 12 months as the test set.
- Scale features using `MinMaxScaler` for normalization.

### 4. Model Training
- Train a **Linear Regression** model on the training set.
- Predict the test set values.

### 5. Reconstruct Actual Sales
- Since the model predicts sales differences, we add predictions to the previous actual sales to get final predicted sales values.

### 6. Evaluation
- Evaluate the model using:
  - **Root Mean Squared Error (RMSE)**: `219.56`
  - **Mean Absolute Error (MAE)**: `170.32`
  - **R² Score**: `0.89`

### 7. Visualization
- A line plot is generated to compare **Actual Sales** vs **Predicted Sales** for the test period (last 13 months).
- This plot provides visual insight into how well the model is performing.

---

## Key Features
- Converts time series data to supervised learning format using lag features.
- Trains a regression model to predict future values.
- Scales input features for improved model accuracy.
- Evaluates performance using industry-standard metrics.
- Visualizes actual vs predicted sales clearly.

---

## Concepts Applied
- Supervised Learning
- Regression Analysis
- Time Series Forecasting
- Feature Engineering with lag values
- Data Normalization

---

## Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**

---

## Possible Extensions
- Explore advanced models like **XGBoost** or **LSTM** for better accuracy.
- Add seasonal features such as month/quarter indicators.
- Deploy the model using Flask or FastAPI.
- Integrate live data input from an API or database.

---

## Dataset
Ensure the `monthly_sales.csv` file is present in your project directory. It should contain:
- A `date` column (monthly timestamps)
- A `sales` column (numeric sales values)

---

## Getting Started
1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`
3. Run the main script to train and evaluate the model

---

Source Code
Full implementation is available in [`sales_predictor.py`](sales_predictor.py)
