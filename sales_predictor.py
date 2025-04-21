import pandas as pd

# Load dataset from the web
url = "Month_Value_1.csv"
df = pd.read_csv(url)

# Convert Month to datetime and rename columns
df = df[['Period', 'Sales_quantity']]
df.columns = ['Period', 'Sales_quantity']
df['Period'] = pd.to_datetime(df['Period'],format='%d.%m.%Y')
# Show the dataset
print(df.head())
# sales_predictor.py


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset

df['Period'] = pd.to_datetime(df['Period'])
df.set_index('Period', inplace=True)
df['sales_diff'] = df['Sales_quantity'].diff()
df.dropna(inplace=True)

# Create supervised learning format
def create_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df_supervised = pd.concat(columns, axis=1)
    df_supervised.fillna(0, inplace=True)
    return df_supervised

data_supervised = create_supervised(df['sales_diff'], 12)

data_values = data_supervised.values
train_size = len(data_values) - 12
train, test = data_values[:train_size], data_values[train_size:]
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
predicted_diff = model.predict(X_test_scaled)

# Reconstruct actual sales

predicted_sales = []
test_sales = df['Sales_quantity'].values[-(len(predicted_diff)+1):]

for i in range(len(predicted_diff)):
    value = predicted_diff[i] + test_sales[i]
    predicted_sales.append(value)

# Ensure same length before evaluation
actual = test_sales[1:]
predicted = predicted_sales

print("Actual:", actual)
print("Predicted:", predicted)

print("Lengths → Actual:", len(actual), "Predicted:", len(predicted))

# Evaluation (only if lengths match)
if len(actual) == len(predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print("Root Mean Squared Error (RMSE):", round(rmse, 2))
    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("R² Score:", round(r2, 2))
else:
    print("❌ Cannot evaluate: Actual and Predicted lengths do not match.")
