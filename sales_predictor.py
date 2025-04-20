# sales_predictor.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("monthly_sales.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['sales_diff'] = df['sales'].diff()
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
test_sales = df['sales'].values[-13:]

for i in range(len(predicted_diff)):
    value = predicted_diff[i] + test_sales[i]
    predicted_sales.append(value)

# Evaluation
rmse = np.sqrt(mean_squared_error(test_sales[1:], predicted_sales))
mae = mean_absolute_error(test_sales[1:], predicted_sales)
r2 = r2_score(test_sales[1:], predicted_sales)

print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R² Score:", round(r2, 2))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df.index[-13:], test_sales, label='Actual Sales')
plt.plot(df.index[-12:], predicted_sales, label='Predicted Sales', color='red')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
