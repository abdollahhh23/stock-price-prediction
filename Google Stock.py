import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load your stock dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\User\Downloads\googl_us_d.csv")  

# Make sure Date is datetime + sorted
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# -------------------------------
# 2. Create lag and rolling features
# -------------------------------
df['Close_Lag1'] = df['Close'].shift(1)                 
df['Close_MA7'] = df['Close'].rolling(window=7).mean()  
df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()  

# Drop NaNs created by shift/rolling
df = df.dropna()

# -------------------------------
# 3. Define features and target
# -------------------------------
X = df[['Close_Lag1', 'Close_MA7', 'Volume_MA7']]
y = df['Close']

# -------------------------------
# 4. Train-test split (time-series safe: no shuffle)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# 5. Train Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print({
    'test_score (R2)': r2,
    'RMSE': rmse
})

# -------------------------------
# 7. Forecast Next 5 Days
# -------------------------------
forecast = []
last_row = df.iloc[-1].copy()

for i in range(5):
    X_future = [[
        last_row['Close_Lag1'],
        last_row['Close_MA7'],
        last_row['Volume_MA7']
    ]]
    
    next_close = model.predict(X_future)[0]
    forecast.append(next_close)
    
    # Update features for next iteration
    last_row['Close_Lag1'] = next_close
    last_row['Close_MA7'] = (last_row['Close_MA7']*6 + next_close) / 7

print("Forecast (next 5 days):", forecast)

# -------------------------------
# Focused plot on test + forecast
# -------------------------------

plt.figure(figsize=(12,6))

# Plot actual vs predicted (test set only)
plt.plot(df['Date'].iloc[-len(y_test):], y_test, label="Actual (Test)", color="blue")
plt.plot(df['Date'].iloc[-len(y_test):], y_pred, label="Predicted (Test)", color="red")

# Plot forecast (next 5 days, extending after last test date)
future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=5)
plt.scatter(future_dates, forecast, color="green", label="Forecast (Next 5 Days)", s=70)

plt.title("Model Performance on Test Data + Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
