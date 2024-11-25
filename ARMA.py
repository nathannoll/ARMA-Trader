import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("C:/Users/Joel Carrasco/OneDrive/PRML/SPX.csv")

# Convert 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index('Date', inplace=True)

# Select the 'Close' column (assuming you're predicting the 'Close' price)
data = data[['Close']]

# Handle missing values by forward filling (or use a different imputation method)
data.fillna(method='ffill', inplace=True)

# Define the training and testing periods
train = data['2019-12-25':'2019-12-31']  # Last 90 days of 2019
test = data['2020-01-01':'2020-01-04']  # First 60 days of 2020

# Fit the ARIMA model with p=3, d=0, q=2
model = ARIMA(train, order=(3, 0, 2))
model_fit = model.fit()

# Make predictions for the test set
predictions = model_fit.forecast(steps=len(test))

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(test, predictions)

# Output the MSE
print(f'Mean Squared Error: {mse}')

# Plot the results (optional, turn off for now if you want no plots)
import matplotlib.pyplot as plt

plt.plot(test.index, test['Close'], label='True Values')
plt.plot(test.index, predictions, label='Predicted Values')
plt.title('ARIMA Model: Predictions vs True Values')
plt.legend()
plt.show()
