import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv("SPX.csv", parse_dates=["Date"], index_col="Date")
##training_data = data.loc["2016":"2018", "Close"]
training_data = data.loc["2010":"2015", "Close"]
##training_data = data.loc["1990":"2005", "Close"]
training_data = training_data.asfreq('B').ffill()

##actual_data= data.loc["2019":"2020", "Close"]
actual_data= data.loc["2018":"2020", "Close"]
##actual_data= data.loc["2015":"2020", "Close"]


'''
def make_stationary(series, sig_level = 0.05):
    difcount = 0
    while True:
        adf = adfuller(series)
        p = adf[1]

        if p < sig_level:
            print(f"Data is stationary after {difcount} differencing steps.")
            break

        else:
            series = series.diff().dropna()
            difcount += 1
    return series

training_data = make_stationary(training_data)
'''

p, d, q = 2, 2, 2 ##change these as needed 

model = ARIMA(training_data, order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

forecast_steps = 3 * 252 ##change as needed, 252 = business days in 1 yr
forecast = model_fit.forecast(steps=forecast_steps)
forecast_dates = pd.date_range(start="2018-01-01", periods=forecast_steps, freq="B") ##also change based on starting date of predictions
forecast_series = pd.Series(forecast, index=forecast_dates)

mae = (forecast_series[:len(actual_data)] - actual_data).abs().mean()
rmse = np.sqrt(((forecast_series[:len(actual_data)] - actual_data) ** 2).mean())

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

print(forecast_series.head())
print(forecast_series.tail())

'''
plt.figure(figsize=(12, 6))
plt.plot(training_data, label="Training Data (2016-2018)")
plt.plot(forecast_series, label="Forecast (2019-2020)", linestyle="--")
plt.plot(actual_data, label="Actual Data (2019-2020)", linestyle="-", color="orange")
plt.legend()
plt.title("ARIMA Model - 1-Year Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
'''

plt.figure(figsize=(12, 6))
plt.plot(training_data, label="Training Data (2010-2015)")
plt.plot(forecast_series, label="Forecast (2018-2020)", linestyle="--")
plt.plot(actual_data, label="Actual Data (2018-2020)", linestyle="-", color="orange")
plt.legend()
plt.title("ARIMA Model - 3-Year Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

'''
plt.figure(figsize=(12, 6))
plt.plot(training_data, label="Training Data (1990-2015)")
plt.plot(forecast_series, label="Forecast (2015-2020)", linestyle="--")
plt.plot(actual_data, label="Actual Data (2015-2020)", linestyle="-", color="orange")
plt.legend()
plt.title("ARIMA Model - 3-Year Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
'''