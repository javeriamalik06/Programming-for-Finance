# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:37:41 2023

@author: Javeria Malik
"""
#STOCK FORECASTING

import talib as ta
import matplotlib.pyplot as plt
#importing library for downloading stock price from PyPI
from psx import stocks, tickers
tickers = tickers()
import datetime
df = stocks('BPL', start=datetime.date(2009, 3, 31), end=datetime.date(2022, 3, 31))

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 300
#importing the stock prices of BPL stock


from prophet import Prophet
from statsmodels.tsa.seasonal import STL

#Plotting library but with better functionalitiies
import seaborn as sns

# Added to make your plots look better
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)

df = df.resample('M').last().dropna()

df.reset_index(drop=False, inplace=True)

# .rename(...)
# Giving our dataframe new column names
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)


train_indices = df.ds.apply(lambda x: x.year).values < 2022

#values for training data
df_train = df.loc[train_indices].dropna()

#values from testing data
#swiggle~ means everything that is false
df_test = df.loc[~train_indices].dropna()

#Prophet(...)
# (1) what kind of decomposition to use
#using the most suitable seasonality mode, in this case additive
model_prophet = Prophet(seasonality_mode='multiplicative')

#.fit() calculates the coefficients on training data
model_prophet.fit(df_train)

#future
df_future = model_prophet.make_future_dataframe(periods = 36, freq = 'M', include_history = 'True')

df_pred = model_prophet.predict(df_future)

model_prophet.plot(df_pred)
model_prophet.plot_components(df_pred)

visualize_test_data = df_test.merge(df_pred[:-1]['yhat'], right_index = True, left_index = True)

print("MAE")
g= sns.jointplot(y='y', x='yhat', data=visualize_test_data, kind='reg', scatter = True, color = 'blue')
sns.scatterplot(y='y', x='yhat', data=visualize_test_data, ax=g.ax_joint, color = 'red')
print(np.mean(np.abs(visualize_test_data['yhat'].values - visualize_test_data['y'].values)))

#the mean absolute error for my data is 1.4662 which means the mean absolute difference between the predictive and actual stock values of BPL Companies is 1.4662.
#if we compare the error to the latest stock price for BPL (19.26 rupees), the error is only 7.61% of the stock price, which is not very significant but still impacts the predicted values of the stock in the model and also questions the efficiency of the model.

######################################################################################################################################################################################
