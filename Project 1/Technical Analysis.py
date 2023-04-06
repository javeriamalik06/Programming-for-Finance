# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:42:10 2023

@author: Javeria Malik
"""

#TECHNICAL ANALYSIS

#5.1
import talib as ta
import matplotlib.pyplot as plt
#importing library for downloading stock price from PyPI
from psx import stocks, tickers
tickers = tickers()
import datetime
#downloading BPL stock data from psx
data = stocks('BPL', start=datetime.date(2009, 3, 31), end=datetime.date(2022, 3, 31))

#SMA(SIMPLE MOVING AVERAGES)
# 1: Closing prices, 2: no. of days etc
# new column formed in data for sma 50 and sma 100 200 etc
data['SMA_50'] = ta.SMA(data['Close'], 50)
data['SMA_100'] = ta.SMA(data['Close'], 100)
data['SMA_200'] = ta.SMA(data['Close'], 200) 
# when short SMA is greater than long SMA then we buy. for alternative we sell
#Plotting SMA
# 1: data we want to plot, 2: label for the data
plt.plot(data['Close'], label = 'Close')
plt.plot(data['SMA_50'], label = 'SMA_50')
plt.plot(data['SMA_100'], label = 'SMA_100')
plt.plot(data['SMA_200'], label = 'SMA_200')
plt.legend()
#giving the plot a title
plt.title('SMA')

#EMA
data['EMA_50'] = ta.EMA(data['Close'], 50)
data['EMA_100'] = ta.EMA(data['Close'], 100)
data['EMA_200'] = ta.EMA(data['Close'], 200)
plt.plot(data['Close'], label = 'Close')
plt.plot(data['EMA_50'], label = 'EMA_50')
plt.plot(data['EMA_100'], label = 'EMA_100')
plt.plot(data['EMA_200'], label = 'EMA_200')
plt.legend()
plt.title('EMA')


#RSI 
# 1: Closing price
# 2: time period over which the strength index should be calculated
data['RSI'] = ta.RSI(data['Close'], 14)
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios":[3, 1]}, figsize=(14, 8))
#Uper wala plot will be closing prices
axs[0].plot(data['Close'])
#plot RSI
axs[1].plot(data['RSI'], color = 'orange')
axs[1].axhline(y = 70, color = "r", linestyle = "--")
axs[1].axhline(y = 30, color = "g", linestyle = "--")
plt.title('RSI')


#MACD
data['macd'], data['macd_signal'], data['macd_hist'] = ta.MACD(data['Close'])
c = ["red" if c1 < 0 else "green" for c1 in data['macd_hist']]
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios":[3, 1]}, figsize=(14, 8))
axs[0].plot(data['Close'], color = 'blue')
axs[1].plot(data['macd'], 'b-')
axs[1].plot(data['macd_signal'], color = 'orange')
axs[1].bar(data['macd_hist'].index, data['macd_hist'], color = c)
plt.title('macd')


#ATR
data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], 14)
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios":[3, 1]}, figsize=(14, 8))
axs[0].plot(data['Close'], color = 'blue')
axs[1].bar(data['ATR'].index, data['ATR'], color = 'green')
plt.title('ATR')

#UPPERBAND
data['UpperBand'], data['MiddleBand'], data['LowerBand'] = ta.BBANDS(data['Close'], 20, nbdevup = 2, nbdevdn = 2)
plt.plot(data['Close'], label = 'Close')
plt.plot(data['UpperBand'], label = 'Upper', linestyle = '--')
plt.plot(data['MiddleBand'],label= 'Middle',linestyle= '--') 
plt.plot(data['LowerBand'], label = 'Lower', linestyle = '--')
plt.title('UpperBand')