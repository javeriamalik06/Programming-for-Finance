# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:14:34 2023

@author: Javeria Malik
"""

#COPULAS

#importing needed libraries for copulas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import math
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 600
from statsmodels.distributions.empirical_distribution import ECDF
from copulae import pseudo_obs
from copulae.elliptical import GaussianCopula
from psx import stocks, tickers
import datetime

#forming a single dataframe with stock information of all 3 stocks in the portfolio
stock_prices = pd.DataFrame(stocks("PPP", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
#stock_returns = stock_prices.pct_change().dropna().to_numpy()
stock_prices.rename(columns={ 'Close': 'PPP_price'}, inplace=True)
stock_prices_1 = pd.DataFrame(stocks("PKGS", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
stock_prices_1.rename(columns={ 'Close': 'PKGS_price'}, inplace=True)
stock_prices_2 = pd.DataFrame(stocks("BPL", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
stock_prices_2.rename(columns={ 'Close': 'BPL_price'}, inplace=True)
stock_prices_3 = stock_prices.merge(stock_prices_1, right_index = True, left_index = True)
stock_prices_4 = stock_prices_3.merge(stock_prices_2, right_index = True, left_index = True)

#stock_returns = stock_prices_4.pct_change().dropna().to_numpy()

stock_returns = np.log(stock_prices_4).diff(periods = 1).iloc[1:]

stock_returns.describe()

# Pseudo observations are a percentile rank of a given obs expressed as a number between 0 and 1
u = pseudo_obs(stock_returns)

# Compute empirical cumulative distribution function, ECDF
# An approximation of the cdf function
# Justifies how our observations are betweem 0 and 1 
f1 = ECDF(stock_returns['PPP_price'])
f2 = ECDF(stock_returns['PKGS_price'])
f3 = ECDF(stock_returns['BPL_price'])

# Instantizing the copula package
model = GaussianCopula(dim = 3)

# Fitting the data to the copula
results = model.fit(data = u)

# Find the probability that we are in the tail of the bivariate distribution
# As we are talking about credit risk
# The joint probability that the log-returns will go below some benchmark at same time
# These are the -ve *daily log returns* for which we want to calculate jprob 
x1 = -0.01
x2 = -0.01
x3 = -0.01

# Apply previously created functions of f1, f2, f3 to convert left-tail log-returns into pseudo observations
u0 = np.asarray([f1(x1), f2(x2), f3(x3)])

# Joint-probability of getting a negative return on all stocks in 1-day
results.pobs(u0)

#Out[21]: array([0.25, 0.5 , 0.75])

##############################################################################################################################
