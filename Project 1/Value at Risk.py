# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:44:47 2023

@author: Javeria Malik
"""

#VALUE AT RISK

#importing libraries needed to calculate var and cVar for BPL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from psx import stocks, tickers
import math
from scipy.stats import norm
import datetime as datetime
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 600


#Using Historical Approach to calculate VaR and CVaR

#importing stock price of BPL from psx
stock_prices= stocks("BPL", start=datetime.date(2000, 1, 1), end=datetime.date.today())
df=stock_prices[['Close']]
#setting inventory of stock 
inv = [100000]
#setting confidence interval of 99%
conf = 0.99

#calculating simple stocks returns
#converting the dataframe into a array through .to_numpy()
stock_returns = df.pct_change().dropna().to_numpy()

#calculating inverse inventory values through transpose function
inv = np.array(inv).T

#porfolio returns
mult = pd.DataFrame(np.matmul(stock_returns, inv))

#Sorting values in ascending order and resetting the index
pnl = pd.DataFrame(mult).sort_values(by=0, ascending = True).reset_index(drop = True)

#setting the tail (alpha) for Var. It is the value that corresponds.
tail = 1 - conf

#math.floor to round the index to an integer
# .loc to extract the value of the index
#calculating var value
var_ha = float(pnl.loc[math.floor(tail * len(pnl)),])

criteria = pnl.apply(lambda x: x).values <= var_ha

#calculating cVar
cvar_ha =float(pnl.loc[criteria].mean())

#plotting var cvar and pnl data 
sns.histplot(data = pnl)
plt.axvline(x=var_ha, color='blue')
plt.axvline(x=cvar_ha, color='red')

##############################################################################################################
