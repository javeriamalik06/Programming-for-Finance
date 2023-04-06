# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:12:17 2023

@author: Javeria Malik
"""

#GROUP MEMBERS:
    #Javeria Malik - 19292
    #Urooba Izhar - 19409
    #Furqan Umer Khan - 19481
#STOCKS: PKGS, PPP, BPL

PORTFOLIO VAR

from psx import stocks, tickers
#importing date and time 
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import math
from scipy.stats import norm
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 600


# Historical Approach VaR and CVaR

# Historical Approach VaR and CVaR

#ticker_list = ["PPP"]
#inv = [500000]
#conf = 0.99
#We are importing the stcok prices for every stock individually
stock_prices = pd.DataFrame(stocks("PPP", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
#stock_returns = stock_prices.pct_change().dropna().to_numpy()
stock_prices.rename(columns={ 'Close': 'PPP_price'}, inplace=True)
stock_prices_1 = pd.DataFrame(stocks("PKGS", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
stock_prices_1.rename(columns={ 'Close': 'PKGS_price'}, inplace=True)
stock_prices_2 = pd.DataFrame(stocks("BPL", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
stock_prices_2.rename(columns={ 'Close': 'BPL_price'}, inplace=True)

#Merging the stock prices into one data frame
stock_prices_3 = stock_prices.merge(stock_prices_1, right_index = True, left_index = True)

stock_prices_4 = stock_prices_3.merge(stock_prices_2, right_index = True, left_index = True)
#Calculating the stock returns
stock_returns = stock_prices_4.pct_change().dropna().to_numpy()

#weightage = 30.% , 50% and 0%. 
#the weightage is taken according to technical analysis
#The investment in terms of money in each stock
inv = [300000 , 500000 , 200000]

#this is the confidence interval 
conf = 0.99

inv = np.array(inv).T
mult = pd.DataFrame(np.matmul(stock_returns, inv))

#sorting values in ascending order
#resetting the index
pnl = pd.DataFrame(mult.sum(axis = 1)).sort_values(by=0, ascending = True).reset_index(drop = True)

#setting the tail
tail = 1 - conf
var_ha = float(pnl.loc[math.floor(tail * len(pnl)),])
#output = -51666.468
#with optimal allocation: 53124

criteria = pnl.apply(lambda x: x).values <= var_ha
cvar_ha =float(pnl.loc[criteria].mean())
#output= - 77320.08
#with optimal allocation 84628.41

sns.histplot(data = pnl)
plt.axvline(x=var_ha, color='blue')
plt.axvline(x=cvar_ha, color='red')

##########################################################
# Model-Building Approach VaR and CVaR (Parametric/Variance-Covariance)


from psx import stocks, tickers
#importing date and time 
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import math
from scipy.stats import norm
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 600

#dividing the investment according to weights of stocks in the portfolio
inv = [300000, 500000, 200000]
#setting confidence level at 99%
conf = 0.99

stock_prices = pd.DataFrame(stocks("PPP", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
#stock_returns = stock_prices.pct_change().dropna().to_numpy()
stock_prices.rename(columns={ 'Close': 'PPP_price'}, inplace=True)
stock_prices_1 = pd.DataFrame(stocks("PKGS", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
stock_prices_1.rename(columns={ 'Close': 'PKGS_price'}, inplace=True)
stock_prices_2 = pd.DataFrame(stocks("BPL", start= datetime.date(2010,3,31), end = datetime.date(2022,12,25))['Close'])
stock_prices_2.rename(columns={ 'Close': 'BPL_price'}, inplace=True)

stock_prices_3 = stock_prices.merge(stock_prices_1, right_index = True, left_index = True)

stock_prices_4 = stock_prices_3.merge(stock_prices_2, right_index = True, left_index = True)

stock_returns = stock_prices_4.pct_change().dropna()

#Calculating the variance-covariance matrix
#We are doing this ex to calculate the variance of portfolio returns
varcovar_matrix = stock_returns.cov().to_numpy()

#transposing the investment
inv = np.array(inv).T

#calculating variance between the stock returns and Investment 
variance = np.matmul(np.matmul(inv.T, varcovar_matrix), inv)
math.sqrt(variance)
#Settimg the tail
tail = 1 - conf

#extracting the value associated to the tail from a standardized normal distribution
#multiplying it with the std deviation in $ terms of portfolio returns
var_mba = norm.ppf(q = tail) * math.sqrt(variance)
#var_mba=-48271.583346766245

#Same as above but now for cvar
cvar_mba = -math.sqrt(variance) * math.exp(-norm.ppf(q=tail) ** 2 / 2) / (math.sqrt(2 * math.pi) * tail)
#cvar=-55303.0403621185

mult = pd.DataFrame(np.matmul(stock_returns, inv))
pnl = pd.DataFrame(mult.sum(axis = 1)).sort_values(by=0, ascending = True).reset_index(drop = True)
sns.histplot(data = pnl)
plt.axvline(x=var_mba, color='blue')
plt.axvline(x=cvar_mba, color='red')


############################################################
# Monte-Carlo Approach VaR and CVaR

from psx import stocks, tickers
#importing date and time 
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import math
from scipy.stats import norm
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 600



#number of iterations
its = 1000000
#dividing the investment according to weights of stocks in the portfolio
inv = [300000, 500000, 200000]
#setting a confidence level of 99%
conf = 0.99

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

#calculing stock returns using the pct change function
stock_returns = stock_prices_4.pct_change().dropna()

varcovar_matrix = stock_returns.cov().to_numpy()

random_stock_returns = np.random.multivariate_normal([0,0,0], varcovar_matrix, its)

mult = pd.DataFrame(np.matmul(random_stock_returns, inv))

pnl = pd.DataFrame(mult.sum(axis = 1)).sort_values(by=0, ascending = True).reset_index(drop = True)

tail = 1 - conf
var_mca = float(pnl.loc[math.floor(tail * len(pnl)),])
#var_mca=-48303.22779877498

criteria = pnl.apply(lambda x: x).values <= var_mca 
cvar_mca =float(pnl.loc[criteria].mean())
#cvar_mca=-55339.59076291749

sns.histplot(data = pnl)
plt.axvline(x=var_mca, color='blue')
plt.axvline(x=cvar_mca, color='red')

#################################################################

# Forecasted Loss Approach VaR and CVaR


from psx import stocks, tickers
#importing date and time 
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import math
from scipy.stats import norm
plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 600

#Number of iterations
its = 1000000
##dividing the investment according to weights of stocks in the portfolio
inv = [300000, 500000, 200000]
T = 1 # Forecast horizon is 1-day
conf = 0.99

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

#calculing stock returns using the pct change function
stock_returns = stock_prices_4.pct_change().dropna()

stock_returns.corr()

cov_mat = stock_returns.cov()

rv = np.random.normal(size=(its, 3))
correlated_rv = np.transpose(np.matmul(cov_mat, np.transpose(rv)))

r = np.mean(stock_returns, axis = 0).values
sigma = np.std(stock_returns, axis=0).values * math.sqrt(252)
s_0 = stock_prices_4.values[-1, :]

shares = []
shares.append(inv/s_0)
shares = np.trunc(shares).astype(int)
p_0 = np.sum(shares * s_0)

s_T = s_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * correlated_rv)

p_T = np.sum(shares * s_T, axis=1)
pnl = pd.DataFrame(p_T - p_0).sort_values(by=0, ascending = True).reset_index(drop = True)

tail = 1 - conf
var_fla = float(pnl.loc[math.floor(tail * len(pnl)),])
#var_fla= -112025.57646501705

criteria = pnl.apply(lambda x: x).values <= var_fla 
cvar_fla =float(pnl.loc[criteria].mean())
#cvar_fla= -112131.61157697783

sns.histplot(data = pnl)
plt.axvline(x=var_fla, color='blue')
plt.axvline(x=cvar_fla, color='red')

##########################################################################################################################################
