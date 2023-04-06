# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:43:54 2023

@author: Javeria Malik
"""

#PROBABILITY TO DEFAULT

#importing libraries needed to calculate probability of default
from psx import stocks, tickers
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import nasdaqdatalink as ndl
import math
from scipy.stats import norm 
from scipy.optimize import minimize
import datetime


#Advanced controls
sigma_a0ini = 0.1
mult_lower_a0ini = 0.001 
mult_upper_a0ini = 10 
lower_sigma_a = 0.001
upper_sigma_a = 4

all_results = {'year': [], 'd2d': [], 'p2d':[]}

#importing data from balance sheet
balance_sheet = pd.DataFrame(pd.read_excel("bs_BPL_p2d.xlsx",index_col=0))

#importing data of the BPL stock
s1 = stocks('BPL', start=datetime.date(2000, 3, 31), end=datetime.date(2022, 3, 31))
#importing riskfree rate from pakistan ten year bonds
riskfree_df = pd.read_excel(r'PK 10YB.xlsx')
#importing the enterprise value containing the number of shares and resetting the index
enterprise = pd.read_excel(r'enterprise.xlsx')
enterprise = enterprise.set_index('Unnamed: 0')
    
from datetime import datetime
#setting up the ending date for data
end_date = datetime.strptime(balance_sheet.loc['fillingDate'][0], '%Y-%m-%d').date()

#setting up the starting date to be 1 year ago from the end date
start_date = end_date - timedelta(365)

#changing the date format to string format
end_date = str(end_date)
start_date = str(start_date)

#setting prices to be the closing stock price of BPL
prices = s1['Close']

#converting the prices into log-returns
ret = np.log(prices).diff(periods = 1).iloc[1:]

#daily volatility * sqrt (252) = annualized volatility of equity
sigma_e = ret.std() * math.sqrt(252) 

#creating dataframe for payables, current debt, short term debt and long term debt
payables = balance_sheet.loc['accountPayables',][0]
current_debt = np.nan_to_num(balance_sheet.loc['shortTermDebt',][0], nan = 0)
short_term_debt = payables + current_debt
long_term_debt = balance_sheet.loc['longTermDebt',][0]

#setting call option
d = short_term_debt + 0.5 * long_term_debt
years = balance_sheet.loc['fillingDate']
# try and expect block to avoid error and set an alternative to error
try:
    i = 0.0455
except:
    end_date = datetime.strptime(balance_sheet.loc['fillingDate'][0], '%Y-%m-%d').date()
    end_date = end_date - timedelta(1)
    end_date = str(end_date)
    i = float(ndl.get(riskfree_df, start_date=end_date, end_date=end_date).loc[:,'1 YR'])/100

#setting n as number of common shares
n = enterprise.loc['Common Shares'][0]

#converting prices from panda series into dataframe
prices=prices.to_frame()
#calculating p0 as latest price
p0 = prices['Close'][-1]

#equity at the time 0 = S0 (call option)
e0 = n * p0

t = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days / 365

# calculating the initial value of assets to be equal to debt and equity at time 0
a0ini = e0 + d

# trying to minimize  a0 = at time t (in the future)
def func(par, e0, sigma_e, i, t, d):
        a0 = par[0]
        sigma_a = par[1]
        d1 = (math.log(a0/d) + (i + sigma_a ** 2/ 2) * t)/(sigma_a * math.sqrt(t))
        d2 = d1 - sigma_a * math.sqrt(t)
        return ((e0 - a0 * norm.cdf(d1) + math.exp(-i * t) * d * norm.cdf(d2)) ** 2 + (sigma_e * e0 - norm.cdf(d1) * sigma_a * a0) ** 2)                  
    
bounds = ((a0ini * mult_lower_a0ini, a0ini * mult_upper_a0ini),(lower_sigma_a, upper_sigma_a))
result = minimize(fun = func, x0 = (a0ini, sigma_a0ini), method = "L-BFGS-B", args = (e0, sigma_e, i, t, d), bounds = bounds)
    
#finding a0 = which is the value of assets in the futrure
a0 = result.x[0]
#finding volatility of assets
sigma_a = result.x[1]
    
d1 = (math.log(a0/d) + (i + sigma_a ** 2/ 2) * t)/(sigma_a * math.sqrt(t))
d2 = d1 - sigma_a * math.sqrt(t)    
prob = norm.cdf(-d2)

#distance to default (in days)
print(round(d2, 2))

#probability to default
print(round(prob, 10))

###############################################################################################################################################################3
