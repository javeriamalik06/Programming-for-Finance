# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:42:59 2023

@author: Javeria Malik
"""

#COST OF CAPITAL

#importing libraries and models for calculating WACC
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
#importing libraries to import the pakistani stocks data
from psx import stocks, tickers
tickers = tickers()
import datetime

#importing BPL stock data
data = stocks('BPL', start=datetime.date(1980, 3, 31), end=datetime.date(2022, 3, 31))['Close'].rename("stock")
#coverting data into log returns
s1 = np.log(data).diff(periods = 1).iloc[1:]

#importing KSE 100 market data
s2 = pd.read_excel(r'KSE 100.xlsx')
s2=s2.set_index('Date')
#renaming the column
s2 = s2['close'].rename("market")
s2 = np.log(s2).diff(periods = 1).iloc[1:]

#converting the panda series to panda dataframe
s1=s1.to_frame()
s2=s2.to_frame()

#removing duplicated values from the dataframes so they could be easily merged
s1.index.duplicated()
s2.index.duplicated()
s1=s1.loc[~s1.index.duplicated(), :]
s2=s2.loc[~s2.index.duplicated(), :]

#merging s1 and s2 into a single dataframe
df = pd.concat([s1 , s2],axis=1).dropna()

#rolling regression, calculating alpha beta again and again over different timeframes. precise. also find out if company specific risk(beta) is changing overtime
#pre req for running a rolling OLS
#model tells python to run regression with a constant
market = sm.add_constant(df.market, prepend = False)
#running OLS model for regression
model = RollingOLS(df.stock, market, window = 252)
#retrieving beta and alpha
rolling_params = model.fit().params.dropna()
#Constant is alpha and beta is market

#merging parameters with stock returns
df = rolling_params.merge(df.stock, right_index = True, left_index = True)
#plotting betas over time
sns.lineplot(data=df[['market']])

#importing balance sheet data
import pandas as pd
balance_sheet = pd.DataFrame(pd.read_excel("bs annual.xlsx",index_col=0))
income_statement = pd.DataFrame(pd.read_excel("is annual.xlsx", index_col=0))
#creating a dataframe with index assigned to financial statement reporting dates
years = balance_sheet.columns

#making a dataframe with beta
beta_df = pd.DataFrame(df.market)

#renaming the columns and resetting the indexes
beta_df.reset_index(drop=False, inplace=True)
beta_df.rename(columns={'Date': 'ds', 'market': 'y'}, inplace=True)

#renaming the columns and resetting the indexes
mkt_df = pd.DataFrame(s2)
mkt_df.reset_index(drop=False, inplace=True)
mkt_df.rename(columns={'Date': 'ds', 'market': 'y'}, inplace=True)

#creating a dataframe with index assigned to financial statement reporting dates
years = balance_sheet.columns


#importing 10-year risk free rates ie Pakistan 10 year bond yields
riskfree_df = pd.read_excel(r'PK 10YB.xlsx')
#renaming the columns
riskfree_df.rename(columns={'Date': 'ds', '10 YR': 'y'}, inplace=True)

#creating a dictionary to store wacc values
wacc_dict = {'year': [],'wacc': []}

#running a loop to calculate and store wacc values in the dictionary created above 

for i in range(0,5):
  # Historical beta

  #setting a criteria for betas for a beta of 5 years. 2017 above.
    criteria = beta_df.ds.apply(lambda x: x.year).values >= (int(years[0]) - 1)
    
    #Applying criteria
    five_year_beta = beta_df.loc[criteria]
    
    #Calcuting mean of Beta
    beta = mean(five_year_beta['y'])
    
    # Market return
    #setting a criteria for market returns for a period of 5 years
    criteria = mkt_df.ds.apply(lambda x: x.year).values >= (int(years[0]) - 1)
    
    #Applying criteria
    five_year_mkt = mkt_df.loc[criteria]
    
    #Calculating average market returns
    mkt = mean(five_year_mkt['y']) * 252
    

   # Risk-free rate
   #setting a criteria for riskfree rates for the desired 2022
    criteria = riskfree_df.ds.apply(lambda x: x.year).values == int(years[0])
    
    #Applying criteria
    one_year_riskfree = riskfree_df.loc[criteria]
    riskfree = mean(one_year_riskfree['y'])/100
        
    #Cost of equity
    ke = riskfree + beta * (mkt - riskfree)
            
    #Cost of debt
    try:
        kd = (income_statement[years[i]]['Interest Expense']*1000000 / (balance_sheet[years[i]]['shortTermDebt'] + balance_sheet[years[i]]['longTermDebt'])) * (1 - (income_statement[years[i]]['Income Tax']/income_statement[years[i]]['Pretax Income']))
    except:
        kd = 0
        
    #Calculating Weight of equity
    we = balance_sheet[years[i]]['Total Stockholders Equity'] / balance_sheet[years[i]]['Total Assets']
    
    #Calculating Weight of debt
    wd = (balance_sheet[years[i]]['Total Assets'] - balance_sheet[years[i]]['Total Stockholders Equity']) / balance_sheet[years[i]]['Total Assets']
        
    wacc = ke * we + kd * wd
    wacc_dict['year'].append(years[i])
    wacc_dict['wacc'].append(wacc)


wacc_df = pd.DataFrame(wacc_dict)
#if min is negative, then there is error in data
wacc_df.describe()
#plot. (right side x axis is current year)
sns.lineplot(data=wacc_df,x=wacc_df.year, y = wacc_df.wacc)
print(wacc_dict)


# Monte-carlo simulation  (to FIND future wacc)  

from fitter import Fitter, get_common_distributions, get_distributions
import seaborn as sns
import math
from scipy.stats import alpha, cauchy, chi2, expon, exponpow, gamma, lognorm, norm, powerlaw, rayleigh, uniform, beta, f, logistic as dist
from pylab import linspace, plot

bins = round(math.sqrt(len((wacc_df.wacc))))

#fitting different distributions on the data
f = Fitter(wacc_df.wacc, distributions=['t','triang','alpha','cauchy','chi2','expon','exponpow','gamma','lognorm','norm','powerlaw','rayleigh','uniform', 'beta', 'f', 'logistic','weibull_min', 'weibull_max', 'exponweib', 'genextreme', 'gumbel_r','skewnorm','gompertz'], bins = bins)
f.fit()
# Comparing between different distributions and discovering the best fit distribution
f.summary()

# Name of best fitting distribution
dist = next(iter(f.get_best(method = 'sumsquare_error')))

f.fitted_param[dist] # important check

# Parameters for distribution 
param_first = f.fitted_param[dist][1]
param_second = f.fitted_param[dist][2]

#EXPONPOW DISCOVERED TO BE THE MOST ACCURATELY FITTING DISTRIBUTION FOR WACC

#setting up iterations for number of values generated
its = 100000
#creating a dictionary for a set of wacc values
all_wacc = []

#generating random values according to the exponpow distribution and appending it to all_wacc dictionary
for i in range(1, its+1):
    #if distribution was normal, then write normal rather than cauchy
    x = (exponpow.rvs(2, loc = param_first, scale = param_second))
    if x <= 0:
        continue
    all_wacc.append(x)

#setting up number of bins for the histogram
bins = round(math.sqrt(len((all_wacc))))

#plotting the histogram
plt.hist(all_wacc, bins=bins, edgecolor="black")

#gives statistical distribution of the all_wacc dictionary
pd.DataFrame(all_wacc).describe()

#calculating 90th percentile of the wacc value for valueing
np.percentile(all_wacc, 90)


#########################################################################################################################################################
