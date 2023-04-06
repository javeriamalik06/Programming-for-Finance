# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:15:12 2023

@author: Javeria Malik
"""

#OPTIMAL PORTFOLIO

#runs perfectly in Jupyter Notebook
#importing libraries to import the stock data for BPL PKGS and PPP
from psx import stocks, tickers
import riskfolio as rp
import matplotlib.pyplot as plt
import warnings
import datetime
import pandas as pd
warnings.filterwarnings("ignore")


# In[4]:


ticker_list = ['BPL', 'PKGS', 'PPP']
#importing stock prices Close prices from psx for BPL from 31st March 2010 to 25th December 2022
stock_prices_BPL = stocks('BPL', start=datetime.date(2010, 3, 31), end=datetime.date(2022, 12, 25))['Close']
#converting the panda series into a dataframe
stock_prices_BPL = stock_prices_BPL.to_frame()
#renaming the close column to BPL
stock_prices_BPL.rename(columns = {'Close':'BPL'}, inplace = True)

#same as above but for the PKGS stock
stock_prices_PKGS = stocks('PKGS', start=datetime.date(2010, 3, 31), end=datetime.date(2022, 12, 25))['Close']
stock_prices_PKGS = stock_prices_PKGS.to_frame()
stock_prices_PKGS.rename(columns = {'Close':'PKGS'}, inplace = True)

#same as above for the PPP stock
stock_prices_PPP = stocks('PPP', start=datetime.date(2010, 3, 31), end=datetime.date(2022, 12, 25))['Close']
stock_prices_PPP = stock_prices_PPP.to_frame()
stock_prices_PPP.rename(columns = {'Close':'PPP'}, inplace = True)

#merged the stock datas of the individual stocks above into one dataframe for portfolio using pd.concat
stock_prices = pd.concat([stock_prices_BPL , stock_prices_PKGS, stock_prices_PPP],axis=1).dropna()


# In[5]:


#calculating stock returns using the pct.change formula on stock prices
stock_returns = stock_prices.pct_change().dropna()


# In[6]:


#specific variables we need for riskfolios
#first 3 dont need to be changed
method_mu = 'hist'
method_cov = 'hist'
model = 'Classic'
#the below can be changed (riskfolio site)
#risk model, mv is standard deviation is used for the portfolio
rm = 'MV'
#objective can be to min risk, max ulitily (for risk aversion), sharpe etc.
obj = 'Sharpe'
rf = 0
#risk aversion coefficient
l = 0 
hist = True
#how many porfolios needed
points = 50


# In[7]:


#xreating Portfolio from the riskfolio library
port = rp.Portfolio(returns = stock_returns)
port.assets_stats(method_mu = method_mu, method_cov = method_cov)
#optimizing the portfolio and storing the optimal weights it as w
w = port.optimization(model = model, rm = rm, obj = obj, rf = rf, hist = hist)


# In[8]:


w


# In[9]:


#creating a pie chart for the weights in the optimal portfolio above
ax = rp.plot_pie(w = w, title = "Optimum Portfolio", others = 0.05, cmap = 'tab20')


# In[10]:


#calculating the set of efficient portfolios for the stocks and their risk model
frontier = port.efficient_frontier(model = model, rm = rm, points = points, hist = hist)
#plotting the efficient portfolios on the graph with the x axis being the risk and y axis being the returns
ax = rp.plot_frontier(w_frontier = frontier, mu = port.mu, cov = port.cov, returns = stock_returns, rm = rm, rf = rf, cmap = 'viridis', w = w)


# In[11]:


frontier


# In[12]:


#showing the efficient portfolios' assets structure ie the weights of stocks for all 50 portfolios
#50th portfolio is max risk max return - PKGS in this case
ax = rp.plot_frontier_area(w_frontier = frontier, cmap = 'tab20')


# In[13]:


#generating a jupyter report on the portfolio
ax = rp.jupyter_report(returns = stock_returns, w = w, rm = rm)


# In[14]:


#setting different risk models for portfolio
rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']

w_s = pd.DataFrame([])

for i in rms:
    w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
    w_s = pd.concat([w_s, w], axis=1)
    
w_s.columns = rms


# In[15]:


#displaying weights of stocks in the optimal portfolio for every risk model
w_s.style.format("{:.2%}").background_gradient(cmap='YlGn')


# In[16]:


#plotting the bar chart for the weights above for every stock in different risk models
fig = plt.gcf()
fig.set_figwidth(14)
fig.set_figheight(6)
ax = fig.subplots(nrows=1, ncols=1)

w_s.plot.bar(ax=ax)
