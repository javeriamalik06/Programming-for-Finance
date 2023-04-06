# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:41:59 2023

@author: Javeria Malik
"""

#GRAHAM VALUE

#importing math library to perform mathematical functions needed to calculate graham value
import math
import pandas as pd

#importing financial statements
balance_sheet = pd.DataFrame(pd.read_excel("balance_sheet_javeria.xlsx",index_col=0))
income_statement = pd.DataFrame(pd.read_excel("income_statement_javeria.xlsx", index_col=0))
cashflow_statement = pd.DataFrame(pd.read_excel("cf_fscore.xlsx",index_col=0))
enterprise = pd.DataFrame(pd.read_excel("enterprise.xlsx",index_col=0))
#creating a dataframe with index assigned to financial statement reporting dates
years = balance_sheet.columns

#calculating earnings per share
eps = income_statement[years[28]]['Net Income'] / enterprise[years[28]]['Common Shares']
#calculating book value per share of the stock
bvps = balance_sheet[years[28]]['Total assets'] / enterprise[years[28]]['Common Shares']
#calculating graham number
graham_number = math.sqrt(22.5 * eps * bvps)

######################################################################################################################################################
