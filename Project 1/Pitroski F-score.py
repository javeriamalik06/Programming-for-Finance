# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:40:25 2023

@author: Javeria Malik
"""

#PITROSKI F-SCORE

import pandas as pd
# importing financial statements
balance_sheet = pd.DataFrame(pd.read_excel("balance_sheet_javeria.xlsx",index_col=0))
income_statement = pd.DataFrame(pd.read_excel("income_statement_javeria.xlsx", index_col=0))
cashflow_statement = pd.DataFrame(pd.read_excel("cf_fscore.xlsx",index_col=0))
enterprise = pd.DataFrame(pd.read_excel("enterprise.xlsx",index_col=0))

#creating a dataframe with index assigned to financial statement reporting dates
years = balance_sheet.columns

#Making a function for calculating f_score

    
f_score=[]
   
for i in range(len(years)-1,0,-1):
    #the scores are contributing to f score
       roa = income_statement[years[i]]['Net Income'] / balance_sheet[years[i]]['Total assets']
       #score is earned if return on assets is greater than zero
       s1 = 1 if roa > 0 else 0
       
       cfo = cashflow_statement[years[i]]['Cashflow from Operations'] / balance_sheet[years[i-1]]['Total assets']
       #score if the cashflow from operations is greater than zero
       s2 = 1 if cfo > 0 else 0
       
       
       delta_roa = (income_statement[years[i]]['Net Income'] / balance_sheet[years[i]]['Total assets']) - (income_statement[years[i-1]]['Net Income'] / balance_sheet[years[i-1]]['Total assets'])
       #score earned if change in roa is greater than zero
       s3 = 1 if delta_roa > 0 else 0
       
       
       accrual = (cashflow_statement[years[i]]['Cashflow from Operations'] - income_statement[years[i]]['Net Income']) / balance_sheet[years[i]]['Total assets']
       #score earned if accrual is greater than zero
       s4 = 1 if accrual > 0 else 0
       
       
       
   # Leverage, liquidity and source of funds
       
       #LEVERAGE
       delta_lever = (balance_sheet[years[i]]['Long-term debt'] / ((balance_sheet[years[i]]['Total assets'] + balance_sheet[years[i-1]]['Total assets']) / 2)) - (balance_sheet[years[-2]]['Long-term debt'] / ((balance_sheet[years[i-1]]['Total assets'] + balance_sheet[years[i-2]]['Total assets']) / 2))
       #leverage to be less than 0 for score
       s5 = 1 if delta_lever < 0 else 0
       
       #LIQUIDITY
       #Liquidity to be greater than 0 for score       
       delta_liq = (balance_sheet[years[i]]['Total current assets'] / balance_sheet[years[i]]['Total current liabilities']) - (balance_sheet[years[i-1]]['Total current assets'] / balance_sheet[years[i-2]]['Total current liabilities'])
       s6 = 1 if delta_liq > 0 else 0
       
       #source of funds
       #stocks offered to be zero or less than zero for score
       eq_offer = enterprise[years[i]]['Common Shares'] - enterprise[years[i-1]]['Common Shares']
       s7 = 1 if eq_offer <= 0 else 0
       
       
       
       # Operating efficiency
   #calculating the increase or decrease in gp margins, increase results in a score
       delta_margin = (income_statement[years[i]]['Gross Profit'] / income_statement[years[i]]['Revenue']) - (income_statement[years[i-1]]['Gross Profit'] / income_statement[years[i-1]]['Revenue'])
       s8 = 1 if delta_margin > 0 else 0
       
       #score if the delta turn is greater than zero
       delta_turn = (income_statement[years[i]]['Revenue'] / balance_sheet[years[i]]['Total assets']) - (income_statement[years[i-1]]['Revenue'] / balance_sheet[years[i-1]]['Total assets'])
       s9 = 1 if delta_turn > 0 else 0
       
       #all the individual scores add up to give a final score out of 9
       profitability_score = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9
       #adds the f_scores of every quarter to the f_score dictionary formed initially
       f_score.append(profitability_score)

   print(f_score)

###########################################################################################################
