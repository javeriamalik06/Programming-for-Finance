# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:38:40 2023

@author: Javeria Malik
"""

#CASHFLOW FORECASTING

#TO FORECAST THE BALANCE SHEET AND INCOME STATEMENT INITIALLY

#importing relevant libraries
import pandas as pd
from finstmt import FinancialStatements, IncomeStatements, BalanceSheets
from prophet import Prophet

#uploading balance sheet and income statement
balance_sheet = BalanceSheets.from_df(pd.read_excel(r'balance_sheet_javeria.xlsx', index_col=0))
income_statement = IncomeStatements.from_df(pd.read_excel(r'income_statement_javeria.xlsx', index_col=0))
statements = FinancialStatements(income_statement, balance_sheet)

statements2 = statements.copy()

#setting forecast methods and models in Jupyter
statements.config.update_all(['forecast_config', 'method'], 'auto')
statements.forecast_assumptions

#forecasting for 3 years ahead
forecast = statements.forecast(periods=12, freq='3M')

forecast

import matplotlib.pyplot
forecast.plot()


#CHANGES in forecast methods for individual components of balance sheet and income statement
statements.config.update('sga', ['forecast_config', 'pct_of'], 'revenue')
statements.forecast_assumptions

statements.config.update('int_exp', ['forecast_config', 'method'], 'recent')
statements.forecast_assumptions

statements.config.update_all(['forecast_config', 'method'], 'auto')
statements.forecast_assumptions


#PREPARING TO EXPORT INCOME STATEMENT AND BALANCE SHEET

#Adding historical statements to df
income_statement = income_statement.to_df()
balance_sheet = balance_sheet.to_df()

# Collecting Forecasted statements in place
forecast_income_statement = pd.DataFrame(forecast.income_statements.to_df())
forecast_balance_sheet = pd.DataFrame(forecast.balance_sheets.to_df())

#Merging historical and forecasted statements to form final income statement and balance sheet
final_income_statement = income_statement.merge(forecast_income_statement, right_index=True, left_index=True)
final_balance_sheet = balance_sheet.merge(forecast_balance_sheet, right_index=True, left_index=True)

#exporting balance sheet and income statement to excel so cashflow forecasting can be done
final_income_statement.to_excel('is.xlsx')
final_balance_sheet.to_excel('bs.xlsx')

#CASHFLOW FORECASTING DONE MANUALLY USING THE EXPORTED STATEMENTS ABOVE in excel sheet 'cf.xlsx'

#####################################################################################################################################################################################
