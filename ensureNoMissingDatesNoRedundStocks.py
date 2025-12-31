import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import platform

# --- DPI FIX ---
if platform.system() == "Windows":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except AttributeError:
        ctypes.windll.user32.SetProcessDPIAware()


# 1. Load the data
root = tk.Tk() # create a tkinter window
root.withdraw() # hide the tkinter window
# open a file dialog to select the csv file
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")]) # open a file dialog to select the csv file
root.destroy() # destroy the tkinter window
#file_path = 'CRSP_20141231-20241231_Dow30Citigroup_lbjozrlulvd4o56w.csv'
df = pd.read_csv(file_path) # read the csv file into a pandas dataframe
# extract earliest and latest date from df
earliest_date = df['date'].min()
latest_date = df['date'].max()
print("earliest_date (t=0): ", earliest_date, "latest_date (t=T_tot-1): ", latest_date)
portfolio_name = file_path.split('.csv')[0].split('_')[2] # extract the portfolio name from the file path

# 2. Confirm that there are no missing dates

# count number of unique dates in the dataframe
unique_dates = df['date'].unique()
print("number of unique dates: ", len(unique_dates))

# convert the date column to a datetime object
df['date'] = pd.to_datetime(df['date'])
# get years into a new column in the dataframe
df['year'] = df['date'].dt.year
# count the number of trading days per year assuming that there are no missing dates for any PERMNO
trading_days_per_year = df['year'].value_counts(dropna=False).sort_index() / len(df['PERMNO'].unique())
# print the trading days per year
print("trading days per year: ", trading_days_per_year)
# check the assumption's validity i.e. that the elements of trading_days_per_year are all integers
if not all(isinstance(days, int) for days in trading_days_per_year):
    print("Error: the elements of trading_days_per_year are not all integers")

# count the number of appearances of each unique PERMNO in the dataframe
permnos_appearances = df['PERMNO'].value_counts(ascending=True, dropna=False)
print("permnos appearances: ", permnos_appearances)


# ensure no redunant stocks. these are securities identified by different PERMNOs but have the same PERMCO
# count the number of unique PERMCOs and PERMNOs in the dataframe
print("number of unique PERMNOs: ", len(df['PERMNO'].unique()))
print("number of unique PERMCOs: ", len(df['PERMCO'].unique()))
# count the number of appearances of each unique PERMNO in the dataframe
permcos_appearances = df['PERMCO'].value_counts(ascending=True, dropna=False)
print("permcos appearances: ", permcos_appearances)


# print number of unique TICKERs in the dataframe
print("number of unique TICKERs: ", len(df['TICKER'].unique()))
# # print number of unique TSYMBOLs in the dataframe
# print("number of unique TSYMBOLs: ", len(df['TSYMBOL'].unique()))

# count number of appearances of each TICKER in the dataframe
tickers_appearances = df['TICKER'].value_counts(ascending=True, dropna=False)
print("tickers appearances: ", tickers_appearances)
# # count number of appearances of each TSYMBOL in the dataframe
# tsymbols_appearances = df['TSYMBOL'].value_counts(ascending=True, dropna=False)
# print("tsymbols appearances: ", tsymbols_appearances)

# # save the unique permnos to a text file
# with open(f'{portfolio_name}_uniquePERMNOs.txt', 'w') as f:
#     for permno in df['PERMNO'].unique():
#         f.write(f'{permno}\n')


# check if the 'PRC' column has any null entires
#print("number of null entries in the 'PRC' column: ", df['PRC'].isnull().sum())