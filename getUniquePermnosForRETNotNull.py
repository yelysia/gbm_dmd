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
#df = pd.read_csv('CRSP_20171231-20231231_612TickersGSheet_jn8zfe2zverlpxj4.csv') # read the csv file into a pandas dataframe
df = pd.read_csv(file_path) # read the csv file into a pandas dataframe
# extract earliest and latest date from df
earliest_date = df['date'].min()
latest_date = df['date'].max()
print("earliest_date (t=0): ", earliest_date, "latest_date (t=T_tot-1): ", latest_date)
portfolio_name = file_path.split('.csv')[0].split('_')[2] # extract the portfolio name from the file path


# save the unique permnos to a txt file
unique_permnos = df['PERMNO'].unique().tolist()
# check that the type of the elements are integers
if not all(isinstance(permno, int) for permno in unique_permnos):
    print("Error: the elements of the unique_permnos list are not integers")
    exit()
with open(f'{portfolio_name}_uniquePERMNOs.txt', 'w') as f:
    for permno in unique_permnos:
        f.write(f'{permno}\n')
# print the number of unique tickers and permnos in the dataframe
print("number of unique tickers: ", len(df['TICKER'].unique()))
print("number of unique permnos: ", len(unique_permnos))

# save the unique permnos for which the RET column is not null
with open(f'{portfolio_name}_PERMNOsForRETNull.txt', 'r') as f:
    # read the file into a list
    permnos_for_RETNull = f.readlines()
    # convert the list to a set of integers
    permnos_for_RETNull_set = set([int(permno) for permno in permnos_for_RETNull])
    # convert the unique_permnos list to a set
    unique_permnos_set = set(unique_permnos)
    # subtract the permnos_for_RETNull from the unique_permnos
    unique_permnos_for_RETNotNull_set = unique_permnos_set - permnos_for_RETNull_set
    # convert the set to a list
    unique_permnos_for_RETNotNull = list(unique_permnos_for_RETNotNull_set)

# assert that the length of the unique_permnos_for_RETNotNull is equal to the length of the unique_permnos minus the length of the permnos_for_RETNull
print("length of unique_permnos_for_RETNotNull: ", len(unique_permnos_for_RETNotNull_set))
print("length of unique_permnos: ", len(unique_permnos_set))
print("length of permnos_for_RETNull: ", len(permnos_for_RETNull_set))
if len(unique_permnos_for_RETNotNull) != len(unique_permnos_set) - len(permnos_for_RETNull_set):
    print("Error: the length of the unique_permnos_for_RETNotNull is not equal to the length of the unique_permnos minus the length of the permnos_for_RETNull")
    exit()


# save the unique permnos for which the RET column is not null to a txt file
with open(f'{portfolio_name}_uniquePERMNOsForRETNotNull.txt', 'w') as f:
    for permno in unique_permnos_for_RETNotNull:
        f.write(f'{permno}\n')