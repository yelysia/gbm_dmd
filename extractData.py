import tkinter as tk
from tkinter import filedialog
import ctypes
import platform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- DPI FIX ---
if platform.system() == "Windows":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except AttributeError:
        ctypes.windll.user32.SetProcessDPIAware()

# Load the data
root = tk.Tk() # create a tkinter window
root.withdraw() # hide the tkinter window
# open a file dialog to select the csv file
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")]) # open a file dialog to select the csv file
root.destroy() # destroy the tkinter window
# check that file_path is in the form of "CRSP_<earliestdate>-<latestdate>_<portfolio name>_dailydata.csv"
print("file_path: ", file_path)
assert 'CRSP' in file_path, "Error: file_path does not contain 'CRSP'"
assert '-' in file_path and file_path.count('-') == 1, "Error: file_path does not contain exactly 1 '-'"
df = pd.read_csv(file_path) # read the csv file into a pandas dataframe
# extract earliest and latest date from df
earliest_date = df['date'].min()
latest_date = df['date'].max()
print("earliest_date (t=0): ", earliest_date, "latest_date (t=T_tot-1): ", latest_date)
portfolio_name = file_path.split('.csv')[0].split('_')[2] # extract the portfolio name from the file path
# Determine dimensions
# 'unique' returns PERMNOs in order of appearance
N = len(df['PERMNO'].unique())  
print("rows in df: ", len(df)) 
assert len(df) % N == 0, "Error: the number of rows in the dataframe is not divisible by the number of unique PERMNOs"
T_tot = len(df) // N  
print(f"Detected N={N} stocks and T_tot={T_tot} trading days. (number of total rows in csv file / N = T_tot = {T_tot})")

# Load the raw original header information (rows sorted by PERMNO) (file name begins with "CRSP_")
headerinfodf = pd.read_csv(f"{file_path.split('CRSP_')[0]}CRSP_{portfolio_name}_headerinfo.csv")
# ensure both the dataframes 'permno' columns are exactly the same
assert df['PERMNO'].unique().tolist() == headerinfodf['permno'].tolist(), "Error: the unique 'PERMNO's in the df in order of appearance is not in the same order as the PERMNOs in headerinfodf dataframe"

# get the unique tickers in the headerinfodf dataframe
unique_tickers = headerinfodf['HTICK'].unique()
# check there are exactly N unique tickers
assert len(unique_tickers) == N, "Error: the number of unique tickers is not equal to the number of unique PERMNOs"
assert len(unique_tickers) == len(headerinfodf['HTICK'].values), "Error: the number of unique tickers is not equal to the number of tickers in headerinfodf dataframe"

# calculate market capitalization for each stock 
# extract the PRC, CFACPR, SHROUT, and CFACSHR values in every row that has date=earliest_date
prc_earliest_date = df[df['date'] == earliest_date]['PRC'].values
cfacpr_earliest_date = df[df['date'] == earliest_date]['CFACPR'].values
shrout_earliest_date = df[df['date'] == earliest_date]['SHROUT'].values
cfacshr_earliest_date = df[df['date'] == earliest_date]['CFACSHR'].values

# calclulate each stock's market capitalization on date earliest_date
mktcaps_day0 = np.abs(prc_earliest_date) / cfacpr_earliest_date * shrout_earliest_date * cfacshr_earliest_date
# ensure that the length of mktcaps_day0 is N
assert len(mktcaps_day0) == N, "Error: the length of mktcaps_day0 is not equal to the number of unique PERMNOs"
# ensure that the elements of mktcaps_day0 are all positive
assert all(mktcaps_day0 > 0), "Error: the elements of mktcaps_day0 are not all positive"

# add mktcaps_day0 to the headerinfodf dataframe
headerinfodf['MKTCAPDAY0'] = mktcaps_day0

# Create the matrix of returns snapshots Y
# We extract the 'RET' column (.values here returns a 1D numpy array) and reshape it.
# .reshape(N, T) fills row-by-row, which matches the file structure perfectly.
#print(df['RET'].values.shape) # this is 1D numpy array of length N * T_tot
Y = df['RET'].values.reshape(N, T_tot)
# check Y is a (N, T_tot) matrix
assert Y.shape == (N, T_tot), "Error: Y is not a (N, T_tot) matrix"
# Y[i, t] is the return of the i-th stock (as ordered by PERMNO in the original dataframe) on day t where t=0 is 2024/12/31.

# ensure Y[i, 0] is the earliest recorded return of the stock whose PERMNO is the ith unique PERMNO in the original df
for i in range(N):
    assert Y[i, 0] == df[df['PERMNO'] == df['PERMNO'].unique()[i]].iloc[0]['RET'], "Error: Y[i, 0] is not the earliest recorded return of the stock whose PERMNO is the ith unique PERMNO in the original df"

# load the headerinfodf dataframe that was manually sorted by NAICS then by market cap in excel (blocks of same NAICS, then in each block, stocks are sorted by decreasing MKTCAPDAY0)
sorted_headerinfodf = pd.read_csv(f'{portfolio_name}_headerinfo_withMKTCAPDAY0_withIndices_sortedByNAICS_sortedByMKTCAPDAY0.csv')

# reorder the rows of Y according to the indices in sorted_headerinfodf['index']
Y = Y[sorted_headerinfodf['index'].values]

# reorder the tickers according to the indices in sorted_headerinfodf['index']
unique_tickers_sorted_by_NAICS_MKTCAP = unique_tickers[sorted_headerinfodf['index'].values]
# # save unique tickers sorted by market cap then by NAICS to a txt file
# with open(f'{portfolio_name}_tickers_sortedByNAICS_sortedByMKTCAPDAY0.txt', 'w') as f:
#     for ticker in unique_tickers_sorted_by_NAICS_MKTCAP:
#         f.write(f'{ticker}\n')

# reorder the PERMNOs according to the indices in sorted_headerinfodf['index']
unique_PERMNOs_sorted_by_NAICS_MKTCAP = df['PERMNO'].unique()[sorted_headerinfodf['index'].values]

# ensure Y[i, 0] is the earliest recorded return of the stock whose PERMNO is the ith PERMNO in the sorted PERMNOs
for i in range(N):
    assert Y[i, 0] == df[df['PERMNO'] == unique_PERMNOs_sorted_by_NAICS_MKTCAP[i]].iloc[0]['RET'], "Error: Y[i, 0] is not the earliest recorded return of the stock whose PERMNO is the ith PERMNO in the sorted PERMNOs"
# save Y 
np.save(f'{file_path.split('.csv')[0]}_returnsSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy', Y)

# plot returns snapshots over time
fig, ax = plt.subplots()
ax.plot(Y.T)
ax.set_title(f'{portfolio_name} {earliest_date} -- {latest_date}')
ax.set_xlabel('Trading Day')
ax.set_ylabel('Return')
plt.show()
# save plot to file
fig.savefig(f'figures/{file_path.split('.csv')[0]}_returnsSnapshots.png', dpi=300)

# create the matrix of adjusted prices snapshots P
# adjusted price at time t is |PRC_t| / CFACPR_t
prc_adjusted = np.abs(df['PRC'].values) / df['CFACPR'].values
# ensure that the dimensions of prc_adjusted is (N * T_tot,)
assert prc_adjusted.shape == (N * T_tot,), "Error: the dimensions of prc_adjusted are not (N * T_tot,)"
# reshape prc_adjusted to a (N, T_tot) matrix
P = prc_adjusted.reshape(N, T_tot)
# ensure P[i, 0] is the adjusted price of the stock whose PERMNO is the ith unique PERMNO in the original df on the earliest date
for i in range(N):
    assert P[i, 0] == df[df['PERMNO'] == df['PERMNO'].unique()[i]].iloc[0]['PRC'] / df[df['PERMNO'] == df['PERMNO'].unique()[i]].iloc[0]['CFACPR'], "Error: P[i, 0] is not the adjusted price of the stock whose PERMNO is the ith unique PERMNO in the original df on the earliest date"
# reorder the rows of P according to the indices in sorted_headerinfodf['index']
P = P[sorted_headerinfodf['index'].values] # sorted into blocks of NAICS then in each block sorted by decreasing market cap
# ensure P[i, 0] is the adjusted price of the stock whose PERMNO is the ith PERMNO in the sorted PERMNOs
for i in range(N):
    assert P[i, 0] == df[df['PERMNO'] == unique_PERMNOs_sorted_by_NAICS_MKTCAP[i]].iloc[0]['PRC'] / df[df['PERMNO'] == unique_PERMNOs_sorted_by_NAICS_MKTCAP[i]].iloc[0]['CFACPR'], "Error: P[i, 0] is not the adjusted price of the stock whose PERMNO is the ith PERMNO in the sorted PERMNOs"
# save P
np.save(f'{file_path.split('.csv')[0]}_adjPricesSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy', P)

# make a temporary copy of unsorted P
P2 = prc_adjusted.reshape(N, T_tot)
# get prices of all stocks at earliest_date
prc_adj_on_day0 = P2[:, 0]
# ensure shape of prc_adj_on_day0 is (N,)
assert prc_adj_on_day0.shape == (N,), "Error: shape of prc_adj_on_day0 is not (N,)"
# sort the rows of P2 according to increasing adj price on day0
P2 = P2[np.argsort(prc_adj_on_day0)]

# add prc_adj_on_day0 to the headerinfodf dataframe
headerinfodf['ADJPRCDAY0'] = prc_adj_on_day0
# save the headerinfodf dataframe to a csv file
headerinfodf.to_csv(f'{portfolio_name}_headerinfo_withMKTCAPDAY0_withADJPRCDAY0_sortedByPERMNO.csv', index=False)

# open txt file of unique tickers sorted by ADJPRCDAY0, and store the tickers in a list
with open(f'{portfolio_name}_tickers_sortedByADJPRCDAY0.txt', 'r') as f:
    tickers_sorted_by_ADJPRCDAY0 = [line.strip() for line in f.readlines()]

# plot prices snapshots over time
fig, ax = plt.subplots()
ax.plot(P.T, label=tickers_sorted_by_ADJPRCDAY0)
ax.set_title(f'{portfolio_name} {earliest_date} -- {latest_date}')
ax.set_xlabel('Trading Day')
ax.set_ylabel('(Adjusted) Price ($)')
plt.show()
# save plot to file
fig.savefig(f'figures/{file_path.split('.csv')[0]}_adjPricesSnapshots.png', dpi=300)

# create a new matrix of natural logarithms of prices snapshots
logP = np.log(P)
# save logP
np.save(f'{file_path.split('.csv')[0]}_logAdjPricesSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy', logP)

# plot natural logarithm of prices snapshots over time
fig, ax = plt.subplots()
ax.plot(logP.T, label=tickers_sorted_by_ADJPRCDAY0)
ax.set_title(f'{portfolio_name} {earliest_date} -- {latest_date}')
ax.set_xlabel('Trading Day')
ax.set_ylabel('ln[(Adjusted) Price]')
plt.show()
# save plot to file
fig.savefig(f'figures/{file_path.split('.csv')[0]}_logAdjPricesSnapshots.png', dpi=300)
