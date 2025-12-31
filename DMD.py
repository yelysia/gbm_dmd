import tkinter as tk
from tkinter import filedialog
import ctypes
import platform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from optht import optht

from pydmd import DMD
from pydmd.plotter import plot_eigs, plot_summary
from pydmd.preprocessing import hankel_preprocessing

# --- DPI FIX ---
if platform.system() == "Windows":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except AttributeError:
        ctypes.windll.user32.SetProcessDPIAware()



def SVD_and_truncation_rank(X, savefig_filename):
    """
    Parameters:
    - X: all snapshots matrix (N, T_tot) numpy array
    - savefig_filename: str, filename to save the plot of singular values

    Returns:
    - U: (N, N) numpy array
    - S: (min(N, T_tot),) numpy array
    - VT: (T_tot, T_tot) numpy array
    - r: int, optimal truncation rank according to Gavish and Donoho 2014 https://arxiv.org/abs/1305.5870 assuming the X matrix is a rank r matrix plus gaussian noise
    - X_denosied_if_noised: (N, T_tot) numpy array, denoised X if noise is present
    """
    N, T_tot = X.shape
    U, S, VT = np.linalg.svd(X)
    # U and the Hermitian transpose of VT are 2D arrays with orthonormal columns and S is a 1D array of Y's singular values
    print("U.shape: ", U.shape)
    print("S.shape: ", S.shape)
    print("VT.shape: ", VT.shape)
    # when X has shape (N, T_tot), if full_matrices=True, U and VT have the shapes (N, N) and (T_tot, T_tot)
    # else if full_matrices=False, U and VT have the shapes (N, min(N, T_tot)) and (min(N, T_tot), T_tot)

    # plot the singular values of returns on a log scale
    snapshots_type = savefig_filename.split('_')[3]
    fig, ax = plt.subplots()
    ax.semilogy(S, 'o-')
    ax.set_title(f'Singular Values of {snapshots_type} Snapshots (N={N}, T_tot={T_tot})')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    fig.savefig('figures/'+savefig_filename, dpi=300)
    plt.close(fig)

    # optimal truncation by Gavish and Donoho 2014 https://arxiv.org/abs/1305.5870 assuming the Y matrix is a rank r matrix plus gaussian noise
    r = optht(Y, sv=S, sigma=None)
    print(r)
    X_denosied_if_noised = (U[:, range(r)] * S[range(r)]).dot(VT[range(r), :])
    # r is 93 if returns

    return U, S, VT, r, X_denosied_if_noised


# # optimal truncation by Gavish and Donoho 2014 https://arxiv.org/abs/1305.5870 assuming the Y matrix is a rank r matrix plus gaussian noise
# aspect_ratio = N / T_tot
# omega = 0.56*aspect_ratio**3 - 0.95*aspect_ratio**2 + 1.82*aspect_ratio + 1.43
# print("omega: ", omega)
# cutoff = omega * np.median(S)
# # get subarray of S where elements are larger than cutoff
# S_cutoff = S[S > cutoff]
# print("Number of singular values larger than cutoff: ", len(S_cutoff))
# # this gives 56  for returns

def train_test_split(X, train_percentage=0.95):
    '''
    split the time series into one training set and the rest test set
    
    X : (N, T_tot) numpy array

    returns:
        X_train : (N, T_train) numpy array
        X_test : (N, T_test) numpy array
        T_train : int
        T_test : int
    '''
    N, T_tot = X.shape
    T_train = int(train_percentage * T_tot)
    T_test = T_tot - T_train
    X_train = X[:, :T_train]
    X_test = X[:, T_train:]
    return X_train, X_test, T_train, T_test

# def train_test_windows(X, T_window):
#     '''
#     divide entire time series into equalsized windows (except the last window which may not be of equal size)
#     where each window alternates between being assigned as a training and test

#     X : (N, T_tot) numpy array
#     T_window : int, desired length of each window (the last window of testing may be shorter than T_window)

#     returns:
#         X_train_windows : list of (N, T_train) numpy arrays
#         X_test_windows : list of (N, T_test) numpy arrays
#         T_train_windows : (len:num_windows) list of ints (lengths of the corresponding training windows)
#         T_test_windows : (len:num_windows) list of ints (lengths of the corresponding test window)
#     '''
#     # LOGIC HERE IS NOT RIGHT; NEED TO FIX
#     N, T_tot = X.shape
#     num_full_windows = T_tot // T_window
#     last_additional_window_length = T_tot - num_full_windows * T_window
#     X_train_windows = []
#     X_test_windows = []
#     T_train_windows = []
#     T_test_windows = []
#     for i in range(num_full_windows-1):
#         X_train_windows.append(X[:, i*T_window:(i+1)*T_window])
#         X_test_windows.append(X[:, (i+1)*T_window:(i+2)*T_window])
#         T_train_windows.append(T_window)
#         T_test_windows.append(T_window)
#     # add the last guaranteed full window which would be a train set
#     X_train_windows.append(X[:, (num_full_windows-1)*T_window:])
#     T_train_windows.append(T_window)
#     # deal with remainder if nonzero, which would be the final test set
#     if last_additional_window_length != 0:
#         X_test_windows.append(X[:, -last_additional_window_length:])
#         T_test_windows.append(last_additional_window_length)

#     return X_train_windows, X_test_windows, T_train_windows, T_test_windows


######### manual dmd algorithm implementation ###############################

def DMD_manual(snapshots, svd_rank):
    """
    Xprime = A X 

    snapshots: 2D numpy array, shape (N, M), columns are snapshots at equal intervals
    svd_rank: int <= M-1, the number of singular values to keep in the truncated SVD of X

    Returns: 
        Abar: 2D numpy array, shape (N, N): the same-order approximation of matrix A.   
        Lambda:  (svd_rank, ) numpy array : the eigenvalues of matrix Abar
        DMD_modes:  (N, svd_rank) numpy array : the eigenvectors of matrix Abar        
    """        
    N, M = snapshots.shape
    if svd_rank >= M:
        raise ValueError(f"Error: svd_rank must be less than M-1, but svd_rank = {svd_rank} and M = {M}")

    X = snapshots[:, :-1]
    Xprime = snapshots[:, 1:]

    U, S, VT = np.linalg.svd(X, full_matrices=True)
    # U and the Hermitian transpose of VT are 2D arrays with orthonormal columns and S is a 1D array of Y's singular values
    # when X has shape (N, M-1), if full_matrices=True, U and VT have the shapes (N, N) and (M-1, M-1)


    # compute the truncated SVD of X: get the first r columns of U, the upperleft rxr block, the first r rows of VT
    U_trunc = U[:, :svd_rank]
    S_trunc = S[:svd_rank]
    VT_trunc = VT[:svd_rank, :]
    #turn S_trunc into a square matrix
    S_trunc = np.diag(S_trunc)
    # assert shapes of U_trunc, S_trunc, VT_trunc
    assert U_trunc.shape == (N, svd_rank), "Error: shape of U_trunc is not (N, svd_rank)"
    assert S_trunc.shape == (svd_rank, svd_rank), "Error: shape of S_trunc is not (svd_rank, svd_rank)"
    assert VT_trunc.shape == (svd_rank, M-1), "Error: shape of VT_trunc is not (svd_rank, M-1)"
    
    # Compute Abar as Xprime @ X_pseudoinverseapproximate
    Abar = Xprime @ VT_trunc.conj().T @ np.linalg.inv(S_trunc) @ U_trunc.conj().T

    ## efficient computation of dynamic modes:

    # transform every snapshot onto linear subspace of dimension r, using U_trunc.conj().T as a convenient transformation
    # yielding the reduced order model given by Atilde
    Atilde = U_trunc.conj().T @ Xprime @ VT_trunc.conj().T @ np.linalg.inv(S_trunc)

    # The eigenvectors of Abar are called the dynamic modes
    # For DMD, the eigenvalues of Atilde and Abar are equivalent, and the eigenvectors of Atilde and Abar are related via a linear transformation.
    
    # do eigendecomposition of Atilde
    # Atilde @ W = W @ Lambda; W is the matrix of eigenvectors and Lambda is the diagonal matrix of eigenvalues
    Lambda, W = np.linalg.eig(Atilde)
    # column W[:,i] is the eigenvector w corresponding to the eigenvalue Lambda[i].

    # check if there are complex values in Lambda
    if np.any(np.iscomplex(Lambda)):
        print("There are complex values in Lambda")
    
    if np.any(np.iscomplex(Abar)):
        print("There are complex values in Abar")
    
    # compute the dynamic modes
    DMD_modes = np.zeros((N, svd_rank))
    for i in range(len(Lambda)):
        w = np.reshape(W[:,i], (W[:,i].shape[0], 1))
        lamb = Lambda[i]
        if lamb != 0:
            phi = Xprime @ VT_trunc.conj().T @ np.linalg.inv(S_trunc) @ w
        else:
            phi = U_trunc @ w
        # assert phi has shape (N, 1)
        assert phi.shape == (N, 1), "Error: shape of phi is not (N, 1)"
        DMD_modes[:,i] = np.reshape(phi, (N,))

    # column DMD_modes[:,i] is the dynamic mode corresponding to the eigenvalue Lambda[i].
    return Abar, Lambda, DMD_modes


################################################ MAIN SCRIPT ####################################################

# load the (N, T_tot) matrix of daily returns snapshots 

#root = tk.Tk() # create a tkinter window
#root.withdraw() # hide the tkinter window
## open a file dialog to select the csv file
#file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")]) # open a file dialog to select the npy file
#root.destroy() # destroy the tkinter window
returns_file_path = 'CRSP_20170101-20241231_527PERMNOs_returnsSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy'
adjPrices_file_path = 'CRSP_20170101-20241231_527PERMNOs_adjPricesSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy'
logAdjPrices_file_path = 'CRSP_20170101-20241231_527PERMNOs_logAdjPricesSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy'
portfolio_name = returns_file_path.split('_')[2]
Y = np.load(returns_file_path)
P = np.load(adjPrices_file_path)
logP = np.load(logAdjPrices_file_path)
N,T_tot = Y.shape
# assert both Y and P have the same shape
assert Y.shape == P.shape, "Error: Y and P have different shapes"
assert Y.shape == logP.shape, "Error: Y and logP have different shapes"
assert P.shape == logP.shape, "Error: P and logP have different shapes"

# load the unique tickers sorted into blocks by NAICS and then within each block, sorted by market cap
with open('527PERMNOs_tickers_sortedByNAICS_sortedByMKTCAPDAY0.txt', 'r') as f:
    tickers = [line.strip() for line in f.readlines()]

# split the returns data into training and test sets
Y_train, Y_test, T_train_Y, T_test_Y = train_test_split(Y)
# split the price data into the same training and test sets
P_train, P_test, T_train_P, T_test_P = train_test_split(P)
# split the log price data into the same training and test sets
logP_train, logP_test, T_train_logP, T_test_logP = train_test_split(logP)


###### SCRIPT SETTINGS #######################################################

chosen_rank = 20
n = 2

# use only last 100 points in training data
#dmd_fitting_window_length = 600

# # or use all training data
_, dmd_fitting_window_length = P_train.shape



######### try prices snapshots DMD #########################################################


#singularvals_prices_filename = f'singularvals_{adjPrices_file_path.split('.npy')[0]}.png' 
#U_prices, S_prices, VT_prices, trunc_r_prices, P_denosied_if_noised = SVD_and_truncation_rank(P, singularvals_prices_filename)
#print("trunc_r_prices: ", trunc_r_prices)

# Build an exact DMD model with <chosen_rank> spatiotemporal modes
dmd_prices = DMD(svd_rank=chosen_rank, exact=True)
# Fit the DMD model
assert P_train[:,-dmd_fitting_window_length:].shape == (N, dmd_fitting_window_length), "Error: shape of P_train[:,-dmd_fitting_window_length:] is not (N, dmd_fitting_window_length)"
dmd_prices.fit(P_train[:,-dmd_fitting_window_length:])
# Plot a summary of the key spatiotemporal modes
plot_summary(dmd_prices, figsize=(12,9), title_fontsize=8, label_fontsize=6, max_sval_plot=50)

# compute the DMD modes manually
Abar_prices, Lambda_prices, DMD_modes_prices = DMD_manual(P_train[:,-dmd_fitting_window_length:], chosen_rank)

# create input snapshots matrix for test set
P_test_input = P_test[:, :-1]
# concatenate the final column of P_train to the leftmost column position in P_test_input
P_test_input = np.concatenate((P_train[:,-1:], P_test_input), axis=1)

# Make predictions on test set
# P_predicted = dmd_prices.predict(P_test) # this is incorrect!
# P_predicted = dmd_prices.predict(P_test_input)


# make test predictions
P_predicted = Abar_prices @ P_test_input

# check if there are complex values in P_predicted
if np.any(np.iscomplex(P_predicted)):
    print("There are complex values in P_predicted")

# plot the predicted test prices
fig, ax = plt.subplots()
for i in range(n):
    ax.plot(np.arange(T_test_P), P_predicted.T[:,i], label=f'{tickers[i]} predictions') # predictions
    ax.plot(np.arange(T_test_P), P_test.T[:,i], label=f'{tickers[i]} actual') # actual
ax.set_title(f'DMD-Predicted Prices on Test-set; First {n} stocks; SVD rank = {chosen_rank}')
ax.set_xlabel('Trading Day')
ax.set_ylabel('Price')
ax.legend()
fig.savefig(f'figures/ExactDMD_prices_svdrank{chosen_rank}_{portfolio_name}_testsetpredictions_fittingwindowlen{dmd_fitting_window_length}.png', dpi=300)
plt.close(fig)

# compute goodness of fit metrics on the training data and loss of the simulation against the test data
# compute mean squared error
MSE_training = np.mean((P_predicted - P_test)**2)
MSE_test = np.mean((P_predicted - P_test)**2)
RMSE_training = np.sqrt(MSE_training)
RMSE_test = np.sqrt(MSE_test)
print(f'RMSE_training: {RMSE_training}, RMSE_test: {RMSE_test}')

# mean absolute error (MAE)
MAE_training = np.mean(np.abs(P_predicted - P_test))
MAE_test = np.mean(np.abs(P_predicted - P_test))
print(f'MAE_training: {MAE_training}, MAE_test: {MAE_test}')




# # ######### try returns snapshots DMD #########################################################


#singularvals_returns_filename = f'singularvals_{returns_file_path.split('.npy')[0]}.png' 
#U_returns, S_returns, VT_returns, trunc_r_returns, Y_denosied_if_noised = SVD_and_truncation_rank(Y, singularvals_returns_filename)
#print("trunc_r_returns: ", trunc_r_returns)

# Build an exact DMD model with <chosen_rank> spatiotemporal modes
dmd_returns = DMD(svd_rank=chosen_rank, exact=True)
# Fit the DMD model
dmd_returns.fit(Y_train[:,-dmd_fitting_window_length:])
# Plot a summary of the key spatiotemporal modes
plot_summary(dmd_returns, figsize=(12,9), title_fontsize=8, label_fontsize=6, max_sval_plot=50)

# compute the DMD modes manually
Abar_rets, Lambda_rets, DMD_modes_rets = DMD_manual(Y_train[:,-dmd_fitting_window_length:], chosen_rank)

# create input snapshots matrix for test set
Y_test_input = Y_test[:, :-1]
# concatenate the final column of Y_train to the leftmost column position in Y_test_input
Y_test_input = np.concatenate((Y_train[:,-1:], Y_test_input), axis=1)

# Make predictions on test set
# Y_predicted = dmd_returns.predict(Y_test) # this is incorrect!
# Y_predicted = dmd_returns.predict(Y_test_input)

# make test predictions
Y_predicted = Abar_rets @ Y_test_input


# check if there are complex values in Y_predicted
if np.any(np.iscomplex(Y_predicted)):
    print("There are complex values in Y_predicted")


# plot the predicted test returns
fig, ax = plt.subplots()
for i in range(n):
    ax.plot(np.arange(T_test_Y), Y_predicted.T[:,i], label=f'{tickers[i]} predictions') # predictions
    ax.plot(np.arange(T_test_Y), Y_test.T[:,i], label=f'{tickers[i]} actual') # actual
ax.set_title(f'DMD-Predicted Returns on Test-set; First {n} stocks; SVD rank = {chosen_rank}')
ax.set_xlabel('Trading Day')
ax.set_ylabel('Return')
ax.legend()
fig.savefig(f'figures/ExactDMD_ret_svdrank{chosen_rank}_{portfolio_name}_testsetpredictions_fittingwindowlen{dmd_fitting_window_length}.png', dpi=300)
plt.close(fig)

# compute goodness of fit metrics on the training data and loss of the simulation against the test data
# compute mean squared error
MSE_training = np.mean((Y_predicted - Y_test)**2)
MSE_test = np.mean((Y_predicted - Y_test)**2)
# root mean squared error (RMSE)
RMSE_training = np.sqrt(MSE_training)
RMSE_test = np.sqrt(MSE_test)
print(f'RMSE_training: {RMSE_training}, RMSE_test: {RMSE_test}')
# mean absolute error (MAE)
MAE_training = np.mean(np.abs(Y_predicted - Y_test))
MAE_test = np.mean(np.abs(Y_predicted - Y_test))
print(f'MAE_training: {MAE_training}, MAE_test: {MAE_test}')
