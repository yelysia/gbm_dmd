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

    # # check if there are complex values in Lambda
    # if np.any(np.iscomplex(Lambda)):
    #     print("There are complex values in Lambda")
    
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



def DMD_fit_predict_evaluate(all_train_snapshots, test_snapshots, chosen_rank, dmd_fitting_window_length, prices_as_state=True, num_stocks_plot=2, plot_DMD_modes_summary=False):
    """
    all_train_snapshots: (N, T_train_all) numpy array, total training set available
        (only last [dmd_fitting_window_length] of these snapshots are used for actual training)
    
    test_snapshots: (N, T_test) numpy array, test set
    
    chosen_rank: int, the number of singular values to keep in the truncated SVD of the actual training snapshots
    
    dmd_fitting_window_length: int, the number of snapshots to use for the actual training
    
    prices_as_state: bool, whether to use prices as the state variable for the DMD model; if false, indicates returns are used
    
    num_stocks_plot: int, the number of stocks to show in the test predictions plots
    
    plot_DMD_modes_summary: bool, whether to plot the summary of the key spatiotemporal modes using PyDMD
    """
    if prices_as_state:
        snapshots_quantity_name= 'prices'
    else:
        snapshots_quantity_name = 'ret'

    N, T_train_all = all_train_snapshots.shape
    _, T_test = test_snapshots.shape # number of snapshots in the test set

    T_train = dmd_fitting_window_length # number of snapshots used for actual training

    train_snapshots = all_train_snapshots[:, -dmd_fitting_window_length:] # set used for actual training

    if plot_DMD_modes_summary:
        # Build an exact DMD model with <chosen_rank> spatiotemporal modes
        pydmd_model = DMD(svd_rank=chosen_rank, exact=True)
        # Fit the DMD model
        pydmd_model.fit(train_snapshots)
        # Plot a summary of the key spatiotemporal modes
        plot_summary(pydmd_model, figsize=(12,9), title_fontsize=8, label_fontsize=6, max_sval_plot=50)
    
    # compute the DMD modes manually
    Abar, Lambda, DMD_modes = DMD_manual(train_snapshots, chosen_rank)

    # create input snapshots matrix for test set
    test_input_snapshots = test_snapshots[:, :-1] # shape: (N, T_test-1)
    # create input snapshots matrix for train set
    train_input_snapshots = train_snapshots[:, :-1] # shape: (N, T_train-1)

    # make predictions on test set
    # one_step_ahead_predicted_test_snapshots = pydmd_model.predict(test_input_snapshots)
    test_one_step_ahead_predicted_snapshots = Abar @ test_input_snapshots
    # make predictions on train set
    train_one_step_ahead_predicted_snapshots = Abar @ train_input_snapshots

    # check if there are complex valued predictions
    if np.any(np.iscomplex(test_one_step_ahead_predicted_snapshots)):
        print("There are complex valued predictions for test set")
    if np.any(np.iscomplex(train_one_step_ahead_predicted_snapshots)):
        print("There are complex valued predictions for train set")

    if num_stocks_plot > 0:
        # concatenate the first test snapshot to the beginning of the predicted test snapshots 
        test_predictions_with_inital = np.concatenate((test_snapshots[:,0:1], test_one_step_ahead_predicted_snapshots), axis=1) # shape: (N, T_test), only for plotting purposes

        # plot the predicted test snapshots
        fig, ax = plt.subplots()
        for i in range(num_stocks_plot):
            ax.plot(np.arange(T_test), test_predictions_with_inital.T[:,i], label=f'{tickers[i]} predictions') # predictions
            ax.plot(np.arange(T_test), test_snapshots.T[:,i], label=f'{tickers[i]} actual') # actual
        ax.set_title(f'DMD-Predicted {snapshots_quantity_name} on Test-set; First {num_stocks_plot} stocks; SVD rank = {chosen_rank}')
        ax.set_xlabel('Trading Day')
        if prices_as_state:
            ax.set_ylabel('Price')
        else:
            ax.set_ylabel('Return')
        ax.legend()
        fig.savefig(f'figures/ExactDMD_{snapshots_quantity_name}_svdrank{chosen_rank}_{portfolio_name}_testsetpredictions_fittingwindowlen{dmd_fitting_window_length}.png', dpi=300)
        plt.close(fig)

    print("svdrank: ", chosen_rank)
    print("dmd_fitting_window_length: ", dmd_fitting_window_length)

    # compute goodness of fit metrics on the training data and loss against the test data
    # compute mean squared error
    MSE_training = np.mean((train_one_step_ahead_predicted_snapshots - train_snapshots[:,1:])**2)
    MSE_test = np.mean((test_one_step_ahead_predicted_snapshots - test_snapshots[:,1:])**2)
    RMSE_training = np.sqrt(MSE_training)
    RMSE_test = np.sqrt(MSE_test)
    print(f'RMSE_training: {RMSE_training}, RMSE_test: {RMSE_test}')

    # mean absolute error (MAE)
    MAE_training = np.mean(np.abs(train_one_step_ahead_predicted_snapshots - train_snapshots[:,1:]))
    MAE_test = np.mean(np.abs(test_one_step_ahead_predicted_snapshots - test_snapshots[:,1:]))
    print(f'MAE_training: {MAE_training}, MAE_test: {MAE_test}')
                

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


############# RUN DMD ALGORITHM #####################

chosen_rank_list = [5, 20]
dmd_fitting_window_lengths_list = [100, 600, P_train.shape[1]] # use only last 100 or 600 points in training data or use all training data

for chosen_rank in chosen_rank_list:
    for dmd_fitting_window_length in dmd_fitting_window_lengths_list:
        # try prices snapshots 
        print("Trying prices snapshots DMD...")
        DMD_fit_predict_evaluate(P_train, P_test, chosen_rank, dmd_fitting_window_length, prices_as_state=True, num_stocks_plot=2, plot_DMD_modes_summary=False)
        # try returns snapshots 
        print("Trying returns snapshots DMD...")
        DMD_fit_predict_evaluate(Y_train, Y_test, chosen_rank, dmd_fitting_window_length, prices_as_state=False, num_stocks_plot=2, plot_DMD_modes_summary=False)
        print("-------------------------------------------")