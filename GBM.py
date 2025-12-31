import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_test_split(X, train_percentage=0.95):
    '''
    split the time series into one training set and the rest test set
    
    X : (N, T_tot) numpy array
    train_percentage : float between 0 and 1: percentage of the data to be used for training

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


# model: Y[:,t] = mu + epsilon[:,t]
# where Y[:,t] is a column vector (2D numpy array with shape (N, 1)) of daily returns of all N stocks on day t
# epsilon[:,t] is a normal random variable with mean 0 and covariance matrix Omega (same Omega for all t = 1,...T)
# epsilon[:,t] is independent of epsilon[:,s] for all t != s
# mu is a column vector (2D numpy array with shape (N, 1)) of the mean daily returns of all N stocks, i.e. the drift parameters of the stocks' geometric Brownian motion models
# epsilon[i,t] is volatility_i * dW_it where dW_it is a standard normal random variable (Wiener process increment for dt = 1)

def MLE(X):
    """
    Maximum likelihood estimates for mean vector mu and covariance matrix Omega
    based on the given data matrix X of returns or log returns

    X : (N, T) numpy array : N stocks' returns or log returns, snapshots over T days
    returns: 
        mu_MLE : (N,) numpy array : mean vector for the N stocks returns or log returns
        Omega_MLE : (N, N) numpy array : covariance matrix of the stocks' returns or log returns
    """
    N, T = X.shape
    mu_MLE = np.mean(X, axis=1)
    Omega_MLE = np.cov(X, bias=True)
    assert Omega_MLE.shape == (N, N), "shape of Omega_MLE is not (N, N)"
    assert mu_MLE.shape == (N,), "shape of mu_MLE is not (N,)"
    # Omega_MLE[i,j] is volatility_i * volatility_j * c_ij where c_ij is the covariance of dW_it and dW_jt (when i=j, c_ij = 1); for any t = 1,...T
    # check that Omega_MLE is symmetric
    assert np.allclose(Omega_MLE, Omega_MLE.T), "Omega_MLE is not symmetric"
    # check that Omega_MLE is positive semidefinite
    assert np.all(np.linalg.eigvals(Omega_MLE) >= 0), "Omega_MLE is not positive semidefinite"
    # check that Omega_MLE is positive definite
    assert np.all(np.linalg.eigvals(Omega_MLE) > 0), "Omega_MLE is not positive definite"
    return mu_MLE, Omega_MLE


def simulateRet_DiscreteGBM(mu, Omega, N, T, dt=1, seed=42):
    """
    Monte Carlo simulation of the GBM model using Euler-Maruyama discretization of the GBM stochastic differential equation.
    This simulation assumes a model using SMALL INCREMENTS OF TIME, i.e. dt = 1 trading day.
    For large increments of time, the model is not valid since the simple returns' deviate significantly from normal distribution.

    Assuming the model Y[:,t] = mu + epsilon[:,t] where epsilon[:,t] is a normal random variable with mean 0 and covariance matrix Omega (same Omega for all t = 1,...T)
    epsilon[i,t] is volatility_i * dW_it where dW_it is a standard normal random variable (Wiener process increment for dt = 1)
    
    Outputs:
    - the simulated returns y[:,t] for all t = 0,...,T-1
    - the time vector t
    
    Parameters:
    - mu:     expected returns (drift parameters): (N,) array
    - Omega:  covariance matrix of the returns (same Omega for all t = 1,...T), must be symmetric positive definite: (N,N) array
    - N:      number of stocks: int
    - T:      time period in trading days: int
    - dt:     time increment in trading days (1 day by default)
    - seed:   seed of simulation: int
    """

    rng = np.random.default_rng(seed) # set the random number generator to a fixed seed for reproducibility
    num_increments = T//dt # number of time increments 
    t = np.arange(0, T, dt)
    A = np.linalg.cholesky(Omega) # Cholesky decomposition of Omega
    y = np.zeros([N, num_increments])

    for i in range(0, num_increments): # i is the time index
        drift = mu * dt # (N,) array of drift terms
        dW = rng.normal(0, np.sqrt(dt), N) # (N,) array of Wiener process increments; these are uncorrelated samples from the normal distribution
        random_shocks = (A @ dW) # (N,) array of random shocks; 
        # Applying the Cholesky decomposition of the covariance matrix to the vector of uncorrelated Wiener increments produces a sample vector with the covariance properties of the system being modeled
        # https://en.wikipedia.org/wiki/Cholesky_decomposition#Monte_Carlo_simulation 
        y[:, i] = drift + random_shocks # (N,) array of returns for time t=i*dt; the sum of the drift and the random shocks; this is the Euler-Maruyama discretization of the GBM stochastic differential equation  
    return t, y


# def simulatePrice_GBM(S0, mu, sigma, Omega, T, dt=1, seed=42):
#     """
#     Parameters

#     S0:     initial stocks' price (t=0) (N,) array
#     mu:     expected returns (drift parameters): (N,) array
#     sigma:  volatility (N,) array
#     Omega:    covariance matrix (N,N) array
#     T:      time period in trading days: int
#     dt:     time increment in trading days (1 day by default)
#     seed:   seed of simulation: int
#     """
#     rng = np.random.default_rng(seed) # set the random number generator to a fixed seed for reproducibility
#     num_increments = T//dt # number of time increments 
#     N = np.size(S0)
#     t = np.linspace(0, T, num_increments)
#     A = np.linalg.cholesky(Omega)
#     S = np.zeros([N, num_increments])
#     S[:, 0] = S0
#     for i in range(1, num_increments):    
#         drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
#         Z = np.random.normal(0., 1., N)
#         diffusion = np.matmul(A, Z) * (np.sqrt(t[i] - t[i-1]))
#         S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
#     return S, t



def GBM_fit_predict_evaluate(all_train_snapshots_prices, all_train_snapshots_returns, 
                         test_snapshots_prices, test_snapshots_returns, 
                         fitting_window_length, rng, num_sims=20, num_stocks_plot=0):
    """
    all_train_snapshots_prices: (N, T_train_all) numpy array, total training set available for prices
        (only last [fitting_window_length] of these snapshots are used for actual training)
    all_train_snapshots_returns: (N, T_train_all) numpy array, total training set available for returns
        (only last [fitting_window_length] of these snapshots are used for actual training)

    test_snapshots_prices: (N, T_test) numpy array, test set for prices
    test_snapshots_returns: (N, T_test) numpy array, test set for returns

    fitting_window_length: int, the number of snapshots to use for the actual training
    
    num_stocks_plot: int, the number of stocks to show in the test predictions plots
    
    num_sims: int, number of simulations to run
    
    rng: numpy.random.Generator, to generate random seeds


    prints:  metrics for the test returns and prices over the num_sims simulations
            each instantiation of simulation computes the RMSE and MAE for the training and test returns and prices 
            (each metric is pooled over N stocks and T_train days for the training data, and over N stocks and T_test days for the test data)
             and then averages them over the num_sims simulations
        avg_RMSE_Y_train : float, the average RMSE 
        avg_MAE_Y_train : float, the average MAE
        avg_RMSE_Y_test : float, the average RMSE 
        avg_MAE_Y_test : float, the average MAE 
        avg_RMSE_P_train : float, the average RMSE 
        avg_MAE_P_train : float, the average MAE 
        avg_RMSE_P_test : float, the average RMSE 
        avg_MAE_P_test : float, the average MAE  
    """
    assert all_train_snapshots_prices.shape == all_train_snapshots_returns.shape, "shape of all_train_snapshots_prices is not the same as shape of all_train_snapshots_returns"
    assert test_snapshots_prices.shape == test_snapshots_returns.shape, "shape of test_snapshots_prices is not the same as shape of test_snapshots_returns"
    
    N, T_train_all = all_train_snapshots_prices.shape
    _, T_test = test_snapshots_prices.shape # number of snapshots in the test set

    T_train = fitting_window_length # number of snapshots used for actual training

    train_snapshots_prices = all_train_snapshots_prices[:, -fitting_window_length:] # set used for actual training
    train_snapshots_returns = all_train_snapshots_returns[:, -fitting_window_length:] # set used for actual training


    # get the MLE estimates for mu and Omega for daily simple returns based on the training set 
    mu_MLE_ret, Omega_MLE_ret = MLE(train_snapshots_returns[:,-fitting_window_length:])

    # initialize lists to store the single-simulation RMSE and MAE values during model evaluation
    RMSEs_Y_train = []
    RMSEs_Y_test = []
    MAEs_Y_train = []
    MAEs_Y_test = []
    RMSEs_P_train = []
    RMSEs_P_test = []
    MAEs_P_train = []
    MAEs_P_test = []

    # generate random seeds, one per simulation
    generated_seeds = rng.integers(low=0, high=1000000, size=num_sims) # generate num_sims random seeds between 0 and 1000000

    for seed in generated_seeds: 
        # simulate returns for train day and test days using the MLE estimates for mu and Omega
        trange_train, simulated_train_ret = simulateRet_DiscreteGBM(mu_MLE_ret, Omega_MLE_ret, N, T_train, seed=seed)
        trange_test, simulated_test_ret = simulateRet_DiscreteGBM(mu_MLE_ret, Omega_MLE_ret, N, T_test, seed=seed)
        assert simulated_train_ret.shape == (N, T_train), f"shape of simulated_train_ret is not (N, T_train={T_train})"
        assert simulated_test_ret.shape == (N, T_test), f"shape of simulated_test_ret is not (N, T_test={T_test})"

        # Get the implied simulated(predicted) price path directly from the simulated daily returns
        # since return(t) = price(t) - price(t-1) / price(t-1) = price(t) / price(t-1) - 1
        # price(t) = price(t-1) * (1 + return(t))
        # column t of the predicted price snapshots matrix = (previous column t-1 of the predicted price snapshots matrix) element-wise-multiplied-by (1 + column t of the predicted returns matrix)
        # start with the actual initial price S0 from each of the train and test sets
        S0_train = all_train_snapshots_prices[:, 0]
        S0_test = test_snapshots_prices[:, 0]
        # create the simulated price snapshots matrix for the training and test sets
        simulated_train_prices = np.zeros([N, T_train])
        simulated_test_prices = np.zeros([N, T_test])
        simulated_train_prices[:, 0] = S0_train
        simulated_test_prices[:, 0] = S0_test
        for t in range(1, T_train):
            simulated_train_prices[:, t] = simulated_train_prices[:, t-1] * (1 + simulated_train_ret[:, t])
        for t in range(1, T_test):
            simulated_test_prices[:, t] = simulated_test_prices[:, t-1] * (1 + simulated_test_ret[:, t])


        ## compute goodness of fit metrics on the training data and loss of the simulation against the test data

        # for returns
        # dicard the initial day return when computing evaluation metrics, 
        # simply to be comparable directly to the corresponding metrics computed for DMD (since DMD code use the final 100 test points, not all 101 test points)
        # compute mean squared error

        MSE_Y_train = np.mean((simulated_train_ret[:, 1:] - train_snapshots_returns[:, 1:])**2)
        MSE_Y_test = np.mean((simulated_test_ret[:, 1:] - test_snapshots_returns[:, 1:])**2)
        # root mean squared error (RMSE)
        RMSE_Y_train = np.sqrt(MSE_Y_train)
        RMSE_Y_test = np.sqrt(MSE_Y_test)
        #print(f'RMSE_Y_train: {RMSE_Y_train}, RMSE_Y_test: {RMSE_Y_test}')
        # mean absolute error (MAE)
        MAE_Y_train = np.mean(np.abs(simulated_train_ret[:, 1:] - train_snapshots_returns[:, 1:]))
        MAE_Y_test = np.mean(np.abs(simulated_test_ret[:, 1:] - test_snapshots_returns[:, 1:]))
        #print(f'MAE_Y_train: {MAE_Y_train}, MAE_Y_test: {MAE_Y_test}')

        RMSEs_Y_train.append(RMSE_Y_train)
        MAEs_Y_train.append(MAE_Y_train)
        RMSEs_Y_test.append(RMSE_Y_test)
        MAEs_Y_test.append(MAE_Y_test)

        # for prices
        # discard the initial prices when computing goodness of fit and prediction evaluation metrics
        # because S0 was given to the model, not predicted by it
        # compute mean squared error
        MSE_P_train = np.mean((simulated_train_prices[:, 1:] - train_snapshots_prices[:, 1:])**2)
        MSE_P_test = np.mean((simulated_test_prices[:, 1:] - test_snapshots_prices[:, 1:])**2)
        # root mean squared error (RMSE)
        RMSE_P_train = np.sqrt(MSE_P_train)
        RMSE_P_test = np.sqrt(MSE_P_test)
        #print(f'RMSE_P_train: {RMSE_P_train}, RMSE_P_test: {RMSE_P_test}')
        # mean absolute error (MAE)
        MAE_P_train = np.mean(np.abs(simulated_train_prices[:, 1:] - train_snapshots_prices[:, 1:]))
        MAE_P_test = np.mean(np.abs(simulated_test_prices[:, 1:] - test_snapshots_prices[:, 1:]))
        #print(f'MAE_P_train: {MAE_P_train}, MAE_P_test: {MAE_P_test}')

        RMSEs_P_train.append(RMSE_P_train)
        MAEs_P_train.append(MAE_P_train)
        RMSEs_P_test.append(RMSE_P_test)
        MAEs_P_test.append(MAE_P_test)

    if num_stocks_plot > 0:
        # plot the predicted test returns
        fig, ax = plt.subplots()
        for i in range(num_stocks_plot):
            ax.plot(trange_test, simulated_test_ret.T[:,i], label=f'{tickers[i]} predictions') # predictions
            ax.plot(trange_test, test_snapshots_returns.T[:,i], label=f'{tickers[i]} actual') # actual
        ax.set_title(f'Test Returns for first {num_stocks_plot} stocks')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Return')
        ax.legend()
        plt.show()
        # savefig
        fig.savefig(f'figures/predicted_and_actual_returns_example_GBM.png', dpi=300)

        # plot the predicted test prices
        fig, ax = plt.subplots()
        for i in range(num_stocks_plot):
            ax.plot(trange_test, simulated_test_prices.T[:,i], label=f'{tickers[i]} predictions') # predictions
            ax.plot(trange_test, test_snapshots_prices.T[:,i], label=f'{tickers[i]} actual') # actual
        ax.set_title(f'Test Prices for first {num_stocks_plot} stocks')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Price')
        ax.legend()
        plt.show() 
        # savefig
        fig.savefig(f'figures/predicted_and_actual_prices_example_GBM.png', dpi=300)

    avg_RMSE_Y_train = np.mean(RMSEs_Y_train)
    avg_MAE_Y_train = np.mean(MAEs_Y_train)
    avg_RMSE_Y_test = np.mean(RMSEs_Y_test)
    avg_MAE_Y_test = np.mean(MAEs_Y_test)
    avg_RMSE_P_train = np.mean(RMSEs_P_train)
    avg_MAE_P_train = np.mean(MAEs_P_train)
    avg_RMSE_P_test = np.mean(RMSEs_P_test)
    avg_MAE_P_test = np.mean(MAEs_P_test)

    print(f'avg_RMSE_ret_train: {avg_RMSE_Y_train}\navg_MAE_ret_train: {avg_MAE_Y_train}\navg_RMSE_ret_test: {avg_RMSE_Y_test}\navg_MAE_ret_test: {avg_MAE_Y_test}\navg_RMSE_P_train: {avg_RMSE_P_train}\navg_MAE_P_train: {avg_MAE_P_train}\navg_RMSE_P_test: {avg_RMSE_P_test}\navg_MAE_P_test: {avg_MAE_P_test}')


    return avg_RMSE_Y_train, avg_MAE_Y_train, avg_RMSE_Y_test, avg_MAE_Y_test, avg_RMSE_P_train, avg_MAE_P_train, avg_RMSE_P_test, avg_MAE_P_test



################################################ MAIN SCRIPT ####################################################

# load the (N, T_tot) matrix of daily returns snapshots 
file_path_returns = 'CRSP_20170101-20241231_527PERMNOs_returnsSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy'
file_path_logAdjPrices = 'CRSP_20170101-20241231_527PERMNOs_logAdjPricesSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy'
file_path_adjPrices = 'CRSP_20170101-20241231_527PERMNOs_adjPricesSnapshots_sortedByNAICS_sortedByMKTCAPDAY0.npy'
Y = np.load(file_path_returns)
logP = np.load(file_path_logAdjPrices)
P = np.load(file_path_adjPrices)
N,T_tot = Y.shape
assert logP.shape == Y.shape, "shape of logP is not the same as shape of Y"
assert P.shape == Y.shape, "shape of P is not the same as shape of Y"
assert logP.shape == P.shape, "shape of logP is not the same as shape of P"
# load the unique tickers sorted into blocks by NAICS and then within each block, sorted by market cap
with open('527PERMNOs_tickers_sortedByNAICS_sortedByMKTCAPDAY0.txt', 'r') as f:
    tickers = [line.strip() for line in f.readlines()]

# split the returns data into training and test sets
Y_train, Y_test, T_train_Y, T_test_Y = train_test_split(Y)
# split the price data into the same training and test sets
P_train, P_test, T_train_P, T_test_P = train_test_split(P)


num_sims = 20
seed_for_rng_object = 42 # for reproducibility of results
rng = np.random.default_rng(seed_for_rng_object)

fitting_window_lengths_list = [600, P_train.shape[1]] # use only last 600 points in training data or use all training data

for fitting_window_length in fitting_window_lengths_list:
    print("Fitting window length: ", fitting_window_length)
    GBM_fit_predict_evaluate(P_train, Y_train, P_test, Y_test, fitting_window_length, rng, num_sims=num_sims, num_stocks_plot=0)
    print("-------------------------------------------")
