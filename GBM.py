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

# get the MLE estimates for mu and Omega for daily simple returns
#mu_MLE_Y, Omega_MLE_Y = MLE(Y_train)
mu_MLE_Y, Omega_MLE_Y = MLE(Y_train[:,-600:])

def predict_and_evaluate(num_sims):
    """
    num_sims: int, number of simulations to run

    return:  metrics for the test returns and prices over the num_sims simulations
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
    RMSEs_Y_train = []
    RMSEs_Y_test = []
    MAEs_Y_train = []
    MAEs_Y_test = []
    RMSEs_P_train = []
    RMSEs_P_test = []
    MAEs_P_train = []
    MAEs_P_test = []

    for i in range(num_sims):

        # pick a seed
        seed_picked = np.random.randint(0, 1000000)

        # simulate returns for train day and test days using the MLE estimates for mu and Omega
        trange_train_Y, Y_simulated_train = simulateRet_DiscreteGBM(mu_MLE_Y, Omega_MLE_Y, N, T_train_Y, seed=seed_picked)
        trange_test_Y, Y_simulated_test = simulateRet_DiscreteGBM(mu_MLE_Y, Omega_MLE_Y, N, T_test_Y, seed=seed_picked)
        # shape of Y_simulated_train and Y_simulated_test
        # print("shape of Y_simulated_train: ", Y_simulated_train.shape)
        # print("shape of Y_simulated_test: ", Y_simulated_test.shape)

        # Get the implied simulated(predicted) price path directly from the simulated daily returns
        # since return(t) = price(t) - price(t-1) / price(t-1) = price(t) / price(t-1) - 1
        # price(t) = price(t-1) * (1 + return(t))
        # column t of the predicted price snapshots matrix = (previous column t-1 of the predicted price snapshots matrix) element-wise-multiplied-by (1 + column t of the predicted returns matrix)
        # start with the actual initial price S0 from each of the train and test sets
        S0_train = P_train[:, 0]
        S0_test = P_test[:, 0]
        # create the price snapshots matrix for the training and test sets
        P_simulated_train = np.zeros([N, T_train_P])
        P_simulated_test = np.zeros([N, T_test_P])
        P_simulated_train[:, 0] = S0_train
        P_simulated_test[:, 0] = S0_test
        for t in range(1, T_train_P):
            P_simulated_train[:, t] = P_simulated_train[:, t-1] * (1 + Y_simulated_train[:, t])
        for t in range(1, T_test_P):
            P_simulated_test[:, t] = P_simulated_test[:, t-1] * (1 + Y_simulated_test[:, t])
        # discard the initial prices when computing goodness of fit and prediction evaluation metrics
        # because S0 was given to the model, not predicted by it

        ## compute goodness of fit metrics on the training data and loss of the simulation against the test data

        # for returns
        # compute mean squared error
        MSE_Y_train = np.mean((Y_simulated_train - Y_train)**2)
        MSE_Y_test = np.mean((Y_simulated_test - Y_test)**2)
        # root mean squared error (RMSE)
        RMSE_Y_train = np.sqrt(MSE_Y_train)
        RMSE_Y_test = np.sqrt(MSE_Y_test)
        print(f'RMSE_Y_train: {RMSE_Y_train}, RMSE_Y_test: {RMSE_Y_test}')
        # mean absolute error (MAE)
        MAE_Y_train = np.mean(np.abs(Y_simulated_train - Y_train))
        MAE_Y_test = np.mean(np.abs(Y_simulated_test - Y_test))
        print(f'MAE_Y_train: {MAE_Y_train}, MAE_Y_test: {MAE_Y_test}')

        RMSEs_Y_train.append(RMSE_Y_train)
        MAEs_Y_train.append(MAE_Y_train)
        RMSEs_Y_test.append(RMSE_Y_test)
        MAEs_Y_test.append(MAE_Y_test)

        # for prices
        # discard the initial prices when computing goodness of fit and prediction evaluation metrics
        # because S0 was given to the model, not predicted by it
        # compute mean squared error
        MSE_P_train = np.mean((P_simulated_train[:, 1:] - P_train[:, 1:])**2)
        MSE_P_test = np.mean((P_simulated_test[:, 1:] - P_test[:, 1:])**2)
        # root mean squared error (RMSE)
        RMSE_P_train = np.sqrt(MSE_P_train)
        RMSE_P_test = np.sqrt(MSE_P_test)
        print(f'RMSE_P_train: {RMSE_P_train}, RMSE_P_test: {RMSE_P_test}')
        # mean absolute error (MAE)
        MAE_P_train = np.mean(np.abs(P_simulated_train[:, 1:] - P_train[:, 1:]))
        MAE_P_test = np.mean(np.abs(P_simulated_test[:, 1:] - P_test[:, 1:]))
        print(f'MAE_P_train: {MAE_P_train}, MAE_P_test: {MAE_P_test}')

        RMSEs_P_train.append(RMSE_P_train)
        MAEs_P_train.append(MAE_P_train)
        RMSEs_P_test.append(RMSE_P_test)
        MAEs_P_test.append(MAE_P_test)

    # # plot the predicted test returns
    # n = 2
    # fig, ax = plt.subplots()
    # for i in range(n):
    #     ax.plot(trange_test_Y, Y_simulated_test.T[:,i], label=f'{tickers[i]} predictions') # predictions
    #     ax.plot(trange_test_Y, Y_test.T[:,i], label=f'{tickers[i]} actual') # actual
    # ax.set_title(f'Test Returns for first {n} stocks')
    # ax.set_xlabel('Trading Day')
    # ax.set_ylabel('Return')
    # ax.legend()
    # plt.show()
    # # savefig
    # fig.savefig(f'figures/predicted_and_actual_returns_example_GBM.png', dpi=300)


    # # plot the predicted test prices
    # fig, ax = plt.subplots()
    # for i in range(n):
    #     ax.plot(trange_test_Y, P_simulated_test.T[:,i], label=f'{tickers[i]} predictions') # predictions
    #     ax.plot(trange_test_Y, P_test.T[:,i], label=f'{tickers[i]} actual') # actual
    # ax.set_title(f'Test Prices for first {n} stocks')
    # ax.set_xlabel('Trading Day')
    # ax.set_ylabel('Price')
    # ax.legend()
    # plt.show() 
    # # savefig
    # fig.savefig(f'figures/predicted_and_actual_prices_example_GBM.png', dpi=300)

    avg_RMSE_Y_train = np.mean(RMSEs_Y_train)
    avg_MAE_Y_train = np.mean(MAEs_Y_train)
    avg_RMSE_Y_test = np.mean(RMSEs_Y_test)
    avg_MAE_Y_test = np.mean(MAEs_Y_test)
    avg_RMSE_P_train = np.mean(RMSEs_P_train)
    avg_MAE_P_train = np.mean(MAEs_P_train)
    avg_RMSE_P_test = np.mean(RMSEs_P_test)
    avg_MAE_P_test = np.mean(MAEs_P_test)
    return avg_RMSE_Y_train, avg_MAE_Y_train, avg_RMSE_Y_test, avg_MAE_Y_test, avg_RMSE_P_train, avg_MAE_P_train, avg_RMSE_P_test, avg_MAE_P_test


num_sims = 20
avg_RMSE_Y_train, avg_MAE_Y_train, avg_RMSE_Y_test, avg_MAE_Y_test, avg_RMSE_P_train, avg_MAE_P_train, avg_RMSE_P_test, avg_MAE_P_test = predict_and_evaluate(num_sims)
print(f'avg_RMSE_Y_train: {avg_RMSE_Y_train}\navg_MAE_Y_train: {avg_MAE_Y_train}\navg_RMSE_Y_test: {avg_RMSE_Y_test}\navg_MAE_Y_test: {avg_MAE_Y_test}\navg_RMSE_P_train: {avg_RMSE_P_train}\navg_MAE_P_train: {avg_MAE_P_train}\navg_RMSE_P_test: {avg_RMSE_P_test}\navg_MAE_P_test: {avg_MAE_P_test}')


