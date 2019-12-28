import pandas as pd
import numpy as np
import scipy.linalg as sla
from scipy.stats import multivariate_normal as mv_norm
import matplotlib.pyplot as plt


def load_data(file_path):
    df_csv = pd.read_csv(file_path, header=0)
    pd_data = df_csv[['dl_RealHP', 'dl_RealDI', 'dl_TotLoanVal_sa', 'dl_HouseStock', 'dl_ltv']].iloc[1:, :]
    X = pd_data.to_numpy()[:-1]
    y = df_csv[['dl_RealHP']].iloc[1:, :].to_numpy()[1:, 0]
    return X, y


class BayesianLinearRegression(object):
    def __init__(self, prior_mu, prior_sigma, likelihood_sigma):
        """
        prior_mu (np.array): prior mean vector of size 1xM
        prior_sigma (np.ndarray): prior covariance matrix of size MxM
        likelihood_sigma (float): variance of data noise
        """
        super().__init__()
        self.w0 = prior_mean
        self.V0 = prior_sigma
        self.likelihood_sigma = likelihood_sigma
        self.dimension = prior_mu.shape[0]

        self.wN = self.w0
        self.VN = self.V0

        self.posterior = None
    
    def fit(self, X, y):
        self.VN = self.likelihood_sigma**2 * \
            np.linalg.inv(X.T.dot(X) + self.likelihood_sigma**2 * np.linalg.inv(self.V0))
        self.wN = self.VN.dot(np.linalg.inv(self.V0).dot(self.w0)) + 1./(self.likelihood_sigma**2) * \
            self.VN.dot(X.T).dot(y)
        
        self.posterior = mv_norm(mean=self.wN, cov=self.VN)

    def predict(self, x):
        mean = self.wN.dot(x)
        cov = self.wN.T.dot(self.VN).dot(self.wN) + self.likelihood_sigma ** 2
        return mean, cov


if __name__ == '__main__':
    X, y = load_data('data.csv')
    print('loaded data from {}'.format('data.csv'))

    D = X.shape[1]
    sigma_w = 1.0
    sigma_y = 0.05

    prior_mean = np.zeros(D).T
    prior_cov = sigma_w * np.eye(D)

    print('fitting data ...', end=' ')
    model = BayesianLinearRegression(prior_mean, prior_cov, sigma_y)
    model.fit(X, y)
    print('done')
    
    N = X.shape[0]
    real_y = []
    pred_y = []
    pred_cov = []

    for i in range(1, N):
        real_y.append(y[i])
        _pred_y, _pred_cov = model.predict(X[i-1])
        pred_y.append(_pred_y)
        pred_cov.append(_pred_cov)

    pred_y = np.asarray(pred_y)
    pred_cov = np.asarray(pred_cov)

    x_axis = np.arange(N-1)
    plt.plot(x_axis, real_y, label='True values')
    plt.plot(x_axis, pred_y, label='Predicted values')
    plt.fill_between(x_axis, pred_y-pred_cov, pred_y+pred_cov, facecolor='green', alpha=0.3)
    plt.legend()
    plt.show()
