import pandas as pd
import numpy as np
import scipy.linalg as sla
from scipy.stats import multivariate_normal as mv_norm


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
        self.prior = mv_norm(mean=prior_mu, cov=prior_sigma)
        self.w0 = np.T(prior_mu)
        self.V0 = prior_sigma
        self.likelihood_sigma = likelihood_sigma
        self.dimension = prior_mu.shape[0]

        self.wN = self.w0
        self.VN = self.V0
        self.posterior = self.prior
    
    def fit(self, X, y):
        self.VN = self.likelihood_sigma**2 * \
            np.linalg.inv(X.T.dot(X) + self.likelihood_sigma**2 * np.linalg(self.V0))
        self.wN = self.VN.dot(np.linalg.inv(self.V0).dot(self.w0)) + 1./(self.likelihood_sigma**2) * \
            self.VN.dot(X.T).dot(y)
        
        self.posterior = mv_norm(mean=self.mN.flatten(), cov=self.VN)

    def predict(self, X):
        mean = self.wN.dot(X)
        cov = self.wN.T.dot(self.VN).dot(self.wN) + self.likelihood_sigma ** 2
        return mean, cov


if __name__ == '__main__':
    X, y = load_data('data.csv')
    
