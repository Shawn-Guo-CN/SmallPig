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
        super().__init__()
        self.prior = mv_norm(mean=prior_mu, cov=prior_sigma)
        self.m0 = np.T(prior_mu)
        self.V0 = prior_sigma
        self.likelihood_sigma = likelihood_sigma

        self.mN = self.m0
        self.VN = self.V0
        self.posterior = self.prior
    
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

def fit_blr(X, yy, sigma_w, sigma_y):
    """Bayesian linear regression, posterior and log marginal likelihood.
    
    Assume spherical zero-mean prior, with width sigma_w
    Return posterior mean w_N, chol(inv(V_N)) where V_N is posterior
    covariance, and log-marginal-likelihood.

    We use upper-triangular Cholesky's throughout (scipy.linalg's default).
    """
    N, D = X.shape
    inv_V_N = ((sigma_y/sigma_w)**2 * np.eye(D) + np.dot(X.T, X)) / sigma_y**2
    chol_inv_V_N = sla.cholesky(inv_V_N)
    w_N = sla.cho_solve((chol_inv_V_N, False), np.dot(X.T, yy)) / sigma_y**2
    # Evaluate p(w), p(y|X,w), p(w|y,X) at w=0. Hence get p(y|X)
    hl2pi = 0.5*np.log(2*np.pi)
    Lp_w0 = -D*(hl2pi + np.log(sigma_w))
    Lp_y_w0 = -0.5*np.dot(yy, yy)/sigma_y**2 - N*(hl2pi + np.log(sigma_y))
    U_w_N = np.dot(chol_inv_V_N, w_N)
    Lp_w0_yX = -0.5*np.dot(U_w_N, U_w_N) \
            - D*hl2pi + np.sum(np.log(np.diag(chol_inv_V_N)))
    Lml = Lp_w0 + Lp_y_w0 - Lp_w0_yX
    return w_N, chol_inv_V_N, Lml

if __name__ == '__main__':
    X, y = load_data('data.csv')
    
