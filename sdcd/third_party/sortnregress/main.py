"""
From https://github.com/Scriddie/Varsortability/blob/main/src/sortnregress.py
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC


def sortnregress(X):
    """Take n x d data, order nodes by marginal variance and
    regresses each node onto those with lower variance, using
    edge coefficients as structure estimates."""
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic")

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]

        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight

    return W
