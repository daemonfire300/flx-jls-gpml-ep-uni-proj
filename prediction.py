# -*- coding: utf-8 -*-
"""
v, tau are the parameters we learned before
X is our training dataset
K is the K we computed previously
y is our label vector with labels +1/-1 (ie. 6 or 8, 3 or 5)
k can be user chosen, but we assume that it is the function from kernel(.py) that we used earlier to compute K

x is our "unknown" input we want to classify/get the probabilities for.
x is not a single point but of the shape of X, but with only a single entry, otherwise kernel.compute seems to complain
about the dimensions.
"""

import scipy.linalg
import scipy.stats as sp_stats
import numpy as np

def classify(v_tilde, tau, X, K, y, kernel_fkt, x):
    S = np.diag(tau)
    S_sqrt = scipy.linalg.sqrtm(S)
    I = np.identity(X.shape[0])
    tmp = I + np.dot( np.dot(S_sqrt, K), S_sqrt)
    L = scipy.linalg.cholesky(tmp)
    z = np.linalg.solve(np.dot(S_sqrt, L.T), np.linalg.solve(L, np.dot(np.dot(S_sqrt, K), v_tilde)))
    kx = kernel_fkt(x, X)
    f = np.dot(kx.T, v_tilde - z)
    v = np.linalg.solve(L, np.dot(S_sqrt, kx))
    V_f = kernel_fkt(x, x) - np.dot(v.T, v)
    pi = sp_stats.norm.cdf(float(f) / np.sqrt(1 + V_f))
    return pi