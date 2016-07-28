import numpy as np
import scipy
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import kernel

# returns data and +1/-1 class
def getTrainingData(data):
    x = np.zeros((data.shape[0],5)) # NxD
    y = np.zeros((data.shape[0], 5)) # YxD
    K = np.zeros((data.shape[0], data.shape[0])) # NxY
    # TODO
    for e in x:
        # todo
        continue
    return (x,y)
    
def getRandomTrainingData(data):
    x = np.random.rand(data.shape[0],5) # NxD
    y = np.zeros((data.shape[0], 1)) # YxD
    """    
    y[:] = 1
    y[0,1] = 0
    y[0,3] = 0
    """
    return (x,y)

"""
Computes the moments using eq (3.58) Rasmussen Ch3.
This is done for Line 7 of Rasmussen Pseudo Code Page 58, code (3.5).
"""
def compute_moments(sigma_sqrd_before, mu_before, y_i, z_i):
    numerator_sigma     = sigma_sqrd_before**2 * scipy.stats.norm.pdf( z_i ) # Numerator Part 1 for sgm sqrd hat i
    denominator_sigma   = (1.0 + sigma_sqrd_before) * scipy.stats.norm.cdf( z_i )
    multi_sigma         = z_i + scipy.stats.norm.pdf( z_i ) / scipy.stats.norm.cdf( z_i )
    sigma_sqrd_hat_i    = sigma_sqrd_before - numerator_sigma / denominator_sigma * multi_sigma # Assemble parts for sigma_sqrd_hat

    numeratorM          = y_i * sigma_sqrd_before * scipy.stats.norm.pdf( z_i )
    denominatorM        = scipy.stats.norm.cdf( z_i ) * np.sqrt(1.0 + sigma_sqrd_before)
    mu_hat_i = mu_before + numeratorM / denominatorM

    return (sigma_sqrd_hat_i, mu_hat_i)


def EP_binary_classification(K, y):
    # init
    N     = y.shape[0]
    v     = np.zeros(N) # v_tilde
    tau   = np.zeros(N) # tau_hat
    Sigma = K.copy()    
    mu    = np.zeros(N)
    z = np.zeros(N) #  TODO
    
    # init all with 0 ?
    sigma_sqrd_before = 0
    mu_before = 0

    # repeat
    for _ in range(50):
        for i in range(N):
                    
            sigma_sqrd_i = Sigma[i,i]
            inv_sigma_sqrd_i = 1 / sigma_sqrd_i
            tau_before   = inv_sigma_sqrd_i - tau[i]
            v_before     = inv_sigma_sqrd_i * mu[i] - v[i]

            sigma_sqrd_hat_i, mu_hat_i = compute_moments(
                                                sigma_sqrd_i,
                                                mu_before, #TODO
                                                y[i],
                                                z[i]) #TODO Was enthält z?
            inv_sigma_sqrd_hat_i = 1.0/sigma_sqrd_hat_i
            delta_tau   = inv_sigma_sqrd_hat_i - tau_before - tau[i] # 3.59
            tau[i]     += delta_tau
            v[i]        = inv_sigma_sqrd_hat_i * mu_hat_i - v_before # 3.59
            Sigma       = Sigma - np.dot( 1.0/delta_tau + Sigma[i,i],
                                          np.dot(Sigma[:,i:i+1], Sigma[i].reshape(1,N) ))
            
            mu          = np.dot(Sigma, v)
            
            #update before vars == variables with subscript -i
            sigma_sqrd_before = sigma_sqrd_i # oder sigma_sqrd_hat_i ?
            mu_before = mu[i]

        # http://stattrek.com/statistics/notation.aspx
        S_tilde = np.diag(tau)
        S_sqrt = scipy.linalg.sqrtm(S_tilde)
        L = scipy.linalg.cholesky(np.identity(N) + np.dot( np.dot(S_sqrt, K), S_sqrt))
        V = np.linalg.solve( L.T, np.dot(S_sqrt, K) )
        # ^----- http://stackoverflow.com/questions/22163113/matrix-multiplication-solve-ax-b-solve-for-x
        Sigma = K - np.dot(V.T, V)
        mu    = np.dot(Sigma, v)
    return (v, tau)


def EP_predictions(v, tau, X, y, k, xi):
    # Es gibt zwei versch. k. Was macht der mit einem input param?
    #L = scipy.linalg.cholesky(....)
    #z = np.linalg.solve(np.dot(...), np.linalg.solve(L, np.dot(...)) )
    #f = ... ??? ist das eine matrix multipliktion? 
    #vau = np.linalg.solve(L, np.dot(....))
    #Vf = k(xi, xi) - np.dot(vau.T, v)
    #pi = scipy.stats.norm.cdf( np.linalg.solve(f, np.sqrt( 1+Vf )) )
    return #pi


if __name__ == '__main__':
    data = np.zeros((100, 10))
    rndData = getRandomTrainingData(data)
    X = rndData[0]
    print(X.shape)
    y = rndData[1]
    print(y.shape)
    newPoint = np.reshape(X[0], (1,5))  # == x*
    K = kernel.compute(newPoint, X, 1) # ==> preparation for classification, 
    K_all = kernel.compute(X, X, 1) # ==> creating our "K" for learning, ie.  "input: K (covariance matrix)"
    plt.plot(K_all)
    plt.show()
    EP_binary_classification(K_all, y)
