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

def compute_eq_3_58(sigma_sqrd_i, mu_i, y_i, z_i):
    # 3.58
    zaehlerS = sigma_sqrd_i**2 * scipy.stats.norm.pdf( z_i )
    nennerS  = (1.0 + sigma_sqrd_i) * scipy.stats.norm.cdf( z_i )
    multiS   = z_i + scipy.stats.norm.pdf( z_i ) / scipy.stats.norm.cdf( z_i )
    sigma_sqrd_hat_i = sigma_sqrd_i - zaehler / nenner * multi

    zaehlerM = y_i * sigma_sqrd_i * scipy.stats.norm.pdf( z_i )
    nennerM  = scipy.stats.norm.cdf( z_i ) * np.sqrt(1.0 + sigma_sqrd_i)
    mu_hat_i = mu_i + zaehlerM / nennerM

    return (0,0)#(sigma_sqrd_hat_i, mu_hat_i) # TODO
    pass


def EP_binary_classification(K, y):
    # init
    v     = np.zeros(y.shape[0]) # v_tilde
    tau   = np.zeros(y.shape[0]) # tau_hat
    Sigma = K.copy()    
    mu    = np.zeros(y.shape[0])
    S_tilde = np.diag(tau)
    
    Sigma_before = Sigma.copy()
    mu_before[:] = mu
    sigma_sqrd_i_minus = 0
    
    z = np.zeros(y.shape[0])
    y = np.zeros(y.shape[0])
    Z_hat = np.zeros(y.shape[0])
    # repeat
    for step in range(50):
        for i in range(y.shape[0]):
            if step == 0:
                # for the first time we just set Z_hat_i = 1
                Z_hat[i] = sp_stats.norm.cdf(1.0)
                z[i] = Z_hat[i] * sp_stats.norm.pdf()
            elif step > 0:
                    sigma_sqrd_i_minus = Sigma_before[i,i]
                    Sigma_before = Sigma.Copy()
                    
            sigma_sqrd_i = Sigma[i,i]
            #sigma_2i= None #?????  (1.0 / sigma2i - 1.0 / sigma2i_tilda ) # 3.56
            tau_before   = 1.0/sigma_sqrd_i - tau[i]
            v_before     = 1.0/sigma_sqrd_i * mu[i] - v[i]

            sigma_sqrd_hat_i, mu_hat_i = compute_eq_3_58(
                                                sigma_sqrd_i,
                                                0, #TODO: mu_i
                                                y[i],
                                                0) #TODO: z_i
            delta_tau = 1.0/sigma_sqrd_hat_i - tau_before - tau[i] # 3.59
            tau[i] += delta_tau
            v[i]    = 1.0/sigma_sqrd_hat_i * mu_hat_i - v_before # 3.59
            Sigma   = Sigma - np.dot( Sigma[i] / float( 1.0/delta_tau + Sigma[i,i] ), Sigma[i].T)
            mu      = np.dot(Sigma, v)

        #L = scipy.linalg.cholesky(....)
        #V = np.linalg.solve( L.T, np.dot(...) ) # ??? was ist S_tilda
        # ^----- http://stackoverflow.com/questions/22163113/matrix-multiplication-solve-ax-b-solve-for-x
        
        # http://stattrek.com/statistics/notation.aspx
        S_sqrt = scipy.linalg.sqrtm(S_tilde)
        SKS = np.dot() # TODO
        SK_dot = np.dot() # TODO
        V = np.linalg.solve(L.T, SK_dot)
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
