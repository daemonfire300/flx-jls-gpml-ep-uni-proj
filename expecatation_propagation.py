import numpy as np
import scipy
import matplotlib.pyplot as plt
import kernel

# returns data and +1/-1 class
def getTrainingData(data):
    x = np.zeros((data.shape[0],5)) # NxD
    y = np.zeros((data.shape[0], 5)) # YxD
    K = np.zeros((data.shape[0], data.shape[0])) # NxY
    # ^---- warum machst du K hier, K wird doch berechnet indem man kernel(X,Y,...) aufruft
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

def compute_sigma_sqrd_hat_i(sigma_sqrd_i, z_i):
    pass

def EP_binary_classification(K, y):
    # init
    v     = np.zeros(y.shape[0])
    tau   = np.zeros(y.shape[0])
    Sigma = K.copy()
    mu    = np.zeros(y.shape[0])
    S_tilde = np.diag(tau)
    # repeat
    for _ in range(50):
        for i in range(y.shape[0]): # ???? is n=N ? yes it is
            sigma_2i= None #?????  (1.0 / sigma2i - 1.0 / sigma2i_tilda ) # 3.56
            tau_i   = sigma_2i - tau[i]
            v_i     = mu[i] * sigma_2i - v[i]

            sigma_2i_dach = None # ???
            delta_tau = sigma_2i_dach - tau_i - tau[i] # 3.59
            tau[i] += delta_tau
            v[i]    = sigma_2i_dach - v_i # 3.59
            Sigma   = Sigma -  np.dot( Sigma[i] / float( 1.0/delta_tau + Sigma[i,i] ), Sigma[i].T)
            mu      = np.dot(Sigma, v)

        #L = scipy.linalg.cholesky(....)
        #V = np.linalg.solve( L.T, np.dot(...) ) # ??? was ist S_tilda
        # ^----- http://stackoverflow.com/questions/22163113/matrix-multiplication-solve-ax-b-solve-for-x
        S_sqrt = scipy.linalg.sqrtm(S_tilde)
        SKS = np.dot()
        SK_dot = np.dot()
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
