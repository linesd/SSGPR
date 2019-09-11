import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR
from utils.plots import plot_predictive_2D
from math import floor
np.random.seed(1)  # set seed

# load the data
data = np.load("../data/example_data/data_2D.npy")
N = data.shape[0]
n = floor(N*0.8)
X_train = data[:n,0:2]
X_test = data[n:,0:2]
Y_train = data[:n,2]
Y_test = data[n:,2]

# create ssgpr instance
nbf = 100  # number of basis functions
ssgpr = SSGPR(num_basis_functions=nbf)
ssgpr.add_data(X_train, Y_train, X_test, Y_test)
ssgpr.optimize(restarts=3, verbose=True)

# create some prediction points
K = 50
xs = np.linspace(-1,1,K)
X1, X2 = np.meshgrid(xs,xs)
Xs = np.zeros((K*K, 2))
Xs[:,0] = X1.flatten()
Xs[:,1] = X2.flatten()

# predict on prediction points
mu, sigma = ssgpr.predict(Xs)
NMSE, MNLP = ssgpr.evaluate_performance()
print("Normalised mean squared error (NMSE): ", np.round(NMSE,5))
print("Mean negative log probability (MNLP): ", np.round(MNLP,5))

path = "../imgs/example_2D.png"
#plot results
plot_predictive_2D(path=path, X_train=X_train, Y_train=Y_train, X1=X1, X2=X2,
                   mu=mu.reshape(X1.shape), stddev=sigma.reshape(X1.shape))
