import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR
from utils.plots import plot_predictive_1D
from math import floor
np.random.seed(1)  # set seed

# load the data
data = np.load("../data/example_data/data_1D.npy")
n = floor(0.8 * data.shape[0])
X_train = data[:n,0].reshape(-1,1)
Y_train = data[:n,1].reshape(-1,1)
X_test = data[n:,0].reshape(-1,1)
Y_test = data[n:,1].reshape(-1,1)

# create ssgpr instance
nbf = 100  # number of basis functions
ssgpr = SSGPR(num_basis_functions=nbf)
ssgpr.add_data(X_train, Y_train, X_test, Y_test)
ssgpr.optimize(restarts=3, verbose=True)

# create some prediction points
Xs = np.linspace(-10,10,100).reshape(-1,1)
mu, sigma, f_post = ssgpr.predict(Xs, sample_posterior=True, num_samples=3)
NMSE, MNLP = ssgpr.evaluate_performance(restarts=1)
print("Normalised mean squared error (NMSE): ", np.round(NMSE,5))
print("Mean negative log probability (MNLP): ", np.round(MNLP,5))

path = "../doc/imgs/example_1D.png"
#plot results
plot_predictive_1D(path=path, X_train=X_train, Y_train=Y_train, Xs=Xs, mu=mu,
                   stddev=sigma, post_sample=f_post)