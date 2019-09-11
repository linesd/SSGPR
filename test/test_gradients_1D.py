import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR
from optimizer.check_grad import check_grad

def test_gradients_1D():
    np.random.seed(1) # set seed
    epsilon = 1e-5    # perturbation size
    precision = 6     # decimal places

    # load the data
    data = np.load("../data/test_data/test_data_1D.npy")
    X = data[:,0].reshape(-1,1)
    Y = data[:,1].reshape(-1,1)

    # create ssgpr instance
    nbf = 10
    ssgpr = SSGPR(num_basis_functions=nbf)
    ssgpr.add_data(X, Y)

    # initialise hyper-parameters from data
    lengthscales = np.log((np.max(X) - np.min(X)).T / 2)
    amplitude = np.log(np.var(Y))
    noisevar = np.log(np.var(Y) / 4)
    spectral_sample = np.random.normal(0,1,size=(nbf))
    params = np.hstack((lengthscales, amplitude, noisevar, spectral_sample)).reshape(-1,1)

    _, dy, dh = check_grad(ssgpr.objective_function, params, epsilon)

    assert all(np.round(dy, precision) == np.round(dh, precision))