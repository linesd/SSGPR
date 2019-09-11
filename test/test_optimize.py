import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR

def test_optimize():
    np.random.seed(1) # set seed
    # load the data
    data = np.load("../data/test_data/test_optimize_data.npy")
    solution = np.load("../data/test_data/test_optimize_solution.npy")
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)

    # create ssgpr instance
    nbf = 100 # number of basis functions
    ssgpr = SSGPR(num_basis_functions=nbf)
    ssgpr.add_data(X, Y)
    Xs, _ = ssgpr.optimize(restarts=1, verbose=False)

    assert all(Xs == solution)
