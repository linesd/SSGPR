import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR
from math import floor

def test_predict():
    np.random.seed(1)  # set seed
    precision = 5

    # load the data
    data = np.load("../data/test_data/test_predict_data.npy")
    solution = np.load("../data/test_data/test_predict_solution.npy")
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
    NMSE, MNLP = ssgpr.evaluate_performance(restarts=1)

    assert np.round(NMSE, precision) == np.round(solution[0], precision)
    assert np.round(MNLP, precision) == np.round(solution[1], precision)

