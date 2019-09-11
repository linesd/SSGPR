import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR

np.random.seed(1) # set seed

# load the data
data = np.load("../data/data_clean_1D.npy")
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

# create ssgpr instance
nbf = 100
ssgpr = SSGPR(num_basis_functions=nbf)
ssgpr.add_data(X, Y)
Xs, convergence = ssgpr.optimize(restarts=1)
print(Xs)