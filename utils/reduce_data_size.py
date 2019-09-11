import numpy as np
import matplotlib.pyplot as plt
from math import floor

data = np.load("../data/example_data/data_1D.npy")
# N = data.shape[0]
# up = floor(0.8 * N)
# train = data[:up,:]
# test = data[up:,:]
data = data[0:30,:]
np.save("../data/test_data/test_data_1D.npy", data)

# plt.figure()
# plt.scatter(data[:,0], data[:,1])
#
# plt.show()