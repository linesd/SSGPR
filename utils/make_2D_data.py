import numpy as np
from utils.plots import plot_predictive_2D

N = 20

x = np.random.uniform(-1,1,N)
y = np.random.uniform(-1,1,N)
X, Y = np.meshgrid(x,y)
fx = X ** 3 - 3*X*Y**2
Z = fx + np.random.normal(0, 0.2, size=(X.shape))

plot_predictive_2D(X, Y, Z)

data = np.zeros(shape=(N*N,3))
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
Z = Z.reshape(-1,1)
data[:,0] = X.flat
data[:,1] = Y.flat
data[:,2] = Z.flat
np.save("../data/example_data/data_2D.npy", data)
np.savetxt("../data/example_data/data_2D.csv", data, delimiter=",")
np.savetxt("/home/dl00065/Documents/MATLAB/SSGPR/data_2D.csv", data, delimiter=",")