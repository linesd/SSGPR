
import numpy as np

data_1D = np.loadtxt("../data/data_1D.csv", delimiter=",")
np.save("../data/data_1D",data_1D)