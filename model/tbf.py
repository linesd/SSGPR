import numpy as np

'''
The following class serves as a Trigonometric Basis Function object. This object
is an approximation to an ARD RBF kernel.
'''

class TBF:

    # initialize lengthscales and amplitude to 1
    def __init__(self, dimensions, features):
        self.D = dimensions
        self.M = features
        self.update_lengthscales(np.ones(self.D))
        self.update_amplitude(1.0)
        self.update_frequencies(np.random.normal(size=(self.M, self.D)))

    # allow user to update lengthscales to desired values
    def update_lengthscales(self, lengthscales):
        assert len(lengthscales) == self.D, 'Lengthscale vector does not agree with dimensionality'
        self.l = lengthscales
        self.scale_frequencies()

    # allow user to update the amplitude to desired value
    def update_amplitude(self, amplitude):
        self.var_0 = amplitude

    # allow user to update the frequencies, note that the latter has to be in a vector format
    # (for compatibility with the SSGPR algorithm)
    def update_frequencies(self, w):
        self.w = w.reshape((self.M, self.D))
        self.scale_frequencies()

    # Function to scale the frequencies with the lengthscale
    def scale_frequencies(self):
        self.W = self.w / self.l

    # build design matrix phi
    def design_matrix(self, X):
        N = X.shape[0]
        phi_x = np.zeros((N, 2 * self.M))
        phi_x[:, :self.M] = np.cos(X @ self.W.T)
        phi_x[:, self.M:] = np.sin(X @ self.W.T)
        return phi_x
