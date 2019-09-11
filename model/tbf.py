import numpy as np

class TBF:
    """
    Trigonometric Basis Function (TBF) object which creates a (N, 2M) design matrix for the SSGPR
    where N is the number of data points and M is the number of basis functions.

    Parameters
    ----------
    dimensions : int
        Dimensionality of the data.

    num_basis_func : int
        The number of basis functions used to create the design matrix.
    """
    def __init__(self, dimensions, num_basis_func=100):
        self.D = dimensions
        self.M = num_basis_func
        self.w = np.random.normal(size=(self.M, self.D))
        self.update_lengthscales(np.ones(self.D))
        self.update_amplitude(1.0)

    def update_lengthscales(self, lengthscales):
        """
        Updates the lengthscales and scale the frequencies with the new lengthscales.

        Parameters
        ----------
        lengthscales : numpy array of shape (D,)
            lengthscale for each dimension of the training data
        """
        assert len(lengthscales) == self.D, 'Lengthscale vector does not agree with dimensionality'
        self.l = lengthscales.reshape(-1,1)
        self.scale_frequencies()

    def update_amplitude(self, amplitude):
        """
        Updates the TBF amplitude parameters

        Parameters
        ----------
        amplitude : float
        """
        self.var_0 = amplitude

    def update_frequencies(self, w):
        """
        Update the frequencies and scale them with the lengthscales.

        Parameters
        ----------
        w : numpy array of shape (M*D, )
            frequencies where M is the number of basis functions and D
            is the dimensionality of the training data
        """
        self.w = w.reshape(self.M, self.D, order='F')
        self.scale_frequencies()

    # Function to scale the frequencies with the lengthscale
    def scale_frequencies(self):
        """
        Scale the frequencies with the lengthscales.
        """
        self.W = self.w / self.l.T

    # build design matrix phi
    def design_matrix(self, X):
        """
        Create Trigonometric Basis Function (TBF) design matrix.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            data with which to create design matrix

        Return
        ------
        phi_x : numpy array of shape (2M, 2M)
            Design matrix
        """
        N = X.shape[0]
        phi_x = np.zeros((N, 2 * self.M))
        phi_x[:, :self.M] = np.cos(X @ self.W.T)
        phi_x[:, self.M:] = np.sin(X @ self.W.T)
        return phi_x
