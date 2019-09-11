
import numpy as np
from optimizer.minimize import minimize
import numpy.linalg as LA

'''
The following class serves as a Trigonometric Basis Function object. This object
is an approximation to an ARD RBF kernel.
'''
class TBF:

    # initialize lengthscales and amplitude to 1
    def __init__(self, dimensions, features):
        self.D     = dimensions
        self.M     = features
        self.update_lengthscales(np.ones(self.D))
        self.update_amplitude(1.0)
        self.update_frequencies(np.random.normal(size=(self.M,self.D)))

    # allow user to update lengthscales to desired values
    def update_lengthscales(self, lengthscales):
        assert len(lengthscales) == self.D , 'Lengthscale vector does not agree with dimensionality'
        self.l = lengthscales
        self.scale_frequencies()
    
    # allow user to update the amplitude to desired value
    def update_amplitude(self, amplitude):
        self.var_0 = amplitude
    
    # allow user to update the frequencies, note that the latter has to be in a vector format 
    # (for compatibility with the SSGPR algorithm)
    def update_frequencies(self, w):
        self.w = w.reshape((self.M,self.D))
        self.scale_frequencies()
    
    # Function to scale the frequencies with the lengthscale
    def scale_frequencies(self):
        self.W = self.w / self.l
    
    # build design matrix phi
    def design_matrix(self, X):
        N = X.shape[0]
        phi_x = np.zeros((N, 2*self.M))
        phi_x[:,:self.M] = np.cos(X @ self.W.T)
        phi_x[:,self.M:] = np.sin(X @ self.W.T)
        return phi_x


# Things this class needs
# - ability to sample a set of frequencies
# - 
    
'''
The following class is a Sparse Spectrum Gaussian Process Regression object.
'''
class SSGPR:

    def __init__(self, num_basis_functions=100, optimize_freq=True):
        self.m             = num_basis_functions  # number of basis functions to use
        self.optimize_freq = optimize_freq        # optimise spectral frequencies?

    def add_data(self, X_train, Y_train, X_test=None, Y_test=None):
        self.X          = X_train
        self.Y_mean     = Y_train.mean()                 # store training data set
        self.Y          = Y_train - self.Y_mean          # store training targets
        self.tbf        = TBF(X_train.shape[1], self.m)  # initialize the TBF                                          # spherical noise parameter
        self.N, self.D  = X_train.shape

        # initialise hyper-parameters from data
        lengthscales = np.log((np.max(self.X) - np.min(self.X)).T / 2)
        amplitude    = np.log(np.var(self.Y))
        noisevar     = np.log(np.var(self.Y) / 4)
        params = np.hstack((lengthscales, amplitude, noisevar, np.ones(self.D)))
        self.update_parameters(params)

    def update_parameters(self, params):
        self.tbf.update_lengthscales(np.exp(params[:self.D]))  # update TBF lengthscales
        self.tbf.update_amplitude(np.exp(params[self.D]))      # update TBF amplitude
        self.var_n = np.exp(params[self.D + 1])                # update noise variance
        self.tbf.update_frequencies(params[self.D + 2:])       # update the TBF spectral frequencies

    # function to make predictions on training points x (x must be in array format)
    def predict(self, Xs):

        # calculate some useful constants
        phi = self.tbf.design_matrix(self.X)
        phi_star = self.tbf.design_matrix(Xs)
        A = (self.tbf.var_0/self.m) * phi.T @ phi + self.var_n * np.eye(2*self.m)
        R = LA.cholesky(A).T
        
        alpha = self.tbf.var_0 / self.m * LA.solve(R, LA.solve(R.T, phi.T@self.Y))#LA.solve(R, Rtiphity)
        mu = phi_star @ alpha + self.d_mean

        var = self.var_n * (1 + self.tbf.var_0/self.m * np.sum((phi_star @ LA.inv(R))**2, axis=1))

        return mu, np.sqrt(var)

    # def sample_posterior_weights(self, num_samples=1):
    #     # calculate some useful constants
    #     phi = self.tbf.design_matrix(self.X)
    #     A = (self.tbf.var_0/self.m) * phi.T @ phi + self.var_n * np.eye(2*self.m)
    #     R = LA.cholesky(A).T
    # 
    #     SN = LA.inv(A) * (self.var_n * self.tbf.var_0 / self.m)
    #     mN = LA.solve(R, LA.solve(R.T, phi.T@self.Y)) * (self.tbf.var_0 / self.m)
    # 
    #     return np.random.multivariate_normal(mN[:,0], SN, num_samples).T
    
    # def predict_with_posterior_sample(self, Xs, W):
    #     
    #     phi_star = self.tbf.design_matrix(Xs)
    # 
    #     return phi_star @ W 

    # function that computes the marginal likelihood
    def negative_marginal_log_likelihood(self, params):

        self.update_parameters(params)
        phi = self.tbf.design_matrix(self.X)
        # calculate some useful constants
        R = LA.cholesky((self.tbf.var_0/self.m) * phi.T @ phi + self.var_n * np.eye(2*self.m)).T

        PhiRi = phi @ LA.inv(R)
        RtiPhit = PhiRi.T
        Rtiphity = RtiPhit @ self.Y
        Rtiphity2 = LA.solve(R.T, phi.T @ self.Y)
        
        #negative marginal log likelihood
        nmll = (self.Y.T @ self.Y - (self.tbf.var_0/self.m)*np.sum(Rtiphity**2)) / (2*self.var_n)
        nmll += np.log(np.diag(R)).sum() + ((self.N/2) - self.m) * np.log(self.var_n)
        nmll += (self.N/2)*np.log(2*np.pi)

        return nmll

    # function that computes gradients
    def gradients(self, params):

        self.update_parameters(params)
        
        grad = np.zeros((self.D + 2 + self.D*self.m, 1))

        # calculate some useful constants
        phi = self.tbf.design_matrix(self.X)
        
        R = LA.cholesky((self.tbf.var_0 / self.m) * phi.T @ phi + self.var_n * np.eye(2 * self.m)).T
        
        PhiRi = phi @ LA.inv(R)
        RtiPhit = PhiRi.T
        Rtiphity = RtiPhit @ self.Y

        A = np.concatenate(((self.Y/self.var_n - (self.tbf.var_0/(self.var_n*self.m)) * PhiRi @ Rtiphity).reshape(-1,1),
                            np.sqrt((self.tbf.var_0/(self.var_n*self.m)))*PhiRi),axis=1)
        diagfact = -(1/self.var_n) + np.sum(A**2, axis=1) 
        Aphi = A.T @ phi
        B = A @ Aphi[:,:self.m]*phi[:,self.m:] - A @ Aphi[:,self.m:]*phi[:,:self.m]                                    
        
        #DERIVATIVES START
        # derivatives wrt the lengthscales
        for d in range(self.D):
            grad[d] = -(0.5*2*self.tbf.var_0/self.m) * (self.X[:,d].T @ B @ self.tbf.W[:,d])

        # derivative wrt signal power hyperparameter
        grad[self.D]= (0.5*2*self.tbf.var_0/self.m) * (((self.N * self.m)/self.var_n) - np.sum(Aphi**2))

        # derivative wrt noise power hyperparameter
        grad[self.D+1] = -0.5*np.sum(diagfact)*2*self.var_n

        if self.opt_spectral:
            # derivatives wrt the representative frequencies
            for d in range(self.D):
                grad[self.D+2+d*self.m:self.D+2+(d+1)*self.m] =+ \
                    (0.5*2*(self.tbf.var_0/self.m)*(self.X[:,d].T @ B)/self.tbf.l[d]).reshape(-1,1)
        else:
            grad[self.D+2+d*self.m:self.D+2+(d+1)*self.m] = 0

        return grad[:,0]

    def objective_function(self, params):
        nmll = self.negative_marginal_log_likelihood(params)
        grad = self.gradients(params)
        return np.vstack((nmll, grad))

    def optimise(self, restarts=3, opt_spectral=False, maxiter=10000, verbose=True):

        if verbose:
            print('***************************************************')
            print('*              Optimizing parameters              *')
            print('***************************************************')

        self.opt_spectral=opt_spectral
        global_opt = np.inf

        for res in range(restarts):

            # initialise the spectral points: try 100 inits and keep best
            nmll=np.inf
            for i in range(100):
                spectral_sample = np.random.normal(0,1,size=(self.m*self.D))
                params = np.hstack((lengthscales, covpower, noisepower, spectral_sample))
                nmllc = self.negative_marginal_log_likelihood(params)
                if nmllc < nmll:
                    spectral_points = spectral_sample
                    nmll = nmllc
            self.tbf.update_frequencies(spectral_points) #update params
            
            # minimize
            params = np.hstack((lengthscales, covpower, noisepower, spectral_points))
            Xs, convergence, _ = minimize(self.objective_function, params, reduction=None, verbose=True)

            # check if the local optimum beat the current global optimum
            if convergence[-1,0] < global_opt:
                global_opt = convergence[-1,0]                                     # update the global optimum
                self.Xs = Xs
                which_res = res

            # print out optimization result if the user wants
            if verbose:
                print('restart # %i, negative log-likelihood = %.6f' %(res+1,convergence[-1,0] ))

            # randomize paramters for next iteration
            if res<restarts-1:
                self.tbf.update_amplitude(np.random.normal())
                self.tbf.update_lengthscales(np.random.normal(size=self.D))
                self.noise  = np.random.normal()

        self.update_parameters(self.opt['x'])

        if verbose:
            print("Using restart # %i results:" % (which_res+1))
            self.print_hyperparams()

    def print_hyperparams(self):
        print("lengthscales: ", self.tbf.l)
        print("prior variance: ", self.tbf.var_0)
        print("noise variance: ", self.var_n)

    # sets the params and then deosnt optimise them
    # def set_parameters(self, lengthscales=None, noise=None, prior=None, W=None):
    #     if lengthscales is not None:
    #         self.tbf.update_lengthscales(lengthscales)
    #         self.skip_lengthscales=True
    #     if noise is not None:
    #         self.var_n = noise
    #         self.skip_noise=True
    #     if prior is not None:
    #         self.tbf.update_amplitude(prior)
    #         self.skip_prior=True
    #     if W is not None:
    #         self.tbf.update_frequencies(W)
    #         self.skip_frequencies=True
        
        

