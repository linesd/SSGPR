
import numpy as np
from scipy.optimize import minimize 
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import numpy.linalg as LA


'''
The following class serves as a Random Fourier Feature object. This RFF object
is an approximation to an ARD RBF kernel.
'''
class RFF:

    # initialize lengthscales and amplitude to 1
    def __init__(self,dimensions,features):
        self.D     = dimensions
        self.M     = features
        self.l     = np.ones(self.D)
        self.var0 = 1.0
        self.w     = np.random.normal(size=(self.M,self.D))
        self.scale_frequencies()
        self.grad_matrix = None
        return

    # allow user to update lengthscales to desired values
    def update_lengthscales(self,lengthscales):
        assert len(lengthscales) == self.D , 'Lengthscale vector does not agree with dimensionality'
        self.l = lengthscales
        self.scale_frequencies()  
        return
    
    # allow user to update the amplitude to desired value
    def update_amplitude(self,amplitude):
        self.var0 = amplitude
        return
    
    # allow user to update the frequencies, note that the latter has to be in a vector format 
    # (for compatibility with the SSGPR algorithm)
    def update_frequencies(self, w):
        self.w = w.reshape((self.M,self.D))
        self.scale_frequencies()
        self.grad_matrix = None
        return
    
    # Function to scale the frequencies with the lengthscale
    def scale_frequencies(self):
        self.W = (1.0/self.l)*self.w #TODO: check this works for D > 1
        self.grad_matrix = None
        return
    
    # build design matrix phi
    def design_matrix(self,x):
        N = x.shape[0]
        phi_x = np.zeros((N,2*self.M))
        phi_x[:,:self.M] = np.cos(x.dot(self.W.T))
        phi_x[:,self.M:] = np.sin(x.dot(self.W.T))
        return phi_x
    
'''
The following class is a Sparse Spectrum Gaussian Process Regression object.
'''
class SSGPR:

    def __init__(self, num_basis_functions=10): 
        self.m          = num_basis_functions                            # number of basis functions to use
        self.varn       = 1   
        self.skip_noise = False
        self.skip_lengthscales = False
        self.skip_prior = False   
        self.skip_frequencies = False                   # noise
    
    def add_data(self, X, Y):
        self.X          = X  
        # self.d_mean     = Y.mean()                                    # store training data set
        self.Y          = Y #- self.d_mean                                      # store training targets
        self.rff        = RFF(X.shape[1], self.m)   # initialize the RFF                                          # spherical noise parameter
        self.N, self.D  = X.shape

    # function to make predictions on training points x (x must be in array format)
    def predict(self, Xs):

        # calculate some useful constants
        phi = self.rff.design_matrix(self.X)
        phi_star = self.rff.design_matrix(Xs)
        A = (self.rff.var0/self.m) * phi.T @ phi + self.varn * np.eye(2*self.m)
        R = LA.cholesky(A).T
        
        alpha = self.rff.var0 / self.m * LA.solve(R, LA.solve(R.T, phi.T@self.Y))#LA.solve(R, Rtiphity)
        mu = phi_star @ alpha #+ self.d_mean

        var = self.varn * (1 + self.rff.var0/self.m * np.sum((phi_star @ LA.inv(R))**2, axis=1))

        return mu, np.sqrt(var)

    def sample_posterior_weights(self, num_samples=1):
        # calculate some useful constants
        phi = self.rff.design_matrix(self.X)
        A = (self.rff.var0/self.m) * phi.T @ phi + self.varn * np.eye(2*self.m)
        R = LA.cholesky(A).T

        SN = LA.inv(A) * (self.varn * self.rff.var0 / self.m)
        mN = LA.solve(R, LA.solve(R.T, phi.T@self.Y)) * (self.rff.var0 / self.m)

        return np.random.multivariate_normal(mN[:,0], SN, num_samples).T
    
    def predict_with_posterior_sample(self, Xs, W):
        
        phi_star = self.rff.design_matrix(Xs)

        return phi_star @ W 

    # function that computes the marginal likelihood
    def negative_marginal_log_likelihood(self, params):

        self.update_optimisation_parameters(params)
        
        # calculate some useful constants
        phi = self.rff.design_matrix(self.X)
        try:
            R = LA.cholesky((self.rff.var0/self.m) * phi.T @ phi + self.varn * np.eye(2*self.m)).T
        except:
            return np.inf

        PhiRi = phi @ LA.inv(R)
        RtiPhit = PhiRi.T
        Rtiphity = RtiPhit @ self.Y

        #negative marginal log likelihood
        nmll = (self.Y.T @ self.Y - (self.rff.var0/self.m)*np.sum(Rtiphity**2)) / (2*self.varn)
        nmll += np.log(np.diag(R)).sum() + ((self.N/2) - self.m) * np.log(self.varn)
        nmll += (self.N/2)*np.log(2*np.pi)

        return nmll

    # function that computes gradients
    def gradients(self, params):

        self.update_optimisation_parameters(params)
        
        grad = np.zeros((self.D + 2 + self.D*self.m, 1))

        # calculate some useful constants
        phi = self.rff.design_matrix(self.X)
        
        try:
            R = LA.cholesky((self.rff.var0 / self.m) * phi.T @ phi + self.varn * np.eye(2 * self.m)).T
        except:
            # print("An exception occurred during gradient calculation...")
            # print("Optimiser tried: ")
            # self.print_hyperparams()
            return grad[:, 0]
        
        PhiRi = phi @ LA.inv(R)
        RtiPhit = PhiRi.T
        Rtiphity = RtiPhit @ self.Y

        A = np.concatenate(((self.Y/self.varn - (self.rff.var0/(self.varn*self.m)) * PhiRi @ Rtiphity).reshape(-1,1),
                            np.sqrt((self.rff.var0/(self.varn*self.m)))*PhiRi),axis=1)
        diagfact = -(1/self.varn) + np.sum(A**2, axis=1) 
        Aphi = A.T @ phi
        B = A @ Aphi[:,:self.m]*phi[:,self.m:] - A @ Aphi[:,self.m:]*phi[:,:self.m]                                    
        
        #DERIVATIVES START

        if self.opt_lengthscales:
            # derivatives wrt the lengthscales
            for d in range(self.D):
                grad[d] = -(0.5*2*self.rff.var0/self.m) * (self.X[:,d].T @ B @ self.rff.W[:,d])
        else:
            for d in range(self.D):
                grad[d] = 0

        if self.opt_prior:
            # derivative wrt signal power hyperparameter
            grad[self.D]= (0.5*2*self.rff.var0/self.m) * (((self.N * self.m)/self.varn) - np.sum(Aphi**2))
        else:
            grad[self.D] = 0


        if self.opt_noise:
            # derivative wrt noise power hyperparameter
            grad[self.D+1] = -0.5*np.sum(diagfact)*2*self.varn
        else:
            grad[self.D+1] = 0

        if self.opt_spectral:
            # derivatives wrt the representative frequencies
            for d in range(self.D):
                grad[self.D+2+d*self.m:self.D+2+(d+1)*self.m] =+ \
                    (0.5*2*(self.rff.var0/self.m)*(self.X[:,d].T @ B)/self.rff.l[d]).reshape(-1,1)
        else:
            grad[self.D+2+d*self.m:self.D+2+(d+1)*self.m] = 0

        return grad[:,0]

    def update_optimisation_parameters(self, params):
        # print("params shape: ", params.shape)
        if params.shape[0] == 1:
            params = params.reshape(self.m*self.D + self.D + 2, )

        self.rff.update_lengthscales(np.exp(params[:self.D]))       # update RFF lengthscales
        self.rff.update_amplitude(np.exp(2*params[self.D]))          # update RFF amplitude 
        
        if np.exp(2*params[self.D+1]) < 1e-14:
            self.varn = 1e-14
        else:
            self.varn = np.exp(2*params[self.D+1])   # update the SSGPR noise variance

        if self.opt_spectral:
            self.rff.update_frequencies(params[self.D+2:])      # update the RFF spectral frequencies

    def objective_function(self, params):
        return self.negative_marginal_log_likelihood(params)
        

    def optimise(self, loghyper=None, restarts=3, opt_spectral=False, opt_lengthscales=True,
                opt_prior=True, opt_noise=True, method='CG', maxiter=10000, verbose=True,
                load_params=False):

        if verbose:
            print('***************************************************')
            print('*              Optimizing parameters              *')
            print('***************************************************')

        self.opt_spectral=opt_spectral
        self.opt_lengthscales = opt_lengthscales
        self.opt_prior = opt_prior
        self.opt_noise = opt_noise
        self.method = method

        global_opt = np.inf

        if self.D > 1:
            covpower = np.random.normal()            
            lengthscales = np.random.normal(size=self.D)
            noisepower  = np.random.normal()
            # but the lengthscales and powers have
            # lengthscales = loghyper[0:self.D]
            # covpower = loghyper[self.D]
            # noisepower = loghyper[self.D+1]

        elif len(self.X) == 1:
            if not self.skip_prior:
                covpower = np.random.normal()
            if not self.skip_lengthscales:
                lengthscales = np.random.normal(size=self.D)
            if not self.skip_noise:
                noisepower  = np.random.normal()
        else:
            if not self.skip_lengthscales:
                # set some initial values
                lengthscales=np.log((np.max(self.X)-np.min(self.X)).T/2)
            else:
                lengthscales=np.log(self.rff.l)
            # lengthscales[lengthscales<-1e2]=-1e2
            if not self.skip_prior:
                covpower=0.5*np.log(np.var(self.Y));
            else:
                covpower=np.log(self.rff.var0)

            if not self.skip_noise:
                noisepower=0.5*np.log(np.var(self.Y)/4);
            else:
                noisepower=np.log(self.varn)


        for res in range(restarts):

            # initialise the spectral points: try 100 inits and keep best 
            if not self.skip_frequencies:
                nmll=np.inf
                for i in range(100):
                    spectral_points = np.random.normal(0,1,size=(self.m*self.D))
                    params = np.hstack((lengthscales, covpower, noisepower, spectral_points))
                    nmllc = self.negative_marginal_log_likelihood(params)
                    # print("nmll: ", nmllc)
                    if nmllc < nmll:
                        w_save = spectral_points
                        nmll = nmllc
                spectral_points = w_save
                self.rff.update_frequencies(spectral_points)
                #update params
            
            if method == "Nelder-Mead":
                if self.opt_spectral:
                    optimizeparams = np.hstack((lengthscales, covpower, noisepower, spectral_points))
                else:
                    optimizeparams = np.hstack((lengthscales, covpower, noisepower))
                #optimize the objective
                opt = minimize(self.objective_function, optimizeparams, method=method, options={'maxiter':maxiter})
            elif method == 'CG':
                optimizeparams = np.hstack((lengthscales, covpower, noisepower, spectral_points))
                #optimize the objective
                opt = minimize(self.objective_function, optimizeparams, jac=self.gradients, method=method, options={'maxiter':maxiter})
            elif method == 'L-BFGS-B':
                optimizeparams = np.hstack((lengthscales, covpower, noisepower, spectral_points))
                bounds = [(0, 1000) for _ in lengthscales] + [(0, 100), (0,100)] + [(-np.inf, np.inf) for _ in spectral_points]
                #optimize the objective
                opt = minimize(self.objective_function, optimizeparams, jac=self.gradients, method=method, bounds=bounds, options={'maxiter':maxiter})

            # check if the local optimum beat the current global optimum
            if opt['fun'] < global_opt:
                global_opt = opt['fun']                                     # update the global optimum
                self.opt = opt
                which_res = res

            # print out optimization result if the user wants
            if verbose:
                print(opt.message)
                print('restart # %i, negative log-likelihood = %.6f' %(res+1,opt['fun']))

            # randomize paramters for next iteration
            if res<restarts-1:
                self.rff.update_amplitude(np.random.normal())
                self.rff.update_lengthscales(np.random.normal(size=self.D))
                self.noise  = np.random.normal()

        # print("lengthscales: %.6f, prior: %.6f, noise: %.6f" % 
        #         (np.exp(self.opt['x'][0]), np.exp(2*self.opt['x'][1]), np.exp(2*self.opt['x'][2])))
        self.update_optimisation_parameters(self.opt['x'])

        if verbose:
            print("Using restart # %i results:" % (which_res+1))
            self.print_hyperparams()

    def print_hyperparams(self):
        print("lengthscales: ", self.rff.l)
        print("prior variance: ", self.rff.var0)
        print("noise variance: ", self.varn)

    # sets the params and then deosnt optimise them
    def set_parameters(self, lengthscales=None, noise=None, prior=None, W=None):
        if lengthscales is not None:
            self.rff.update_lengthscales(lengthscales)
            self.skip_lengthscales=True
        if noise is not None:
            self.varn = noise
            self.skip_noise=True
        if prior is not None:
            self.rff.update_amplitude(prior)
            self.skip_prior=True
        if W is not None:
            self.rff.update_frequencies(W)
            self.skip_frequencies=True
        
        

