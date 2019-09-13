# Sparse Spectrum Gaussian Process Regression (SSGPR) 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/disentangling-vae/blob/master/LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

This repository contains python code (training / predicting / evaluating / plotting) for [Sparse Spectrum Gaussian Process Regression](http://jmlr.csail.mit.edu/papers/v11/lazaro-gredilla10a.html).

Notes:
- Tested for python >= 3.6

Table of Contents:
1. [Install](#install)
2. [General Use](#general-use)

## Install

```
# clone repo
pip install -r requirements.txt
```

## General Use

- **Create object:** create an instance of the SSGPR object with: 

`ssgpr = SSGPR(num_basis_functions=nbf, optimize_freq=True)`

```
Sparse Spectrum Gaussian Process Regression (SSGPR) object.

Parameters
----------
num_basis_functions : int
    Number of trigonometric basis functions used to construct the design matrix.

optimize_freq : bool
    If true, the spectral points are optimized.
```
- **Add data:** add data to SSGPR with the add_data method: 

`ssgpr.add_data(X_train, Y_train, X_test, Y_test)`

```
Add data to the SSGPR object.

Parameters
----------
X_train : numpy array of shape (N, D)
    Training data inputs where N is the number of data training points and D is the
    dimensionality of the training data.

Y_train : numpy array of shape (N, 1)
    Training data targets where N is the number of data training points.

X_test : numpy array of shape (N, D) - default is None
    Test data inputs where N is the number of data test points and D is the
    dimensionality of the test data.

Y_test : numpy array of shape (N, 1) - default is None
    Test data inputs where N is the number of data test points.

```
- **Optimize:** optimize the SSGPR negative marginal log likelihood with conjugate gradients minimization:

`Xs, best_convergence = ssgpr.optimize(restarts=3, maxiter=1000, verbose=True)`

```
Optimize the marginal log likelihood with conjugate gradients minimization.

Parameters
----------
restarts : int
    The number of restarts for the minimization process. Defaults to 3.
    - The first minimization attempt is initialized with:
        - lengthscales: 	half of the ranges of the input dimensions
        - amplitude: 		variance of the targets
        - noise variance: 	variance of the targets divided by four
        - spectral points: 	choose the best from 100 random initializations
    - Subsequent restarts have random initialization.

maxiter : int
    The maximum number of line searches for the minimization process. Defaults to 1000.

verbose : bool
    If True, prints minimize progress.

Return
------
Xs : numpy array - Shape : (D + 2 + num_basis_functions, 1)
    The found solution.

best_convergence : numpy array - Shape : (i, 1 + D + 2 + num_basis_functions)
    Convergence information from the best restart. The first column is the negative marginal log
    likelihood returned by the function being minimized. The next D + 2 + num_basis_functions
    columns are the guesses during the minimization process. i is the number of
    linesearches performed.
```

- **Predict:** make predictions on new data with the predict method. Predict returns the predictive mean and standard deviation. Predict also has the option to return `num_samples` samples drawn from the posterior distribution over the model parameters.

`mu, stddev, f_post = ssgpr.predict(Xs, sample_posterior=False, num_samples=1)`

```
Predict on new inputs with the posterior predictive.

If sample_posterior is True, predict returns mu, stddev and f_post. 
If sample_posterior is False, predict returns mu and stddev.

Parameters
----------
Xs : numpy array of shape (N, D)
    New data inputs where N is the number of new data points and D is the dimensionality of the data.

sample_posterior : bool
    If true, predict returns num_samples samples drawn from the posterior distribution over the model 
    parameters (weights). Default is False.


num_samples : int
    Number of samples to draw from the posterior distribution over the model parameters (weights).

Return
------
mu : numpy array of shape (N, 1)
    Predictive mean where N is the number of new data points (Xs.shape[0])

stddev : numpy array of shape (N, 1)
    Predictive standard deviation where N is the number of new data points.

f_post : numpy array of shape (N, num_samples)
    num_samples samples from the posterior distribution over the model parameters
    (weights). f_post is only returned if sample_posterior = True.
```

- **Evaluate:** evaluate the SSGPR performance. `evaluate_performance` calculates the Normalised Mean Squared Error (MNSE) and the Mean Negative Log Probability (MNLP) of the predictive mean against the test data.

`NMSE, MNLP = ssgpr.evaluate_performance(restarts=3)`

```
Evaluates the performance of the predictive mean by calculating the normalized mean 
squared error (NMSE) and the mean negative log probability (MNLP) of the predictive 
mean against the test data.

If optimize has not previously been called, it is called in this
function.

Test data must first be loaded with the add_data method.

Parameters
----------
restarts : int
    The number of restarts for the minimization process. Defaults to 3.
    - The first minimization attempt is initialized with:
        - lengthscales: half of the ranges of the input dimensions
        - amplitude: variance of the targets
        - noise variance: variance of the targets divided by four
        - spectral points: choose the best from 100 random initializations
    - Subsequent restarts have random initialization.

Return
------
NMSE : numpy.float64
    Normalized mean squared error (NMSE)

MNLP : numpy.float64
    Mean negative log probability (MNLP)
``` 

- **Plotting:** plot the predictive distribution for 1-dimensional and 2-dimentional data.

	- Predictive distribution for 1-dimensional input data

	`plot_predictive_1D(path=None, X_train=None, Y_train=None, Xs=None, mu=None, stddev=None, post_sample=None)`

	```
	Plot the predictive distribution for one dimensional data.

	See example_1D.py for use.

	Parameters
	----------
	path : str
	    Path to save figure. If no path is provided then the figure is not saved.
	    
	X_train : numpy array of shape (N, 1)
	    Training data.
	    
	Y_train : numpy array of shape (N, 1)
	    Training targets.
	    
	Xs : numpy array of shape (n, 1)
	    New points used to predict on.
	    
	mu : numpy array of shape (n, 1)
	    Predictive mean generated from new points Xs.
	    
	stddev : numpy array of shape (n, 1)
	    Standard deviation generated from the new points Xs.
	    
	post_sample : numpy array of shape (n, num_samples)
	    Samples from the posterior distribution over the model parameters.
	```

	![plot_predicitive_1D](doc/imgs/example_1D.png)

	- Predictive distribution for 2-dimensional input data

	`plot_predictive_2D(path=None, X_train=None, Y_train=None, Xs1=None, Xs2=None, mu=None, stddev=None)`

	```
	Plot the predictive distribution for one dimensional data.

	See example_2D.py for use.

	Parameters
	----------
	path : str
	    Path to save figure. If no path is provided then the figure is not saved.

	X_train : numpy array of shape (N, 1)
	    Training data.

	Y_train : numpy array of shape (N, 1)
	    Training targets.

	Xs1 : numpy array of shape (n, n)
	    New points used to predict on. Xs1 should be generated with np.meshgrid (see example_2D.py).

	Xs2 : numpy array of shape (n, n)
	    New points used to predict on. Xs2 should be generated with np.meshgrid (see example_2D.py).

	mu : numpy array of shape (n, n)
	    Predictive mean generated from new points Xs1 and Xs2.

	stddev : numpy array of shape (n, n)
	    Standard deviation generated from the new points Xs1 and Xs2.

	post_sample : numpy array of shape (n, num_samples)
	    Samples from the posterior distribution over the model parameters. 

	```

	![plot_predicitive_2D](doc/imgs/example_2D.png)

## Examples

Python scripts for the following examples

## Testing
