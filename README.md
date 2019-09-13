# Sparse Spectrum Gaussian Process Regression (SSGPR) 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/disentangling-vae/blob/master/LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

This repository contains python code (training / predicting / evaluating / plotting) for [Sparse Spectrum Gaussian Process Regression](http://jmlr.csail.mit.edu/papers/v11/lazaro-gredilla10a.html).

Notes:
- Tested for python >= 3.6

Table of Contents:
1. [General Use](#General Use)

## General Use

- Create an instance of the SSGPR object with: 

`ssgpr = SSGPR(num_basis_functions=nbf, optimize_freq=True)`

```
SSGPR
-----

Sparse Spectrum Gaussian Process Regression (SSGPR) object.

Parameters
----------
num_basis_functions : int
    Number of trigonometric basis functions used to construct the design matrix.

optimize_freq : bool
    If true, the spectral points are optimized.
```
- Add data to SSGPR with the add_data method: 

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


## Examples
### 1-Dimensional Data

### 2-Dimensional Data

### High-Dimensioal Data

### Evaluating SSGPR Performance

## Testing
