import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_convergence(path, convergence):
    N = convergence.shape[0]
    X = np.arange(N)
    Y = convergence[:,0]
    plt.figure()
    plt.plot(X, Y)
    plt.grid()
    plt.ylabel("Negative marginal log likelihood")
    plt.xlabel("Number of iterations")
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()

def plot_predictive_1D(path=None, X_train=None, Y_train=None, Xs=None, mu=None, stddev=None, post_sample=None):
    """
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
    """
    plt.figure()
    if (X_train is not None) and (Y_train is not None):
        plt.plot(X_train, Y_train, '*', label="Training data", color='blue')  # training data
    if post_sample is not None:
        plt.plot(Xs, post_sample, '--', label="Posterior sample")
    plt.plot(Xs, mu, 'k', lw=2, label="Predictive mean")
    plt.fill_between(Xs.flat, (mu - 2 * stddev).flat, (mu + 2 * stddev).flat, color="#dddddd",
                     label="95% confidence interval")
    plt.grid()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("inputs, X")
    plt.ylabel("targets, Y")
    plt.legend(loc='lower right')
    # save img
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()

def plot_predictive_2D(path=None, X_train=None, Y_train=None, Xs1=None, Xs2=None, mu=None, stddev=None):
    """
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
    """
    # instantiate figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')
    # plot scatter of training data.
    if (X_train is not None) and (Y_train is not None):
        ax.scatter(X_train[:,0], X_train[:,1], Y_train, label="Training data", c='red')
    # plot lower 95% confidence interval
    if stddev is not None:
        p1 = ax.plot_surface(Xs1, Xs2, mu-2*stddev, label="Lower 95% confidence interval",
                            color='yellow' ,alpha=0.2, linewidth=0, antialiased=False)
        p1._facecolors2d = p1._facecolors3d # fixes matplotlib bug in surface legend
        p1._edgecolors2d = p1._edgecolors3d
    # plot surface of predicted data
    if (Xs1 is not None) and (Xs2 is not None) and (mu is not None):
        p2 = ax.plot_surface(Xs1, Xs2, mu, label="Predictive mean",
                             cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(p2, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
        p2._facecolors2d = p2._facecolors3d
        p2._edgecolors2d = p2._edgecolors3d
    # plot upper 95% confidence interval
    if stddev is not None:
        p3 = ax.plot_surface(Xs1, Xs2, mu+2*stddev, label="Upper 95% confidence interval",
                             color='green', alpha=0.2, linewidth=0, antialiased=False)
        p3._facecolors2d = p3._facecolors3d
        p3._edgecolors2d = p3._edgecolors3d
    # axis labels
    ax.set_xlabel("inputs, X1")
    ax.set_ylabel("inputs, X2")
    ax.set_zlabel("targets, y")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best')
    # save img
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()