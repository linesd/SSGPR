import numpy as np

def check_grad(f, X, e, args=()):
    """
    This function checks the gradients of f by comparing them to
    finite difference approximations. The partial derivatives
    returned by f and the finite difference approximations are
    printed for comparison.

    Parameters
    ----------
    f : function to minimize. The function must return the value
        of the function (float) and a numpy array of partial
        derivatives of shape (D,) with respect to X, where D is
        the dimensionality of the function.

    X : numpy array - Shape : (D, 1)
        argument for function f that the partial derivatives
        relate to.

    e : float
        size of the perturbation used for the finite differences.

    args : tuple
        Tuple of parameters to be passed to the function f.

    Return
    ------
    d : the norm of the difference divided by the norm of
        the sum of the gradients and finite differences.

    dy : gradients

    dh : finite differences
    """

    y, dy = eval('f')(X, *list(args))
    dy = dy.reshape(-1,1)
    dh = np.zeros_like(X)

    for i in range(X.shape[0]):
        dx = np.zeros_like(X)
        dx[i] = dx[i] + e
        y2, _ = eval('f')(X + dx, *list(args))
        dx = -dx
        y1, _ = eval('f')(X + dx, *list(args))
        dh[i] = (y2 - y1) / (2 * e)

    vec = np.hstack((dy,dh))
    print("Gradients vs finite difference:")
    print(vec)

    d = np.linalg.norm(dh - dy) / np.linalg.norm(dh + dy)

    return d, dy, dh