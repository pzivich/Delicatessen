import numpy as np
from scipy.misc import derivative


def partial_derivative(func, var, point, output, dx, order):
    """Calculate the partial derivatives for a given function

    Parameters
    ----------
    func : function
        Function with multiple variables to calculate the partial derivatives for
    var : int
        Index of variable to take the derivative of.
    point : ndarray
        Values to evaluate the partial derivative at
    output : int
        Index of the variable to extract the partial derivative of.
    dx : float
        Spacing to use to numerically approximate the partial derivatives of the bread matrix.
    order : int
        Number of points to use to evaluate the derivative. Must be an odd number

    Returns
    -------
    float
        Partial derivative at the variable at a particular points
    """
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(args)[output]

    return derivative(wraps, point[var], dx=dx, order=order)


def logit(prob):
    """Logit transformation of probabilities. Input can be a single probability of array of probabilities

    Parameters
    ----------
    prob : float, array
        A single probability or an array of probabilities

    Returns
    -------
    logit-transformed probabilities
    """
    return np.log(prob / (1 - prob))


def inverse_logit(logodds):
    """Inverse logit transformation. Returns probabilities

    Parameters
    ----------
    logodds : float, array
        A single log-odd or an array of log-odds

    Returns
    -------
    inverse-logit transformed results (i.e. probabilities for log-odds)
    """
    return 1 / (1 + np.exp(-logodds))
