import numpy as np


def generate_weights(weights, n_obs):
    """Internal use function to return the weights assigned to each observation. Returns a vector of 1's when no
    weights are provided. Otherwise, converts provided vector into a numpy array.

    Parameters
    ----------
    weights : None, ndarray, list
        Vector of weights, or None if no weights are provided
    n_obs : int
        Number of observations in the data

    Returns
    -------
    ndarray
    """
    if weights is None:                     # If weights is unspecified
        w = np.ones(n_obs)                      # ... assign weight of 1 to all observations
    else:                                   # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector
    return w
