# Custom functions to check for errors
import numpy as np


def check_alpha_level(alpha):
    """Error checking function to verify :math:`alpha` is between [0,1].

    Parameters
    ----------
    alpha : float
        User-provided alpha value

    Returns
    -------
    None
    """
    if not 0 < alpha < 1:
        raise ValueError("`alpha` must be 0 < a < 1")


def check_variance_is_not_none(variance):
    """Error checking function to verify variance was successfully estimated.

    Parameters
    ----------
    variance : float, ndarray, None
        Internally computed variance

    Returns
    -------
    None
    """
    if variance is None:
        raise ValueError("Either theta has not been estimated yet, or there is a np.nan in the bread matrix. "
                         "Therefore, confidence_intervals() cannot be called.")


def check_penalty_shape(theta, penalty, center):
    """Error checking function to verify provided penalty terms are valid.

    Parameters
    ----------
    theta : float, ndarray
        Parameters to be estimate for a regression model
    penalty : ndarray
        User-provided penalty terms
    center : ndarray
        User-provided place to penalize parameters towards

    Returns
    -------
    None
    """
    # Checking the penalty term is correct shape
    if penalty.size != 1:
        if penalty.shape[0] != len(theta):
            raise ValueError("The penalty term must be either a single number or the same length as theta.")

    # Checking the penalty center is correct shape
    if center.size != 1:
        if center.shape[0] != len(theta):
            raise ValueError("The center term must be either a single number or the same length as theta.")

    # Checking the penalty term is non-negative
    if np.any(penalty < 0):
        raise ValueError("All penalty terms must be non-negative")