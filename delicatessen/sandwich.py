import warnings

import numpy as np
from scipy.optimize import approx_fprime

from delicatessen.derivative import auto_differentiation, approx_differentiation


def compute_sandwich(stacked_equations, theta, deriv_method='approx', dx=1e-9, allow_pinv=True):
    """Compute the empirical sandwich variance estimator given [...] and parameter estimates.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a b-by-n NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    theta : list, set, array
        Parameter estimates to compute the empirical sandwich variance estimator at. Note that this function assumes
        that you have solved for the ``theta`` that correspond to the root of the input estimating equations.
    deriv_method : str, optional
        Method to compute the derivative of the estimating equations for the bread matrix. Options include numerical
        approximation via the central difference method (``'approx'``) and forward-mode automatic differentiation
        (``'exact'``). Default is ``'approx'``.
    dx : float, optional
        Spacing to use to numerically approximate the partial derivatives of the bread matrix. Here, a small value
        for ``dx`` should be used, since some large values can result in poor approximations. This argument is only
        used when ``deriv_method='approx'``. Default is 1e-9.
    allow_pinv : bool, optional
        Whether to allow for the pseudo-inverse (via ``numpy.linalg.pinv``) if the bread matrix is determined to be
        non-invertible. If you want to disallow the pseudo-inverse (i.e., use ``numpy.linalg.inv``), set this
        argument to ``False``. Default is ``True``, which  is more robust to the possible bread matrices.

    Returns
    -------

    """
    # Evaluating at provided theta values
    evald_theta = np.asarray(stacked_equations(theta=theta))        # Evaluating EE at theta-hat
    if len(theta) == 1:                                             #
        n_obs = evald_theta.shape[0]                                # Number of observations
    else:                                                           #
        n_obs = evald_theta.shape[1]                                # Number of observations

    # Step 1: Compute the bread matrix
    bread = compute_bread(stacked_equations=stacked_equations,      #
                          theta=theta,                              # Provide theta-hat
                          deriv_method=deriv_method,                # Method to use
                          dx=dx)                                    #
    bread = bread / n_obs                                           #

    # Step 2: Compute the meat matrix
    meat = compute_meat(stacked_equations=stacked_equations,        #
                        theta=theta)                                #
    meat = meat / n_obs                                             # Meat is dot product of arrays

    # Step 3: Construct sandwich from the bread and meat matrices
    sandwich = build_sandwich(bread=bread,                          #
                              meat=meat,                            #
                              allow_pinv=allow_pinv)                #

    # Return the constructed empirical sandwich variance estimator
    return sandwich


def compute_bread(stacked_equations, theta, deriv_method, dx=1e-9):
    """

    Parameters
    ----------
    stacked_equations
    theta
    deriv_method
    dx

    Returns
    -------

    """
    def estimating_equation(input_theta):
        if len(input_theta) == 1:
            return np.sum(stacked_equations(theta=input_theta))
        else:
            return np.sum(stacked_equations(theta=input_theta), axis=1)

    # Compute the derivative
    if deriv_method.lower() == 'approx':
        bread_matrix = approx_fprime(xk=theta,
                                     f=estimating_equation,
                                     epsilon=dx)
        if len(theta) == 1:
            bread_matrix = np.asarray([bread_matrix, ])
    elif deriv_method.lower() == 'capprox':
        bread_matrix = approx_differentiation(xk=theta,
                                              f=estimating_equation,
                                              method='capprox',
                                              epsilon=dx)
    elif deriv_method.lower() == 'fapprox':
        bread_matrix = approx_differentiation(xk=theta,
                                              f=estimating_equation,
                                              method='fapprox',
                                              epsilon=dx)
    elif deriv_method.lower() == 'bapprox':
        bread_matrix = approx_differentiation(xk=theta,
                                              f=estimating_equation,
                                              method='bapprox',
                                              epsilon=dx)
    elif deriv_method.lower() == "exact":  # Automatic Differentiation
        bread_matrix = auto_differentiation(xk=theta,  # Compute the exact derivative at theta
                                            f=estimating_equation)  # ... for the given estimating equations
    else:
        raise ValueError("The input for deriv_method was "
                         + str(deriv_method)
                         + ", but only 'approx', 'fapprox', 'capprox', 'bapprox' "
                           "and 'exact' are available.")

    # Checking for an issue when trying to invert the bread matrix
    if np.isnan(bread_matrix).any():
        warnings.warn("The bread matrix contains at least one np.nan, so it cannot be inverted. The variance will "
                      "not be calculated. This may be an issue with the provided estimating equations or the "
                      "evaluated theta.",
                      UserWarning)

    # Returning the constructed bread matrix according to SB 2002
    return -1 * bread_matrix


def compute_meat(stacked_equations, theta):
    """

    Parameters
    ----------
    stacked_equations
    theta

    Returns
    -------

    """
    evald_theta = np.asarray(stacked_equations(theta=theta))  # Evaluating EE at theta-hat
    return np.dot(evald_theta, evald_theta.T)


def build_sandwich(bread, meat, allow_pinv):
    """

    Parameters
    ----------
    bread
    meat
    allow_pinv

    Returns
    -------

    """
    if np.any(np.isnan(bread)):
        return np.nan

    if allow_pinv:  # Support 1D theta-hat
        bread_invert = np.linalg.pinv(bread)  # ... find pseudo-inverse
    else:  # Support 1D theta-hat
        bread_invert = np.linalg.inv(bread)  # ... find inverse
    sandwich = np.dot(np.dot(bread_invert, meat), bread_invert.T)  # Compute sandwich
    return sandwich
