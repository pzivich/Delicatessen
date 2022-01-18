import numpy as np
from delicatessen.utilities import logit, inverse_logit

#################################################################
# Basic Estimating Equations


def ee_mean(theta, y):
    r"""Default stacked estimating equation for the mean. The estimating equation for the mean is

    .. math::

        \sum_i^n \psi(Y_i, \theta_1) = \sum_i^n Y_i - \theta_1 = 0

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Parameters
    ----------
    theta : vector
        Theta in the case of the mean consists of a single value. Therefore, an initial value like the form of
        [0, ] is recommended.
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the mean).

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_mean` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean(theta=theta, y=y_dat)

    Calling the M-estimation procedure

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Output 1-by-n array
    return np.asarray(y) - theta     # Estimating equation for the mean


def ee_mean_robust(theta, y, k):
    r""" Default stacked estimating equation for robust mean (robust location) estimator. The estimating equation for
    the robust mean is

    .. math::

        \sum_i^n \psi(Y_i, \theta_1) = \sum_i^n Y^*_i - \theta_1 = 0

    where Y* is bounded between k and -k.

    Note
    ----
    Since psi is non-differentiable at k or -k, it must be assumed that the mean is sufficiently far from k. Otherwise,
    difficulties might arise in the variance calculation.


    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Parameters
    ----------
    theta : vector
        Theta in the case of the robust mean consists of a single value. Therefore, an initial value like the form of
        [0, ] is recommended.
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the robust mean).
    k : int, float
        Value to set the maximum upper and lower bounds on the observed values.

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_mean_robust` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean_robust

    Some generic data to estimate the mean for

    >>> y_dat = [-10, 1, 2, 4, 1, 2, 3, 1, 5, 2, 33]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean_robust(theta=theta, y=y_dat, k=9)

    Calling the M-estimation procedure

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Huber PJ. (1992). Robust estimation of a location parameter. In Breakthroughs in statistics (pp. 492-518).
    Springer, New York, NY.
    """
    var = np.asarray(y)                   # Convert y to NumPy array
    var = np.where(var > k, k, var)       # Apply the upper bound
    var = np.where(var < -k, -k, var)     # Apply the lower bound

    # Output 1-by-n array
    return var - theta                    # Estimating equation for robust mean


def ee_mean_variance(theta, y):
    r"""Default stacked estimating equation for mean and variance. The estimating equations for the mean and
     variance are

    .. math::

        \sum_i^n \psi_1(Y_i, \theta_1) = \sum_i^n Y_i - \theta_1 = 0

        \sum_i^n \psi_2(Y_i, \theta_1) = \sum_i^n (Y_i - \theta_1)^2 - \theta_2 = 0

    Unlike `ee_mean`, theta consists of 2 elements. The output covariance matrix will also provide estimates for each
    of the theta values.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Parameters
    ----------
    theta : vector
        Theta in this case consists of two values. Therefore, initial values like the form of [0, 0] is recommended.
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the mean).

    Returns
    -------
    array :
        Returns a 2-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_mean_variance` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean_variance

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean_variance(theta=theta, y=y_dat)

    Calling the M-estimation procedure (note that `init` has 2 values now).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    In this example, `mestimation.theta[1]` and `mestimation.asymptotic_variance[0][0]` are expected to be equal.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Output 2-by-n matrix
    return (y - theta[0],                  # Estimating equation for mean
            (y - theta[0])**2 - theta[1])  # Estimating equation for variance


def ee_positive_mean_deviation(theta, y):
    """Default stacked estimating equations for the positive mean deviation. The estimating equations are

    .. math::

        \sum_i^n \psi_1(Y_i, \theta) = \sum_i^n 2(Y_i - \theta_2)I(Y_i > \theta_2) - \theta_1 = 0

        \sum_i^n \psi_2(Y_i, \theta) = \sum_i^n 0.5 - I(Y_i \le - \theta_2) = 0

    where the first estimating equation is for the positive mean difference, and the second estimating equation is for
    the median.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : vector
        Theta in this case consists of two values. Therefore, initial values like the form of [0, 0] is recommended.
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the positive mean deviation).

    Returns
    -------
    array :
        Returns a 2-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_positive_mean_deviation` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_positive_mean_deviation

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_positive_mean_deviation(theta=theta, y=y_dat)

    Calling the M-estimation procedure (note that `init` has 2 values now).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    return ((2 * (y - theta[1]) * (y > theta[1])) - theta[0],
            1/2 - (y <= theta[1]), )


#################################################################
# Regression Estimating Equations


def ee_linear_regression(theta, X, y, weights=None):
    r"""Default stacked estimating equation for linear regression without the homoscedastic assumption. The estimating
    equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i = 0

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Here, theta corresponds to the coefficients in a linear regression model

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely OLS). Therefore, optimization of OLS via a separate functionality can be done then those
    estimated parameters are fed forward as the initial values (which should result in a more stable optimization).

    Parameters
    ----------
    theta : vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    weights : vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_linear_regression` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_linear_regression

    Some generic data to estimate a linear regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['C'] = 1

    Note that `C` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])


    Calling the M-estimation procedure (note that `init` has 3 values now, since `X.shape[1]` is equal to 3).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    X = np.asarray(X)                       # Convert to NumPy array
    y = np.asarray(y)[:, None]              # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra

    # Allowing for a weighted linear model
    if weights is None:                     # If weights is unspecified
        w = np.ones(X.shape[0])                 # ... assign weight of 1 to all observations
    else:                                   # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector

    # Output b-by-n matrix
    return w*((y -                            # Speedy matrix algebra for regression
               np.dot(X, beta))               # ... linear regression requires no transfomrations
               * X).T                         # ... multiply by coefficient and transpose for correct orientation


def ee_robust_linear_regression(theta, X, y, k, weights=None):
    """Default stacked estimating equation for robust linear regression. Specifically, robust linear regression is
    robust to outlying observations of the outcome variable (``y``). The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n \psi_k(Y_i - X_i^T \theta) X_i = 0

    where k indicates the upper and lower bounds. Here, theta is a 1-by-b array, where b is the distinct covariates
    included as part of X. For example, if X is a 3-by-n matrix, then theta will be a 1-by-3 array. The code is general
    to allow for an arbitrary number of X's (as long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Here, theta corresponds to the coefficients in a robust linear regression model

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely OLS). Therefore, optimization of OLS via a separate functionality can be done then those
    estimated parameters are fed forward as the initial values (which should result in a more stable optimization).

    Parameters
    ----------
    theta : vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    k : int, float
        Value to set the symmetric maximum upper and lower bounds on the difference between the observations and
        predicted values
    weights : vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_robust_linear_regression` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_robust_linear_regression

    Some generic data to estimate a robust linear regresion model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, scale=3, size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_robust_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'], k=3)


    Calling the M-estimation procedure (note that `init` has 3 values now, since `X.shape[1]` is equal to 3).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    X = np.asarray(X)                       # Convert to NumPy array
    y = np.asarray(y)[:, None]              # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra

    # Allowing for a weighted linear model
    if weights is None:                     # If weights is unspecified
        w = np.ones(X.shape[0])                 # ... assign weight of 1 to all observations
    else:                                   # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector

    # Generating predictions and applying Huber function for robust
    preds = y - np.dot(X, beta)
    preds_bound = np.asarray(preds)                               # Convert y to NumPy array
    preds_bound = np.where(preds_bound > k, k, preds_bound)       # Apply the upper bound
    preds_bound = np.where(preds_bound < -k, -k, preds_bound)     # Apply the lower bound

    # Output b-by-n matrix
    return w*(preds_bound            # ... linear regression requires no transformations
              * X).T                 # ... multiply by coefficient and transpose for correct orientation


def ee_logistic_regression(theta, X, y, weights=None):
    r"""Default stacked estimating equation for logistic regression. The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    where expit, or the inverse logit is

    .. math::

        expit(u) = 1 / (1 + exp(u))

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Here, theta corresponds to the coefficients in a logistic regression model, and therefore are the log-odds.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely logistic regression). Therefore, optimization of logistic regression via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).


    Parameters
    ----------
    theta : vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    weights : vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_logistic_regression` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_logistic_regression

    Some generic data to estimate a logistic regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that `C` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_logistic_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])


    Calling the M-estimation procedure (note that `init` has 3 values now, since `X.shape[1]` is equal to 3).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    X = np.asarray(X)                      # Convert to NumPy array
    y = np.asarray(y)[:, None]             # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]      # Convert to NumPy array and ensure correct shape for matrix algebra

    # Allowing for a weighted linear model
    if weights is None:                     # If weights is unspecified
        w = np.ones(X.shape[0])                 # ... assign weight of 1 to all observations
    else:                                   # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector

    # Output b-by-n matrix
    return w*((y -                                # Speedy matrix algebra for regression
               inverse_logit(np.dot(X, beta)))    # ... inverse-logit transformation of predictions
              * X).T                              # ... multiply by coefficient and transpose for correct orientation


def ee_4p_logistic(theta, X, y):
    r"""Default stacked estimating equation estimating equations for the four parameter logistic model (4PL). 4PL is
    often used for dose-response and bioassay analyses. The estimating equations are

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    Here, theta is a 1-by-4 array, where 4 are the 4 parameters of the 4PL. The first theta corresponds to lower limit,
    the second corresponds to the steepness of the curve, the third corresponds to the effective dose (ED50), and the
    fourth corresponds to the upper limit.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : vector
        Theta in this case consists of 4 values. In general, starting values >0 are better choices for the 4PL model
    X : vector
        1-dimensional vector of n dose values. No missing data should be included (missing data may cause unexpected
        behavior).
    y : vector
        1-dimensional vector of n response values. No missing data should be included (missing data may cause
        unexpected behavior).

    Returns
    -------
    array :
        Returns a 4-by-n NumPy array evaluated for the input theta, y, X

    Examples
    --------
    Construction of a estimating equation(s) with `ee_4p_logistic` should be done similar to the following

    """
    # Creating rho to cut down on typing
    rho = (X / theta[1]) ** theta[2]

    # Generalized 4PL model function for y-hat
    fx = theta[0] + (theta[3] - theta[0]) / (1 + rho)

    # Using a special implementatin of natural log here
    nested_log = np.log(X / theta[1],             # ... to avoid dose=0 issues only take log
                        where=0<X)                # ... where dose>0 (otherwise puts zero in place)

    # Calculate the derivatives for the gradient
    deriv = np.array((1 - 1/(1+rho),                                           # Gradient for lower limit
                     (theta[3]-theta[0])*theta[2]/theta[1]*rho/(1+rho)**2,     # Gradient for steepness
                     (theta[3] - theta[0]) * nested_log * rho / (1 + rho)**2,  # Gradient for ED50
                     1 / (1 + rho)), )                                         # Gradient for upper limit

    # Compute gradient and return for each i
    return -2*(y-fx)*deriv


def ee_3p_logistic(theta, X, y, lower):
    r"""Default stacked estimating equation estimating equations for the three parameter logistic model (3PL). 3PL is
    often used for dose-response and bioassay analyses. The estimating equations are

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    Here, theta is a 1-by-3 array, where 3 are the 3 parameters of the 3PL. The lower limit is specified by the user,
    and thus is no longer estimated for the 3PL. The theta's now correspond to: steepness of the curve, effective dose
    (ED50), and the upper limit.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : vector
        Theta in this case consists of 3 values. In general, starting values >0 are better choices for the 3PL model
    X : vector
        1-dimensional vector of n dose values. No missing data should be included (missing data may cause unexpected
        behavior).
    y : vector
        1-dimensional vector of n response values. No missing data should be included (missing data may cause
        unexpected behavior).
    lower : int, float
        Set value for the lower limit.

    Returns
    -------
    array :
        Returns a 3-by-n NumPy array evaluated for the input theta, y, X

    Examples
    --------
    Construction of a estimating equation(s) with `ee_3p_logistic` should be done similar to the following

    """
    # Creating rho to cut down on typing
    rho = (X / theta[0])**theta[1]

    # Generalized 3PL model function for y-hat
    fx = lower + (theta[2] - lower) / (1 + rho)

    # Using a special implementatin of natural log here
    nested_log = np.log(X / theta[0],             # ... to avoid dose=0 issues only take log
                        where=0<X)                # ... where dose>0 (otherwise puts zero in place)

    # Calculate the derivatives for the gradient
    deriv = np.array(((theta[2]-lower)*theta[1]/theta[0]*rho/(1+rho)**2,     # Gradient for steepness
                      (theta[2]-lower) * nested_log * rho / (1+rho)**2,      # Gradient for ED50
                      1 / (1 + rho)), )                                      # Gradient for upper limit

    # Compute gradient and return for each i
    return -2*(y-fx)*deriv


def ee_2p_logistic(theta, X, y, lower, upper):
    r"""Default stacked estimating equation estimating equations for the two parameter logistic model (2PL). 2PL is
    often used for dose-response and bioassay analyses. The estimating equations are

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    Here, theta is a 1-by-2 array, where 2 are the 2 parameters of the 2PL. The lower and upper limits are specified by
    the user, and thus is no longer estimated for the 2PL. The theta's now correspond to: steepness of the curve, and
    effective dose (ED50).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : vector
        Theta in this case consists of 2 values. In general, starting values >0 are better choices for the 3PL model
    X : vector
        1-dimensional vector of n dose values. No missing data should be included (missing data may cause unexpected
        behavior).
    y : vector
        1-dimensional vector of n response values. No missing data should be included (missing data may cause
        unexpected behavior).
    lower : int, float
        Set value for the lower limit.
    upper : int, float
        Set value for the upper limit.

    Returns
    -------
    array :
        Returns a 2-by-n NumPy array evaluated for the input theta, y, X

    Examples
    --------
    Construction of a estimating equation(s) with `ee_2p_logistic` should be done similar to the following

    """
    # Creating rho to cut down on typing
    rho = (X / theta[0])**theta[1]

    # Generalized 3PL model function for y-hat
    fx = lower + (upper - lower) / (1 + rho)

    # Using a special implementatin of natural log here
    nested_log = np.log(X / theta[0],             # ... to avoid dose=0 issues only take log
                        where=0<X)                # ... where dose>0 (otherwise puts zero in place)

    # Calculate the derivatives for the gradient
    deriv = np.array(((upper-lower)*theta[1]/theta[0]*rho/(1+rho)**2,     # Gradient for steepness
                      (upper-lower) * nested_log * rho / (1+rho)**2), )   # Gradient for ED50

    # Compute gradient and return for each i
    return -2*(y-fx)*deriv


def ee_effective_dose_alpha(theta, y, alpha, steepness, ed50, lower, upper):
    r"""Default stacked estimating equation to pair with the 4 parameter logistic model for estimation of the alpha
    effective dose. The estimating equation is

    .. math::

        \psi(Y_i, \theta) = \beta_1 + \frac{\beta_4 - \beta_1}{1 + (\theta / \beta_2)^{\beta_3}} - \beta_4(1-\alpha) - \beta_1 \alpha

    where theta is the ED(alpha), and the beta values are from a 4PL model (1: lower limit, 2: steepness, 3: ED(50), 4:
    upper limit). When lower or upper limits are place, the corresponding beta's are replaced by constants. For proper
    uncertainty estimation, this estimating equation should be stacked together with the correspond PL model.

    Note
    ----
    This estimating equation is meant to be paired with the estimating equations for either the 4PL, 3PL, or 2PL models.

    Parameters
    ----------
    theta : int, float
        Theta value corresponding to the ED(alpha).
    y : vector
        1-dimensional vector of n response values, used to construct correct shape for output.
    alpha : float
        The effective dose level of interest, ED(alpha).
    steepness : float
        Estimated parameter for the steepness from the PL.
    ed50 : float
        Estimated parameter for the ED50, or ED(alpha=50), from the PL.
    lower : int, float
        Estimated parameter or pre-specified constant for the lower limit. This should be a pre-specified constant for
        both the 3PL and 2PL.
    upper : int, float
        Estimated parameter or pre-specified constant for the lower limit. This should be a pre-specified constant for
        the 2PL.

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta

    Examples
    --------
    Construction of a estimating equations for ED25 with `ee_4p_logistic` should be done similar to the following

    """
    # Creating rho to cut down on typing
    rho = (theta / steepness)**ed50            # Theta is the corresponds ED(alpha) value

    # Calculating the predicted value for f(x,\theta), or y-hat
    fx = lower + (upper - lower) / (1 + rho)

    # Subtracting off (Upper*(1-alpha) + Lower*alpha) since theta should result in zeroing of quantity
    ed_alpha = fx - upper*(1-alpha) - lower*alpha

    # Returning constructed 1-by-ndarray for stacked estimating equations
    return np.ones(y.shape[0])*ed_alpha


#################################################################
# Causal Inference (ATE) Estimating Equations


def ee_gformula(theta, X, y, treat_index, force_continuous=False):
    r"""Default stacked estimating equation for the parametric g-formula in the time-fixed setting. The parameter(s) of
    interest is the average treatment effect, with potential interest in the underlying risk or mean functions. For
    continuous Y, the linear regression estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i = 0

    and for logistic regression, the estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    By default, `ee_gformula` detects whether `y` is all binary (zero or one), and applies logistic regression if that
    happens. See the parameters for more details.

    For the implementation of the g-formula, stacked estimating equations are also used for the risk / mean had
    everyone been given treatment=1, the risk / mean had everyone been given treatment=0, and the risk / mean
    difference between those two risks. Respectively, those estimating equations look like

    .. math::

        \sum_i^n \psi_1(Y_i, X_i, \theta_1) = \sum_i^n g(\hat{Y}_i) - \theta_1 = 0

    .. math::

        \sum_i^n \psi_0(Y_i, X_i, \theta_2) = \sum_i^n g(\hat{Y}_i) - \theta_2 = 0

    .. math::

        \sum_i^n \psi_0(Y_i, X_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

    Here, the function g() is a generic function. If linear regression was used, g() is the identity function. If
    logistic regression was used, g() is the expit or inverse-logit function.

    Due to these 3 extra values, the length of the theta vector is b+3, where b is the number of parameters in the
    regression model.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Here, theta corresponds to a variety of different quantities. The *first* value in theta vector is the risk / mean
    difference (or average treatment effect), the *second* is the risk / mean had everyone been given treatment=0, the
    *third* is the risk / mean had everyone been given treatment=1. The remainder of the parameters correspond to the
    regression model coefficients, in the order input.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely regression models). Therefore, optimization of the regression model via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).


    Parameters
    ----------
    theta : array, list
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    treat_index : int
        Column index for the treatment vector.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (3+b)-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_gformula` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_gformula

    Some generic confounded data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations. Note that 'A' is the treatment of interest, so `treat_index` is
    set to 1 (compared to input `X`).

    >>> def psi(theta):
    >>>     return ee_gformula(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

    Calling the M-estimation procedure. Since `X` is 3-by-n here and g-formula has 3 additional parameters, the initial
    values should be of length 3+3=6. In general, it will be best to start with [0., 0.5, 0.5, ...] as the initials for
    the risk parameters. This will start the initial at the exact middle value for each of the first 3 parameters. For
    the regression coefficients, those can be set as zero, or if there is difficulty in simultaneous optimization,
    coefficient estimates from outside `MEstimator` can be provided as inputs.

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0., 0.])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    More specifically, the average treatment effect and its variance are

    >>> mestimation.theta[0]
    >>> mestimation.variance[0, 0]

    The risk / mean had all been given treatment=1

    >>> mestimation.theta[1]
    >>> mestimation.variance[1, 1]

    The risk / mean had all been given treatment=0

    >>> mestimation.theta[2]
    >>> mestimation.variance[2, 2]

    References
    ----------
    Snowden JM, Rose S, & Mortimer KM. (2011). Implementation of G-computation on a simulated data set: demonstration
    of a causal inference technique. American Journal of Epidemiology, 173(7), 731-738.
    """
    # Ensuring correct typing
    X = np.asarray(X)                            # Convert to NumPy array
    y = np.asarray(y)                            # Convert to NumPy array
    beta = theta[3:]                       # Extracting out theta's for the regression model

    # Checking outcome variable type
    if np.isin(y, [0, 1]).all() and not force_continuous:
        continuous = False
    else:
        continuous = True

    if continuous:
        # Linear regression parameters
        preds_reg = ee_linear_regression(theta=beta,
                                         X=X,
                                         y=y)
        # Calculating Y(a=0)
        X[:, treat_index] = 0
        ya0 = np.dot(X, beta) - theta[2]
        # Calculating Y(a=1)
        X[:, treat_index] = 1
        ya1 = np.dot(X, beta) - theta[1]
    else:
        # Logistic regression parameters
        preds_reg = ee_logistic_regression(theta=beta,
                                           X=X,
                                           y=y)
        # Calculating Y(a=0)
        X[:, treat_index] = 0
        ya0 = inverse_logit(np.dot(X, beta)) - theta[2]
        # Calculating Y(a=1)
        X[:, treat_index] = 1
        ya1 = inverse_logit(np.dot(X, beta)) - theta[1]

    # Calculating Y(a=1) - Y(a=0)
    ate = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

    # Output (3+b)-by-n stacked array
    return np.vstack((ate,            # theta[0] is for the ATE
                      ya1[None, :],   # theta[1] is for R1
                      ya0[None, :],   # theta[2] is for R0
                      preds_reg))     # theta[3:] is for the regression coefficients


def ee_ipw(theta, X, y, treat_index):
    r"""Default stacked estimating equation for inverse probability weighting in the time-fixed setting. The
    parameter(s) of interest is the average treatment effect, with potential interest in the underlying risk or mean
    functions. For estimation of the weights (or propensity scores), a logistic model is used. Therefore, the first
    estimating equation is

    .. math::

        \sum_i^n \psi_g(A_i, W_i, \theta) = \sum_i^n (A_i - expit(W_i^T \theta)) W_i = 0

    where A is the treatment and W is the set of confounders. Both of these are processed from the input `X` and the
    specified `treat_index`.

    For the implementation of the inverse probability weighting estimator, stacked estimating equations are also used
    for the risk / mean had everyone been given treatment=1, the risk / mean had everyone been given treatment=0, and
    the risk / mean difference between those two risks. Respectively, those estimating equations look like

    .. math::

        \sum_i^n \psi_1(Y_i, A_i, \pi_i, \theta_1) = \sum_i^n \frac{A_i \times Y_i}{\pi_i} - \theta_1 = 0

    .. math::

        \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n \frac{(1-A_i) \times Y_i}{1-\pi_i} - \theta_2 = 0

    .. math::

        \sum_i^n \psi_d(Y_i, A_i, \pi_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

    Due to these 3 extra values, the length of the theta vector is b+3, where b is the number of parameters in the
    regression model.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Here, theta corresponds to a variety of different quantities. The *first* value in theta vector is the risk / mean
    difference (or average treatment effect), the *second* is the risk / mean had everyone been given treatment=0, the
    *third* is the risk / mean had everyone been given treatment=1. The remainder of the parameters correspond to the
    logistic regression model coefficients, in the order input.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely regression models). Therefore, optimization of the regression model via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).


    Parameters
    ----------
    theta : array, list
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    treat_index : int
        Column index for the treatment vector.

    Returns
    -------
    array :
        Returns a (3+b)-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_ipw` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_ipw

    Some generic causal data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations. Note that 'A' is the treatment of interest, so `treat_index` is
    set to 1 (compared to input `X`).

    >>> def psi(theta):
    >>>     return ee_ipw(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

    Calling the M-estimation procedure. Since `X` is 3-by-n here and g-formula has 3 additional parameters, the initial
    values should be of length 3+3=6. In general, it will be best to start with [0., 0.5, 0.5, ...] as the initials for
    the risk parameters. This will start the initial at the exact middle value for each of the first 3 parameters. For
    the regression coefficients, those can be set as zero, or if there is difficulty in simultaneous optimization,
    coefficient estimates from outside `MEstimator` can be provided as inputs.

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0., 0.])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    More specifically, the average treatment effect and its variance are

    >>> mestimation.theta[0]
    >>> mestimation.variance[0, 0]

    The risk / mean had all been given treatment=1

    >>> mestimation.theta[1]
    >>> mestimation.variance[1, 1]

    The risk / mean had all been given treatment=0

    >>> mestimation.theta[2]
    >>> mestimation.variance[2, 2]

    References
    ----------
    ... looking for a good one ...
    """
    # Ensuring correct typing
    X = np.asarray(X)                            # Convert to NumPy array
    y = np.asarray(y)                            # Convert to NumPy array
    beta = theta[3:]                             # Extracting out theta's for the regression model

    # Splitting X into A,W (treatment, covariates)
    W = np.delete(X, treat_index, axis=1)        # Extract all-but treatment col A
    A = X[:, treat_index]                        # Extract treatment col A

    # Estimating propensity score
    preds_reg = ee_logistic_regression(theta=beta,    # Using logistic regression
                                       X=W,           # Plug-in covariates for X
                                       y=A)           # Plug-in treatment for Y

    # Estimating weights
    pi = inverse_logit(np.dot(W, beta))          # Getting Pr(A|W) from model

    # Calculating Y(a=1)
    ya1 = (A * y) / pi - theta[1]                # i's contribution is (AY) / \pi

    # Calculating Y(a=0)
    ya0 = ((1-A) * y) / (1-pi) - theta[2]        # i's contribution is ((1-A)Y) / (1-\pi)

    # Calculating Y(a=1) - Y(a=0)
    ate = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

    # Output (3+b)-by-n stacked array
    return np.vstack((ate,             # theta[0] is for the ATE
                      ya1[None, :],    # theta[1] is for R1
                      ya0[None, :],    # theta[2] is for R0
                      preds_reg))      # theta[3:] is for the regression coefficients


def ee_aipw(theta, X, y, treat_index, force_continuous=False):
    r"""Default stacked estimating equation for augmented inverse probability weighting in the time-fixed setting. The
    parameter(s) of interest is the average treatment effect, with potential interest in the underlying risk or mean
    functions. For estimation of the weights (or propensity scores), a logistic model is used. Therefore, the first
    estimating equation is

    .. math::

        \sum_i^n \psi_g(A_i, W_i, \theta) = \sum_i^n (A_i - expit(W_i^T \theta)) W_i = 0

    where A is the treatment and W is the set of confounders. Both of these are processed from the input `X` and the
    specified `treat_index`.

    Next, an outcome model is specified. For continuous Y, the linear regression estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i = 0

    and for logistic regression, the estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    By default, `ee_aipw` detects whether `y` is all binary (zero or one), and applies logistic regression if that
    happens. See the parameters for more details.

    For the implementation of the augmented inverse probability weighting estimator, stacked estimating equations are
    also used for the risk / mean had everyone been given treatment=1, the risk / mean had everyone been given
    treatment=0, and the risk / mean difference between those two risks. Respectively, those estimating equations
    look like

    .. math::

        \sum_i^n \psi_1(Y_i, A_i, W_i, \pi_i, \theta_1) = \sum_i^n (\frac{A_i \times Y_i}{\pi_i} - \frac{\hat{Y^1}(A_i-\pi_i}{\pi_i}) - \theta_1 = 0

    .. math::

        \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n (\frac{(1-A_i) \times Y_i}{1-\pi_i} + \frac{\hat{Y^0}(A_i-\pi_i}{1-\pi_i})) - \theta_2 = 0

    .. math::

        \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

    where Y^a is the predicted values of Y from the outcome model under treatment assignment A=a.

    Due to these 3 extra values and two nuisance models, the length of the theta vector is b+(b-1)+3, where b is the
    number of covariats included in `X`.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.


    Here, theta corresponds to a variety of different quantities. The *first* value in theta vector is the risk / mean
    difference (or average treatment effect), the *second* is the risk / mean had everyone been given treatment=0, the
    *third* is the risk / mean had everyone been given treatment=1. The remainder of the parameters correspond to the
    regression model coefficients, in the order input. The first 'chuck' of coefficients correspond to the outcome model
    and the last 'chuck' correspond to the treatment model.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely regression models). Therefore, optimization of the regression model via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).


    Parameters
    ----------
    theta : array, list
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    treat_index : int
        Column index for the treatment vector.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (3+b+(b-1))-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_aipw` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_aipw

    Some generic causal data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations. Note that 'A' is the treatment of interest, so `treat_index` is
    set to 1 (compared to input `X`).

    >>> def psi(theta):
    >>>     return ee_aipw(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

    Calling the M-estimation procedure. Since `X` is 3-by-n here and g-formula has 3 additional parameters, the initial
    values should be of length 3+3=6. In general, it will be best to start with [0., 0.5, 0.5, ...] as the initials for
    the risk parameters. This will start the initial at the exact middle value for each of the first 3 parameters. For
    the regression coefficients, those can be set as zero, or if there is difficulty in simultaneous optimization,
    coefficient estimates from outside `MEstimator` can be provided as inputs.

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0., 0., 0., 0.])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    More specifically, the average treatment effect and its variance are

    >>> mestimation.theta[0]
    >>> mestimation.variance[0, 0]

    The risk / mean had all been given treatment=1

    >>> mestimation.theta[1]
    >>> mestimation.variance[1, 1]

    The risk / mean had all been given treatment=0

    >>> mestimation.theta[2]
    >>> mestimation.variance[2, 2]

    References
    ----------
    Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust estimation of causal
    effects. *American Journal of Epidemiology*, 173(7), 761-767.

    Tsiatis AA. (2006). Semiparametric theory and missing data. Springer, New York, NY.
    """
    # Giving parameters short-hand names for simplicity of code
    diff = theta[0]                 # This is the ATE
    r1 = theta[1]                   # This is R1
    r0 = theta[2]                   # This is R0
    beta = theta[3:3+X.shape[1]]    # These are the regression coefficients for the outcome model
    alpha = theta[3+X.shape[1]:]    # These are the regression coefficients for the treatment model

    # Ensuring correct typing
    X = np.asarray(X)               # Convert to NumPy array
    y = np.asarray(y)               # Convert to NumPy array

    # Splitting X into A,W (treatment, covariates)
    W = np.delete(X, treat_index, axis=1)             # Extract all-but treatment col A
    A = X[:, treat_index].copy()                      # Extract treatment col A (MUST BE A COPY)

    # pi-model (logistic regression)
    pi_model = ee_logistic_regression(theta=alpha,    # Estimating logistic model
                                      X=W,
                                      y=A)
    pi = inverse_logit(np.dot(W, alpha))              # Estimating the predicted probabilities for pi

    # m-model (logistic regression)
    if np.isin(y, [0, 1]).all() and not force_continuous:      # Checking outcome variable type is binary
        m_model = ee_logistic_regression(theta=beta,           # Logistic regression parameters
                                         X=X,
                                         y=y)
        X[:, treat_index] = 1                                  # Setting treatment to A=1 for all
        ya1 = inverse_logit(np.dot(X, beta))                   # Calculating \hat{Y}(a=1)
        X[:, treat_index] = 0                                  # Setting treatment to A=0 for all
        ya0 = inverse_logit(np.dot(X, beta))                   # Calculating \hat{Y}(a=1)
    else:
        m_model = ee_linear_regression(theta=beta,             # Linear regression parameters
                                       X=X,
                                       y=y)
        X[:, treat_index] = 1                                  # Setting treatment to A=1 for all
        ya1 = np.dot(X, beta)                                  # Calculating \hat{Y}(a=1)
        X[:, treat_index] = 0                                  # Setting treatment A=0 for all
        ya0 = np.dot(X, beta)                                  # Calculating \hat{Y}(a=0)

    # AIPW estimator
    y1_star = (y*A/pi - ya1*(A-pi)/pi) - r1                    # Calculating \tilde{Y}(a=1)
    y0_star = (y*(1-A)/(1-pi) + ya0*(A-pi)/(1-pi)) - r0        # Calculating \tilde{Y}(a=0)
    ate = np.ones(y.shape[0]) * (r1 - r0) - diff               # Calculating the ATE

    # Output (3+b+(b-1))-by-n array
    return np.vstack((ate,               # theta[0] is for the ATE
                      y1_star[None, :],  # theta[1] is for R1
                      y0_star[None, :],  # theta[2] is for R0
                      m_model,           # theta[3:b] is for the outcome model coefficients
                      pi_model))         # theta[b:] is for the treatment model coefficients
