import numpy as np
from delicatessen.utilities import logit, inverse_logit, identity

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
    theta : ndarray, list, vector
        Theta in the case of the mean consists of a single value. Therefore, an initial value like the form of
        [0, ] is recommended.
    y : ndarray, list, vector
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
    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Output 1-by-n array estimating equation for the mean
    return y_array - theta


def ee_mean_robust(theta, y, k):
    r""" Default stacked estimating equation for robust mean (location) estimator. The estimating equation for the
    robust mean is

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
    theta : ndarray, list, vector
        Theta in the case of the robust mean consists of a single value. Therefore, an initial value like the form of
        [0, ] is recommended.
    y : ndarray, vector, list
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
    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Bounding via np.clip
    y_bound = np.clip(y_array, a_min=-k, a_max=k)

    # Output 1-by-n array estimating equation for robust mean
    return y_bound - theta


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
    theta : ndarray, list, vector
        Theta in this case consists of two values. Therefore, initial values like the form of [0, 0] is recommended.
    y : ndarray, list, vector
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

    For this estimating equation, `mestimation.theta[1]` and `mestimation.asymptotic_variance[0][0]` are expected to
    always be equal.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Output 2-by-n matrix of estimating equations
    return (y_array - theta[0],                  # Estimating equation for mean
            (y_array - theta[0])**2 - theta[1])  # Estimating equation for variance


def ee_percentile(theta, y, q):
    r"""Default stacked estimating equation for percentiles (or quantiles).

    Note
    ----
    Due to this estimating equation being non-smooth, estimated percentile values may differ from the closed-form
    definition of the percentile. In general, closed form solutions for percentiles will be preferred, but this
    estimating equation is offered for completeness.

    .. math::

        \sum_i^n \psi_q(Y_i, \theta_q) = \sum_i^n q - I(Y_i \le \theta_q) = 0

    Notice that this estimating equation is non-smooth. Therefore, optimization and numerically approximating
    derivatives for this estimating equation are more difficult.

    Note
    ----
    The optional parameters `MEstimator.estimate()` may benefit from the following changes `solver='hybr'`,
    `tolerance=1e-5`, `dx=1`, and `order=15`. Try a few different values.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of two values. Therefore, initial values like the form of [0, 0] is recommended.
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the mean).
    q : float
        Percentile to calculate. Must be (0, 1)

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_percentile` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_percentile

    Some generic data to estimate the mean for

    >>> np.random.seed(89041)
    >>> y_dat = np.random.normal(size=100)

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_percentile(theta=theta, y=y_dat, q=0.5)

    Calling the M-estimation procedure (note that `init` has 2 values now).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, ])
    >>> mestimation.estimate(solver='hybr', tolerance=1e-3, dx=1, order=15)

    Notice that we use a different solver, tolerance values, and parameters for numerically approximating the derivative
    here. These changes generally work better for percentile optimizations since the estimating equation is non-smooth.
    Furthermore, optimization is hard when only a few observations (<100) are available. In general, closed form
    solutions for percentiles will be preferred.

    >>> mestimation.theta

    Then displays the estimated percentile / median. In this example, there is a difference between the closed form
    solution (``-0.07978``) and M-Estimation (``-0.06022``). Again, this results from the non-smooth estimating
    equation.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    if q >= 1 or q <= 0:
        raise ValueError("`q` must be (0, 1)")

    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Output 1-by-n array of the estimating equations
    return q - 1*(y_array <= theta)


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
    theta : ndarray, list, vector
        Theta in this case consists of two values. Therefore, initial values like the form of [0, 0] is recommended.
    y : ndarray, list, vector
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
    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Calculating median with built-in estimating equation
    median = ee_percentile(theta=theta[1], y=y_array, q=0.5)

    # Output 2-by-n matrix of estimating equations
    return ((2*(y_array - theta[1])*(y_array > theta[1])) - theta[0],   # Estimating equation for positive mean dev
            median, )                                                   # Estimating equation for median


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
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    weights : ndarray, list, vector, None, optional
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
    return w*((y -                   # Speedy matrix algebra for regression
               np.dot(X, beta))      # ... linear regression requires no transformations
              * X).T                 # ... multiply by coefficient and transpose for correct orientation


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
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    k : int, float
        Value to set the symmetric maximum upper and lower bounds on the difference between the observations and
        predicted values
    weights : ndarray, list, vector, None, optional
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
    preds = np.clip(y - np.dot(X, beta), -k, k)

    # Output b-by-n matrix
    return w*(preds                  # ... linear regression requires no transformations
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
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    weights : ndarray, list, vector, None, optional
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


#################################################################
# Dose-Response Estimating Equations


def ee_4p_logistic(theta, X, y):
    r"""Default stacked estimating equation estimating equations for the four parameter logistic model (4PL). 4PL is
    often used for dose-response and bioassay analyses. The estimating equations are

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    Here, theta is a 1-by-4 array, where 4 are the 4 parameters of the 4PL. The first theta corresponds to lower limit,
    the second corresponds to the effective dose (ED50), the third corresponds to the steepness of the curve, and the
    fourth corresponds to the upper limit.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 4 values. In general, starting values >0 are better choices for the 4PL model
    X : ndarray, list, vector
        1-dimensional vector of n dose values. No missing data should be included (missing data may cause unexpected
        behavior).
    y : ndarray, list, vector
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
                        where=0 < X)              # ... where dose>0 (otherwise puts zero in place)

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

    Here, theta is a 1-by-3 array, where 3 are the 3 parameters of the 3PL. The first theta corresponds to the
    effective dose (ED50), the second corresponds to the steepness of the curve, and the third corresponds to the upper
    limit. The lower limit is pre-specified by the user (and is no longer estimated)

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 3 values. In general, starting values >0 are better choices for the 3PL model
    X : ndarray, list, vector
        1-dimensional vector of n dose values. No missing data should be included (missing data may cause unexpected
        behavior).
    y : ndarray, list, vector
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

    # Using a special implementation of natural log here
    nested_log = np.log(X / theta[0],             # ... to avoid dose=0 issues only take log
                        where=0 < X)              # ... where dose>0 (otherwise puts zero in place)

    # Calculate the derivatives for the gradient
    deriv = np.array(((theta[2]-lower)*theta[1]/theta[0]*rho/(1+rho)**2,     # Gradient for steepness
                      (theta[2]-lower) * nested_log * rho / (1+rho)**2,      # Gradient for ED50
                      1 / (1 + rho)), )                                      # Gradient for upper limit

    # Compute gradient and return for each i
    return -2*(y - fx)*deriv


def ee_2p_logistic(theta, X, y, lower, upper):
    r"""Default stacked estimating equation estimating equations for the two parameter logistic model (2PL). 2PL is
    often used for dose-response and bioassay analyses. The estimating equations are

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    Here, theta is a 1-by-2 array, where 2 are the 2 parameters of the 2PL. The first theta corresponds to the
    effective dose (ED50), and the second corresponds to the steepness of the curve. Both the lower limit and upper
    limit are pre-specified by the user (and no longer estimated).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 2 values. In general, starting values >0 are better choices for the 3PL model
    X : ndarray, list, vector
        1-dimensional vector of n dose values. No missing data should be included (missing data may cause unexpected
        behavior).
    y : ndarray, list, vector
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
                        where=0 < X)              # ... where dose>0 (otherwise puts zero in place)

    # Calculate the derivatives for the gradient
    deriv = np.array(((upper-lower)*theta[1]/theta[0]*rho/(1+rho)**2,     # Gradient for steepness
                      (upper-lower) * nested_log * rho / (1+rho)**2), )   # Gradient for ED50

    # Compute gradient and return for each i
    return -2*(y-fx)*deriv


def ee_effective_dose_delta(theta, y, delta, steepness, ed50, lower, upper):
    r"""Default stacked estimating equation to pair with the 4 parameter logistic model for estimation of the
    :math:`delta` effective dose. The estimating equation is

    .. math::

        \psi(Y_i, \theta) = \beta_1 + \frac{\beta_4 - \beta_1}{1 + (\theta / \beta_2)^{\beta_3}} - \beta_4(1-\delta)
        - \beta_1 \delta

    where theta is the :math:`ED(\delta)`, and the beta values are from a 4PL model (1: lower limit, 2: steepness,
    3: ED(50), 4: upper limit). When lower or upper limits are place, the corresponding beta's are replaced by
    constants. For proper uncertainty estimation, this estimating equation should be stacked together with the
    correspond PL model.

    Note
    ----
    This estimating equation is meant to be paired with the estimating equations for either the 4PL, 3PL, or 2PL models.

    Parameters
    ----------
    theta : int, float
        Theta value corresponding to the ED(alpha).
    y : ndarray, list, vector
        1-dimensional vector of n response values, used to construct correct shape for output.
    delta : float
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

    # Subtracting off (Upper*(1-delta) + Lower*delta) since theta should result in zeroing of quantity
    ed_delta = fx - upper*(1-delta) - lower*delta

    # Returning constructed 1-by-ndarray for stacked estimating equations
    return np.ones(np.asarray(y).shape[0])*ed_delta


#################################################################
# Causal Inference (ATE) Estimating Equations


def ee_gformula(theta, y, X, X1, X0=None, force_continuous=False):
    r"""Default stacked estimating equation for parametric g-computation in the time-fixed setting. The parameter of
    interest can either be the mean under a single interventions or plans on an action, or the mean difference between
    two interventions or plans on an action. This is accomplished by providing the estimating equation the observed
    data (``X``, ``y``), and the same data under the actions (``X1`` and optionally ``X0``).

    For continuous Y, the linear regression estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i = 0

    and for logistic regression, the estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    By default, `ee_gformula` detects whether `y` is all binary (zero or one), and applies logistic regression if that
    is evaluated to be true. See the parameters for further details.

    There are two variations on the parameter of interest. The first could be the mean under a plan, where the plan sets
    the values of action :math:`A` (e.g., exposure, treatment, vaccination, etc.). The estimating equation for this
    causal mean is

    .. math::

        \sum_i^n \psi_1(Y_i, X_i, \theta_1) = \sum_i^n g(\hat{Y}_i) - \theta_1 = 0

    Here, the function :math:`g(.)` is a generic function. If linear regression was used, :math:`g(.)` is the identity
    function. If logistic regression was used, :math:`g(.)` is the expit or inverse-logit function.

    Note
    ----
    This variation includes :math:`1+b` parameters, where the first parameter is the causal mean, and the remainder are
    the parameters for the regression model.

    The alternative parameter of interest could be the mean difference between two plans. A common example of this would
    be the average causal effect, where the plans are all-action-one versus all-action-zero. Therefore, the estimating
    equations consist of the following three equations

    .. math::

        \sum_i^n \psi_0(Y_i, X_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

        \sum_i^n \psi_1(Y_i, X_i, \theta_1) = \sum_i^n g(\hat{Y}_i) - \theta_1 = 0

        \sum_i^n \psi_0(Y_i, X_i, \theta_2) = \sum_i^n g(\hat{Y}_i) - \theta_2 = 0


    Note
    ----
    This variation includes :math:`3+b` parameters, where the first parameter is the causal mean difference, the second
    is the causal mean under plan 1, the third is the causal mean under plan 0, and the remainder are the parameters
    for the regression model.

    The parameter of interest is designated by the user via whether the optional argument ``X0`` is left as ``None``
    (which estimates the causal mean) or is given an array (which estimates the causal mean difference and the
    corresponding causal means).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    See the examples below for how action plans are specified.

    Parameters
    ----------
    theta : ndarray, list, vector
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    y : ndarray, list, vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    X1 : ndarray, list, vector
        2-dimensional vector of n observed values for b variables under the action plan. If the action is indicated by
        ``A``, then ``X1`` will take the original data ``X`` and update the values of ``A`` to follow the deterministic
        plan. No missing data should be included (missing data may cause unexpected behavior).
    X0 : ndarray, list, vector, None, optional
        2-dimensional vector of n observed values for b variables under the action plan. This second argument is
        optional and should be specified if a causal mean difference between two action plans is of interest. If the
        action is indicated by ``A``, then ``X0`` will take the original data ``X`` and update the values of ``A`` to
        follow the deterministic reference plan. No missing data should be included (missing data may cause unexpected
        behavior).
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (1+b)-by-n NumPy array if ``X0=None``, or returns a (3+b)-by-n NumPy array if ``X0!=None``

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

    In the first example, we will estimate the causal mean had everyone been set to ``A=1``. Therefore, the optional
    argument ``X0`` is left as ``None``. Before creating the estimating equation, we need to do some data prep. First,
    we will create an interaction term between ``A`` and ``W`` in the original data. Then we will generate a copy of
    the data and update the values of ``A`` to be all ``1``.

    >>> d['AW'] = d['A']*d['W']
    >>> d1 = d.copy()
    >>> d1['A'] = 1
    >>> d1['AW'] = d1['A']*d1['W']

    Having setup our data, we can now define the psi function.

    >>> def psi(theta):
    >>>     return ee_gformula(theta,
    >>>                        y=d['Y'],
    >>>                        X=d[['C', 'A', 'W', 'AW']],
    >>>                        X1=d1[['C', 'A', 'W', 'AW']])

    Notice that ``y`` corresponds to the observed outcomes, ``X`` corresponds to the observed covariate data, and ``X1``
    corresponds to the covariate data *under the action plan*.

    Now we can call the M-Estimation procedure. Since we are estimating the causal mean, and the regression parameters,
    the length of the initial values needs to correspond with this. Our linear regression model consists of 4
    coefficients, so we need 1+4=5 initial values. When the outcome is binary (like it is in this example), we can be
    nice to the optimizer and give it a starting value of 0.5 for the causal mean (since 0.5 is in the middle of that
    distribution). Below is the call to ``MEstimator``

    >>> estr = MEstimator(psi, init=[0.5, 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the causal mean is

    >>> estr.theta[0]

    Continuing from the previous example, let's say we wanted to estimate the average causal effect. Therefore, we want
    to contrast two plans (all ``A=1`` versus all ``A=0``). As before, we need to create the reference data for ``X0``

    >>> d0 = d.copy()
    >>> d0['A'] = 0
    >>> d0['AW'] = d0['A']*d0['W']

    Having setup our data, we can now define the psi function.

    >>> def psi(theta):
    >>>     return ee_gformula(theta,
    >>>                        y=d['Y'],
    >>>                        X=d[['C', 'A', 'W', 'AW']],
    >>>                        X1=d1[['C', 'A', 'W', 'AW']],
    >>>                        X0=d0[['C', 'A', 'W', 'AW']], )

    Notice that ``y`` corresponds to the observed outcomes, ``X`` corresponds to the observed covariate data, ``X1``
    corresponds to the covariate data under ``A=1``, and ``X0`` corresponds to the covariate data under ``A=0``. Here,
    we need 3+4=7 starting values, since there are two additional parameters from the previous example. For the
    difference, a starting value of 0 is generally a good choice. Since ``Y`` is binary, we again provide 0.5 as
    starting values for the causal means

    >>> estr = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates

    >>> estr.theta[0]    # causal mean difference of 1 versus 0
    >>> estr.theta[1]    # causal mean under X1
    >>> estr.theta[2]    # causal mean under X0
    >>> estr.theta[3:]   # logistic regression coefficients

    References
    ----------
    Snowden JM, Rose S, & Mortimer KM. (2011). Implementation of G-computation on a simulated data set: demonstration
    of a causal inference technique. American Journal of Epidemiology, 173(7), 731-738.

    Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
    *Journal of Epidemiology & Community Health*, 60(7), 578-586.
    """
    # Ensuring correct typing
    X = np.asarray(X)                # Convert to NumPy array
    y = np.asarray(y)                # Convert to NumPy array
    X1 = np.asarray(X1)              # Convert to NumPy array

    # Error checking for misaligned shapes
    if X.shape != X1.shape:
        raise ValueError("The dimensions of X and X1 must be the same.")

    # Processing data depending on if two plans were specified
    if X0 is None:                   # If no reference was specified
        mu1 = theta[0]                  # ... only a single mean
        beta = theta[1:]                # ... immediately followed by the regression parameters
    else:                            # Otherwise difference and both plans are to be returned
        X0 = np.asarray(X0)             # ... reference data to NumPy array
        if X.shape != X0.shape:         # ... error checking for misaligned shapes
            raise ValueError("The dimensions of X and X0 must be the same.")
        mud = theta[0]                  # ... first parameter is mean difference
        mu1 = theta[1]                  # ... second parameter is mean under X1
        mu0 = theta[2]                  # ... third parameter is mean under X0
        beta = theta[3:]                # ... remainder are for the regression model

    # Checking outcome variable type
    if np.isin(y, [0, 1]).all() and not force_continuous:
        regression = ee_logistic_regression         # Use a logistic regression model
        transform = inverse_logit                   # ... and need to inverse-logit transformation
    else:
        regression = ee_linear_regression           # Use a linear regression model
        transform = identity                        # ... and need to apply the identity (no) transformation

    # Estimating regression parameters
    preds_reg = regression(theta=beta,              # beta coefficients
                           X=X, y=y)                # along with observed X and observed y

    # Calculating mean under X1
    ya1 = transform(np.dot(X1, beta)) - mu1         # mean under X1

    if X0 is None:                                  # if no X0, then nothing left to do
        # Output (1+b)-by-n stacked array
        return np.vstack((ya1[None, :],     # theta[0] is the mean under X1
                          preds_reg))       # theta[1:] is the regression coefficients
    else:                                           # if X0, then need to predict mean under X0 and difference
        # Calculating mean under X0
        ya0 = transform(np.dot(X0, beta)) - mu0
        # Calculating mean difference between X1 and X0
        ace = np.ones(y.shape[0])*(mu1 - mu0) - mud
        # Output (3+b)-by-n stacked array
        return np.vstack((ace,            # theta[0] is the mean difference between X1 and X0
                          ya1[None, :],   # theta[1] is the mean under X1
                          ya0[None, :],   # theta[2] is the mean under X0
                          preds_reg))     # theta[3:] is for the regression coefficients


def ee_ipw(theta, y, A, X, truncate=None):
    r"""Default stacked estimating equation for inverse probability weighting in the time-fixed setting. The
    parameter of interest is the average causal effect. For estimation of the weights (or propensity scores), a
    logistic model is used.

    Note
    ----
    Unlike the g-formula, ``ee_ipw`` only focuses on the average causal effect (and also outputs the causal means for
    ``A=1`` and ``A=0``. In other words, the implementation of IPW does not support generic action plans off-the-shelf,
    unlike ``ee_gformula``.

    The first estimating equation for the logistic regression model is

    .. math::

        \sum_i^n \psi_g(A_i, W_i, \theta) = \sum_i^n (A_i - expit(W_i^T \theta)) W_i = 0

    where A is the treatment and W is the set of confounders.

    For the implementation of the inverse probability weighting estimator, stacked estimating equations are used
    for the mean had everyone been set to ``A=1``, the mean had everyone been set to ``A=0``, and the mean difference
    between the two causal means. The estimating equations are

    .. math::

        \sum_i^n \psi_d(Y_i, A_i, \pi_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

        \sum_i^n \psi_1(Y_i, A_i, \pi_i, \theta_1) = \sum_i^n \frac{A_i \times Y_i}{\pi_i} - \theta_1 = 0

        \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n \frac{(1-A_i) \times Y_i}{1-\pi_i} - \theta_2 = 0


    Due to these 3 extra values, the length of the theta vector is 3+b, where b is the number of parameters in the
    regression model.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Here, theta corresponds to a variety of different quantities. The *first* value in theta vector is the mean
    difference (or average causal effect), the *second* is the mean had everyone been set to ``A=1``, the *third* is the
    mean had everyone been set to ``A=0``. The remainder of the parameters correspond to the logistic regression model
    coefficients.

    Parameters
    ----------
    theta : ndarray, list, vector
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause
        unexpected behavior).
    A : ndarray, list, vector
        1-dimensional vector of n observed values. The A values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables to model the probability of ``A`` with. No missing
        data should be included (missing data may cause unexpected behavior).
    truncate : None, list, set, optional
        Bounds to truncate the estimated probabilities of ``A`` at. For example, estimated probabilities above 0.99 or
        below 0.01 can be set to 0.99 or 0.01, respectively. This is done by specifying ``truncate=(0.01, 0.99)``. Note
        this step is done via ``numpy.clip(.., a_min=truncate[0], a_max=truncate[1])``, so order is important. Default
        is None, which applies to no truncation.

    Returns
    -------
    array :
        Returns a (3+b)-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_ipw`` should be done similar to the following

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

    Defining psi, or the stacked estimating equations. Note that 'A' is the action.

    >>> def psi(theta):
    >>>     return ee_ipw(theta, y=d['Y'], A=d['A'],
    >>>                   X=d[['C', 'W']])

    Calling the M-estimation procedure. Since `X` is 2-by-n here and IPW has 3 additional parameters, the initial
    values should be of length 3+2=5. In general, it will be best to start with [0., 0.5, 0.5, ...] as the initials when
    ``Y`` is binary. Otherwise, starting with all 0. as initials is reasonable.

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the corresponding parameters are

    >>> estr.theta[0]    # causal mean difference of 1 versus 0
    >>> estr.theta[1]    # causal mean under X1
    >>> estr.theta[2]    # causal mean under X0
    >>> estr.theta[3:]   # logistic regression coefficients

    If you want to see how truncating the probabilities works, try repeating the above code but specifying
    ``truncate=(0.1, 0.9)`` as an optional argument in ``ee_ipw``.

    References
    ----------
    Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
    *Journal of Epidemiology & Community Health*, 60(7), 578-586.

    Cole SR, & Hernán MA. (2008). Constructing inverse probability weights for marginal structural models.
    *American Journal of Epidemiology*, 168(6), 656-664.
    """
    # Ensuring correct typing
    X = np.asarray(X)                            # Convert to NumPy array
    A = np.asarray(A)                            # Convert to NumPy array
    y = np.asarray(y)                            # Convert to NumPy array
    beta = theta[3:]                             # Extracting out theta's for the regression model

    # Estimating propensity score
    preds_reg = ee_logistic_regression(theta=beta,    # Using logistic regression
                                       X=X,           # Plug-in covariates for X
                                       y=A)           # Plug-in treatment for Y

    # Estimating weights
    pi = inverse_logit(np.dot(X, beta))          # Getting Pr(A|W) from model
    if truncate is not None:                     # Truncating Pr(A|W) when requested
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

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

        \sum_i^n \psi_1(Y_i, A_i, W_i, \pi_i, \theta_1) = \sum_i^n (\frac{A_i \times Y_i}{\pi_i} -
        \frac{\hat{Y^1}(A_i-\pi_i}{\pi_i}) - \theta_1 = 0

        \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n (\frac{(1-A_i) \times Y_i}{1-\pi_i} +
        \frac{\hat{Y^0}(A_i-\pi_i}{1-\pi_i})) - \theta_2 = 0

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
    theta : ndarray, list, vector
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
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
