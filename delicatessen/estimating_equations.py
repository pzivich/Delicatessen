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
    user-defined functions are defined as ``psi``.

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
    Construction of a estimating equation(s) with ``ee_mean`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean(theta=theta, y=y_dat)

    Calling the M-estimation procedure

    >>> estr = MEstimator(stacked_equations=psi, init=[0, ])
    >>> estr.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> estr.theta
    >>> estr.variance
    >>> estr.asymptotic_variance

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

    where :math:`Y^*` is bounded between :math:`k` and :math:`-k`.

    Note
    ----
    Since psi is non-differentiable at :math:`k` or :math:`-k`, it must be assumed that the mean is sufficiently far
    from :math:`k`. Otherwise, difficulties might arise in the variance calculation.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in the case of the robust mean consists of a single value. Therefore, an initial value like the form of
        ``[0, ]`` is recommended.
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
    Construction of a estimating equation(s) with ``ee_mean_robust`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean_robust

    Some generic data to estimate the mean for

    >>> y_dat = [-10, 1, 2, 4, 1, 2, 3, 1, 5, 2, 33]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean_robust(theta=theta, y=y_dat, k=9)

    Calling the M-estimation procedure

    >>> estr = MEstimator(stacked_equations=psi, init=[0, ])
    >>> estr.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> estr.theta
    >>> estr.variance
    >>> estr.asymptotic_variance

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

    Unlike ``ee_mean``, theta consists of 2 elements. The output covariance matrix will also provide estimates for each
    of the theta values.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.


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
    Construction of a estimating equation(s) with ``ee_mean_variance`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean_variance

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean_variance(theta=theta, y=y_dat)

    Calling the M-estimation procedure (note that `init` has 2 values now).

    >>> estr = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> estr.theta
    >>> estr.variance
    >>> estr.asymptotic_variance

    For this estimating equation, ``mestimation.theta[1]`` and ``mestimation.asymptotic_variance[0][0]`` are expected
    to always be equal.

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
    The following optional parameters ``MEstimator.estimate()`` may benefit from these changes ``solver='hybr'``,
    ``dx=1``, ``order=15``, and increasing the ``tolerance``.

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
    Construction of a estimating equation(s) with ``ee_percentile`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_percentile

    Some generic data to estimate the mean for

    >>> np.random.seed(89041)
    >>> y_dat = np.random.normal(size=100)

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_percentile(theta=theta, y=y_dat, q=0.5)

    Calling the M-estimation procedure (note that ``init`` has 2 values now).

    >>> estr = MEstimator(stacked_equations=psi, init=[0, ])
    >>> estr.estimate(solver='hybr', tolerance=1e-3, dx=1, order=15)

    Notice that we use a different solver, tolerance values, and parameters for numerically approximating the derivative
    here. These changes generally work better for percentile optimizations since the estimating equation is non-smooth.
    Furthermore, optimization is hard when only a few observations (<100) are available. In general, closed form
    solutions for percentiles will be preferred.

    >>> estr.theta

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
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of two values. Therefore, initial values like the form of ``[0, 0]`` are
        recommended.
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the positive mean deviation).

    Returns
    -------
    array :
        Returns a 2-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_positive_mean_deviation`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_positive_mean_deviation

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_positive_mean_deviation(theta=theta, y=y_dat)

    Calling the M-estimation procedure (note that ``init`` has 2 values now).

    >>> estr = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> estr.theta
    >>> estr.variance
    >>> estr.asymptotic_variance

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
    user-defined functions are defined as ``psi``.

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
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
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
    Construction of a estimating equation(s) with ``ee_linear_regression`` should be done similar to the following

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

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])

    Calling the M-estimation procedure (note that ``init`` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

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
    user-defined functions are defined as ``psi``.

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
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
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
    Construction of a estimating equation(s) with ``ee_robust_linear_regression`` should be done similar to the
    following

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


    Calling the M-estimation procedure (note that ``init`` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

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


def ee_ridge_linear_regression(theta, y, X, penalty=1.0, weights=None):
    r"""Default stacked estimating equation for ridge linear regression. Ridge regression applies an L2-regularization
    through a squared magnitude penalty. The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i - \lambda \theta = 0

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    The 'strength' of the penalty term is indicated by :math:`\lambda`, which is the ``penalty`` argument scaled (or
    divided by) the number of observations.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

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
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    penalty : int, float, optional
        Penalty parameter for the ridge regression, which is further scaled by the number of observations.
        Default is 1.0
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_ridge_linear_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_ridge_linear_regression

    Some generic data to estimate a linear regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_ridge_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'], penalty=5.5)

    Calling the M-estimation procedure (note that ``init`` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    References
    ----------
    ...
    """
    X = np.asarray(X)                       # Convert to NumPy array
    y = np.asarray(y)[:, None]              # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra

    # Allowing for a weighted linear model
    if weights is None:                     # If weights is unspecified
        w = np.ones(X.shape[0])                 # ... assign weight of 1 to all observations
    else:                                   # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector

    # Creating penalty term for ridge regression
    penalty_scaled = penalty / X.shape[0]         # Scaling penalties by 1/N
    penalty_terms = (penalty_scaled *             # Penalty term applied to regression coefficients
                     np.asarray(theta))[:, None]

    # Output b-by-n matrix
    return (w*((y -                  # Speedy matrix algebra for regression
               np.dot(X, beta))      # ... linear regression requires no transformations
               * X).T                # ... multiply by coefficient and transpose for correct orientation
            - penalty_terms)         # Subtract off penalty term(s) from each observation


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
    user-defined functions are defined as ``psi``.

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
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
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
    Construction of a estimating equation(s) with ``ee_logistic_regression`` should be done similar to the following

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

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_logistic_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])


    Calling the M-estimation procedure (note that `init` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    X = np.asarray(X)                             # Convert to NumPy array
    y = np.asarray(y)[:, None]                    # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]             # Convert to NumPy array and ensure correct shape for matrix algebra

    # Allowing for a weighted logistic model
    if weights is None:                           # If weights is unspecified
        w = np.ones(X.shape[0])                   # ... assign weight of 1 to all observations
    else:                                         # Otherwise
        w = np.asarray(weights)                   # ... set weights as input vector

    # Output b-by-n matrix
    return w*((y -                                # Speedy matrix algebra for regression
               inverse_logit(np.dot(X, beta)))    # ... inverse-logit transformation of predictions
              * X).T                              # ... multiply by covariates and transpose for correct orientation


def ee_poisson_regression(theta, X, y, weights=None):
    r"""Default stacked estimating equation for Poisson regression. The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - \exp(X_i^T \theta)) X_i = 0

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

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
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
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

    References
    ----------
    """
    X = np.asarray(X)                           # Convert to NumPy array
    y = np.asarray(y)[:, None]                  # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]           # Convert to NumPy array and ensure correct shape for matrix algebra

    # Allowing for a weighted Poisson model
    if weights is None:                         # If weights is unspecified
        w = np.ones(X.shape[0])                 # ... assign weight of 1 to all observations
    else:                                       # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector

    # Output b-by-n matrix
    return w*((y                                # Speedy matrix algebra for regression
              - np.exp(np.dot(X, beta)))        # ... exp transformation of predictions
              * X).T                            # ... multiply by covariates and transpose for correct orientation


#################################################################
# Survival Estimating Equations


def ee_aft_weibull(theta, X, t, delta, weights=None):
    r"""Default stacked estimating equation for accelerated failure time (AFT) model with a Weibull distribution. The
    estimating equation is

    .. math::

        \psi(T_i,X_i,\delta_i; \lambda) = \frac{\delta_i}{\lambda} -  t_i^{\gamma} \exp(\beta X_i) \\
        \psi(T_i,X_i,\delta_i; \beta) = \delta_i X_i - (\lambda  t_i^{\gamma} \exp(\beta X_i))X_i \\
        \psi(T_i,X_i,\delta_i; \gamma) = \frac{\delta_i}{\gamma} + \delta_i \log(t) - \lambda t_i^{\gamma}
        \exp(\beta X_i) \log(t)

    Here, the Weibull-AFT actually consists of the following parameters: :math:`\mu, \beta, \sigma`. The above
    estimating equations use the proportional hazards form of the Weibull model. For the Weibull AFT, notice the
    following relation between the coefficients: :math:`\lambda = - \mu \gamma`,
    :math:`\beta_{PH} = - \beta_{AFT} \gamma`, and :math:`\gamma = \exp(\sigma)`.

    Here, :math:`\theta` is a 1-by-(2+b) array, where b is the distinct covariates included as part of X. For example,
    if X is a 3-by-n matrix, then theta will be a 1-by-5 array. The code is general to allow for an arbitrary number of
    X's (as long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        theta consists of 1+b+1 values. Therefore, initial values should consist of the same number as the number of
        columns present in ``X`` plus 2. This can easily be accomplished generally by
        ``[0, ] + [0, ] * X.shape[1] + [0, ]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    t : ndarray, list, vector
        1-dimensional vector of n observed times. Note that times can either be events (indicated by :math:`\delta_i=1`)
        or censored (indicated by :math:`\delta_i=0`). No missing data should be included (missing data may cause
        unexpected behavior).
    delta : ndarray, list, vector
        1-dimensional vector of n values indicating whether the time was an event or censoring. No missing data should
        be included (missing data may cause unexpected behavior).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta. The first element of theta corresponds to the scale
        parameter, the last element corresponds to the shape parameter, and the middle parameters correspond to the
        model coefficients.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_aft_weibull`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_aft_weibull

    Some generic survival data to estimate a Weibull AFT regresion model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> data['W'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> data['T'] = (1/1.25 + 1/np.exp(0.5)*data['X'])*np.random.weibull(a=0.75, size=n)
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 10, 10, data['C'])
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])
    >>> d_obs = data[['X', 'W', 't', 'delta']].copy()

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_aft_weibull(theta=theta, X=d_obs[['C', 'X', 'Z']],
    >>>                               t=d_obs['t'], delta=d_obs['delta'])

    Calling the M-estimation procedure (note that `init` has 2+2 values now, since ``X.shape[1] = 2``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting parameter the specific parameter estimates

    >>> estr.theta[0]     # log(mu)    (scale)
    >>> estr.theta[1:-1]  # log(beta)  (scale coefficients)
    >>> estr.theta[-1]    # log(sigma) (shape)

    References
    ----------
    Collett D. (2015). Parametric proportional hazards models In: Modelling survival data in medical research.
    CRC press. pg171-220

    Collett D. (2015). Accelerated failure time and other parametric models. In: Modelling survival data in medical
    research. CRC press. pg171-220
    """
    # TODO expand sigma to allow for coefficients too!
    X = np.asarray(X)                          # Convert to NumPy array
    t = np.asarray(t)[:, None]                 # Convert to NumPy array and ensure correct shape for matrix algebra
    delta = np.asarray(delta)[:, None]         # Convert to NumPy array and ensure correct shape for matrix algebra
    beta_dim = X.shape[1]

    # Extract coefficients
    sigma = np.exp(theta[-1])                  # exponential so as to be nice to optimizer
    mu = np.exp(-1 * theta[0] * sigma)         # exponential so as to be nice to optimizer, and apply PH->AFT transform
    beta = (-1 * sigma *                       # exponential so as to be nice to optimizer, and apply PH->AFT transform
            np.asarray(theta[1:beta_dim+1])[:, None])
    #   Rationale: I apply some transformations for the AFT model. These transformations are to go from the proportional
    #       hazards form of the Weibull model to the AFT form of the Weibull model. Explicitly,
    #           lambda = exp(-mu * sigma)
    #           beta   = -alpha * sigma
    #           gamma  = exp(sigma)
    #       I used the proportional hazards form because the log-likelihood has written out on page 200 of Collett's
    #       "Modeling Survival Data in Medical Research" (3ed). I then solved for the derivative, which gives the
    #       3 contributions to the score function (which is also the estimating equations here).

    # Allowing for a weighted Weibull-AFT model
    if weights is None:                         # If weights is unspecified
        w = np.ones(X.shape[0])                 # ... assign weight of 1 to all observations
    else:                                       # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector

    # Intermediate calculations (evaluated once to optimize run-time)
    exp_coefs = np.exp(np.dot(X, beta))         # Calculates the exponential of coefficients
    log_t = np.log(t)                           # Calculates natural log of the time contribution

    # Estimating equations
    contribution_1 = w*(delta/mu                               # Estimating equation: mu
                        - exp_coefs*(t**sigma)).T
    contribution_2 = w*((delta                                 # Estimating equation: beta
                         - mu*(t**sigma)*exp_coefs)*X).T
    contribution_3 = w*(delta/sigma                            # Estimating equation: sigma
                        + delta*log_t
                        - mu*(t**sigma)*exp_coefs*log_t).T

    # Output b-by-n matrix
    return np.vstack((contribution_1,      # mu contribution
                      contribution_2,      # beta contribution
                      contribution_3))     # sigma contribution


def ee_aft_weibull_measure(theta, times, X, measure, mu, beta, sigma):
    r"""Default stacked estimating equation to calculate a survival measure (survival, density, risk, hazard,
    cumulative hazard) given a specific covariate pattern and coefficients from a Weibull accelerated failure time
    (AFT) model. The estimating equation for the survival function at time :math:`t` is

    .. math::

        \psi_S(T_i,X_i,\delta_i; \theta, \mu, \beta, \sigma) = \exp(-1 \lambda_i t^{\gamma}) - \theta

    and the estimating equation for the hazard function at time :math:`t` is

    .. math::

        \psi_h(T_i,X_i,\delta_i; \theta, \mu, \beta, \sigma) = \lambda_i \gamma t^{\gamma - 1} - \theta

    where

    .. math::

        \gamma = \exp(\sigma) \\
        \lambda_i = \exp(-1 (\mu + X \beta) * \gamma)

    For the other measures, we take advantage of the following known transformation behind survival meaures

    .. math::

        F(t) = 1 - S(t) \\
        H(t) = -\log(S(t)) \\
        f(t) = h(t) S(t)

    Note
    ----
    For proper uncertainty estimation, this estimating equation is meant to be stacked together with the corresponding
    Weibull AFT model.

    Parameters
    ----------
    theta : ndarray, list, vector
        theta consists of t values. The initial values should consist of the same number of elements as provided in the
        ``times`` argument.
    times : int, float, ndarray, list, vector
        A single time or 1-dimensional collection of times to calculate the measure at. The number of provided times
        should consist of the same number of elements as provided in the ``theta`` argument.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    measure : str
        Measure to calculate. Options include survival (``'survival'``), density (``'density'``), risk or the cumulative
        density (``'risk'``), hazard (``'hazard'``), or cumulative hazard (``'cumulative_hazard'``).
    mu : float, int
        The estimated scale parameter from the Weibull AFT. From ``ee_aft_weibull``, will be the first element.
    beta :
        The estimated scale coefficients from the Weibull AFT. From ``ee_aft_weibull``, will be the middle element(s).
    sigma :
        The estimated shape parameter from the Weibull AFT. From ``ee_aft_weibull``, will be the last element.

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta

    Examples
    --------
    Construction of a estimating equations for :math:`S(t=5)` with ``ee_aft_weibull_measure`` should be done similar to
    the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_aft_weibull, ee_aft_weibull_measure

    For demonstration, we will generated generic survival data

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> data['W'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> data['T'] = (1/1.25 + 1/np.exp(0.5)*data['X'])*np.random.weibull(a=0.75, size=n)
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 10, 10, data['C'])
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])
    >>> d_obs = data[['X', 'W', 't', 'delta']].copy()

    Our interest will be in the survival among those with :math:`X=1,W=1`. Therefore, we will generate a copy of the
    data and set the values in that copy (to keep the dimension the same across both estimating equations).

    >>> d_coef = d_obs.copy()
    >>> d_coef['X'] = 1
    >>> d_coef['W'] = 1

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     aft = ee_aft_weibull(theta=theta[0:4],
    >>>                     t=d_obs['t'], delta=d_obs['delta'], X=d_obs[['X', 'W']])
    >>>     pred_surv_t = ee_aft_weibull_measure(theta=theta[4], X=d_coef[['X', 'W']],
    >>>                                          times=5, measure='survival',
    >>>                                          mu=theta[0], beta=theta[1:3], sigma=theta[3])
    >>>     return np.vstack((aft, pred_surv_t))

    Calling the M-estimation procedure (note that `init` has 2+2+1 values now, since ``X.shape[1] = 2`` and we are
    calculating the survival at time 5).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0., 0.5])
    >>> estr.estimate(solver='lm')

    Inspecting the estimate, variance, and confidence intervals for :math:`S(t=5)`

    >>> estr.theta[-1]                      # \hat{S}(t)
    >>> estr.variance[-1, -1]               # \hat{Var}(\hat{S}(t))
    >>> estr.confidence_intervals()[-1, :]  # 95% CI for S(t)

    Next, we will consider evaluating the survival function at multiple time points (so we can easily create a plot of
    the survival function and the corresponding confidence intervals)

    Note
    ----
    When calculate the survival (or other measures) at many time points, it is generally best to optimize the Weibull
    AFT coefficients in a separate model, then use the pre-washed coefficients in another M-estimator with the many
    time points. This helps the optimizer to converge faster in number of iterations and total run-time.

    To make everything easier, we will generate a list of uniformly spaced values between the start and end points of
    our desired survival function. We will also generate initial values of the same length (to help the optimizer, we
    also start our starting values from near one and end near zero).

    >>> resolution = 50
    >>> time_spacing = list(np.linspace(0.01, 8, resolution))
    >>> fast_inits = list(np.log(np.linspace(0.99, 0.01, resolution)))

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     aft = ee_aft_weibull(theta=theta[0:4],
    >>>                     t=d_obs['t'], delta=d_obs['delta'], X=d_obs[['X', 'W']])
    >>>     pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=d_coef[['X', 'W']],
    >>>                                          times=5, measure='survival',
    >>>                                          mu=theta[0], beta=theta[1:3], sigma=theta[3])
    >>>     return np.vstack((aft, pred_surv_t))

    Calling the M-estimation procedure. As stated in the note above, we use the pre-washed covariates to help the
    optimizer (since the resolution means we are estimating 50 different parameters).

    >>> estr = MEstimator(psi, init=list(mest.theta) + fast_inits)
    >>> estr.estimate(solver="lm")

    To plot the survival curves, we could do the following:

    >>> import matplotlib.pyplot as plt
    >>> ci = estr.confidence_intervals()[4:, :]  # Extracting relevant CI
    >>> plt.fill_between(time_spacing, ci[:, 0], ci[:, 1], alpha=0.2)
    >>> plt.plot(time_spacing, estr.theta[4:], '-')
    >>> plt.show()

    References
    ----------
    Collett D. (2015). Accelerated failure time and other parametric models. In: Modelling survival data in medical
    research. CRC press. pg171-220
    """
    X = np.asarray(X)                      # Convert to NumPy array

    # Extract coefficients
    gamma = np.exp(sigma)                                # exponential to convert to regular sigma
    beta = np.asarray(beta)[:, None]                     # Pulling out the coefficients
    lambd = np.exp(-1 * (mu + np.dot(X, beta)) * gamma)  # Calculating lambda

    def calculate_metric(time, theta_t):
        # Intermediate calculations
        survival_t = np.exp(-1 * lambd * time**gamma)   # Survival calculation from parameters
        hazard_t = lambd * gamma * time**(gamma-1)      # hazard calculation from parameters

        # Calculating specific measures
        if measure == "survival":
            metric = survival_t                       # S(t) = S(t)
        elif measure == "risk":
            metric = 1 - survival_t                   # F(t) = 1 - S(t)
        elif measure == "cumulative_hazard":
            metric = -1 * np.log(survival_t)          # H(t) = -log(S(t))
        elif measure == "hazard":
            metric = hazard_t                         # h(t) = h(t)
        elif measure == "density":
            metric = hazard_t * survival_t            # f(t) = h(t) * S(t)
        else:
            raise ValueError("The measure '"
                             + str(measure)
                             + "' is not supported. Please select one of the following: "
                               "survival, density, risk, hazard, cumulative_hazard.")
        return (metric - theta_t).T                   # Calculate difference from theta, and do transpose for vstack

    # Logic to allow for either a single time or multiple times
    if type(times) is int or type(times) is float:               # For single time,
        return calculate_metric(time=times, theta_t=theta)       # ... calculate the transformation and return
    else:                                                        # For multiple time points,
        if len(theta) != len(times):                             # ... check length is the same (to prevent errors)
            raise ValueError("There is a mismatch between the number of "
                             "`theta`'s and the number of `times` provided.")
        stacked_time_evals = []                                  # ... empty list for stacking the equations
        for t, thet in zip(times, theta):                        # ... loop through each theta and each time
            metric_t = calculate_metric(time=t, theta_t=thet)    # ... ... calculate the transformation
            stacked_time_evals.append(metric_t)                  # ... ... stack transformation into storage
        return np.vstack(stacked_time_evals)                     # ... return a vstack of the equations


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
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 4 values. In general, starting values ``>0`` are better choices for the 4PL model
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
    Construction of a estimating equation(s) with ``ee_4p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_4p_logistic

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> dose_data = d[:, 1]   # Dose data
    >>> resp_data = d[:, 0]   # Response data

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_4p_logistic(theta=theta, X=dose_data, y=resp_data)

    The 4PL model and others are harder to optimize compared to other estimating equations. Namely, the optimizer is
    not aware of implicit bounds on the parameters. To reduce non-convergence issues, we can give the root-finder good
    starting values.

    For the 4PL, the upper limit should *always* be greater than the lower limit. Second, the ED50 should be between
    the lower and upper limits. Third, the sign for the steepness depends on whether the response declines (positive)
    or the response increases (negative). Finally, some solvers may be better suited to the problem, so try a few
    different options.

    Here, we use some general starting values that should perform well in many cases. For the lower-bound, give the
    minimum response value as the initial. For ED50, give the mid-point between the maximum response and the minimum
    response. The initial value for steepness is more difficult. Ideally, we would give a starting value of zero, but
    that will fail in this example. Giving a small positive starting value works in this example. For the upper-bound,
    give the maximum response value as the initial. Finally, we use the ``lm`` solver.

    Note
    ----
    To summarize the recommendations, be sure to examine your data (e.g., scatterplot). This will help to determine the
    initial starting values for the root-finding procedure. Otherwise, you may come across a convergence error.

    >>> estr = MEstimator(psi, init=[np.min(resp_data),
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              np.max(resp_data)])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # lower limit
    >>> estr.theta[1]    # ED(50)
    >>> estr.theta[2]    # steepness
    >>> estr.theta[3]    # upper limit

    References
    ----------
    Ritz C, Baty F, Streibig JC, & Gerhard D. (2015). Dose-response analysis using R. *PloS One*, 10(12), e0146021.

    An H, Justin TL, Aubrey GB, Marron JS, & Dittmer DP. (2019). dr4pl: A Stable Convergence Algorithm for the 4
    Parameter Logistic Model. *R J.*, 11(2), 171.

    Inderjit, Streibig JC, & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
    allelopathy research. *Physiologia Plantarum*, 114(3), 422-428.
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
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 3 values. In general, starting values ``>0`` are better choices for the 3PL model
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
    Construction of a estimating equation(s) with ``ee_3p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_3p_logistic

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> dose_data = d[:, 1]   # Dose data
    >>> resp_data = d[:, 0]   # Response data

    Since there is a natural lower-bound of 0 for root growth, we set ``lower=0``. Defining psi, or the stacked
    estimating equations

    >>> def psi(theta):
    >>>     return ee_3p_logistic(theta=theta, X=dose_data, y=resp_data,
    >>>                           lower=0)

    The 3PL model and others are harder to optimize compared to other estimating equations. Namely, the optimizer is
    not aware of implicit bounds on the parameters. To reduce non-convergence issues, we can give the root-finder good
    starting values.

    For the 3PL, the upper limit should *always* be greater than the set lower limit. Second, the ED50 should be between
    the lower and upper limits. Third, the sign for the steepness depends on whether the response declines (positive)
    or the response increases (negative). Finally, some solvers may be better suited to the problem, so try a few
    different options.

    Here, we use some general starting values that should perform well in many cases. For ED50, give the mid-point
    between the maximum response and the minimum response. The initial value for steepness is more difficult. Ideally,
    we would give a starting value of zero, but that will fail in this example. Giving a small positive starting value
    works in this example. For the upper-bound, give the maximum response value as the initial. Finally, we use the
    ``lm`` solver.

    Note
    ----
    To summarize the recommendations, be sure to examine your data (e.g., scatterplot). This will help to determine the
    initial starting values for the root-finding procedure. Otherwise, you may come across a convergence error.

    >>> estr = MEstimator(psi, init=[(np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              np.max(resp_data)])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # ED(50)
    >>> estr.theta[1]    # steepness
    >>> estr.theta[2]    # upper limit

    References
    ----------
    Ritz C, Baty F, Streibig JC, & Gerhard D. (2015). Dose-response analysis using R. *PloS One*, 10(12), e0146021.

    An H, Justin TL, Aubrey GB, Marron JS, & Dittmer DP. (2019). dr4pl: A Stable Convergence Algorithm for the 4
    Parameter Logistic Model. *R J.*, 11(2), 171.

    Inderjit, Streibig JC, & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
    allelopathy research. *Physiologia Plantarum*, 114(3), 422-428.
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
    user-defined functions are defined as ``psi``.

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
    Construction of a estimating equation(s) with ``ee_2p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_2p_logistic

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> dose_data = d[:, 1]   # Dose data
    >>> resp_data = d[:, 0]   # Response data

    Since there is a natural lower-bound of 0 for root growth, we set ``lower=0``. While a natural upper bound does not
    exist for this example, we set ``upper=8`` for illustrative purposes. Defining psi, or the stacked estimating
    equations

    >>> def psi(theta):
    >>>     return ee_2p_logistic(theta=theta, X=dose_data, y=resp_data,
    >>>                           lower=0, upper=8)

    The 2PL model and others are harder to optimize compared to other estimating equations. Namely, the optimizer is
    not aware of implicit bounds on the parameters. To reduce non-convergence issues, we can give the root-finder good
    starting values.

    First, the ED50 should be between the lower and upper limits. Second, the sign for the steepness depends on whether
    the response declines (positive) or the response increases (negative). Finally, some solvers may be better suited
    to the problem, so try a few different options.

    Here, we use some general starting values that should perform well in many cases. For ED50, give the mid-point
    between the maximum response and the minimum response. The initial value for steepness is more difficult. Ideally,
    we would give a starting value of zero, but that will fail in this example. Giving a small positive starting value
    works in this example. Finally, we use the ``lm`` solver.

    Note
    ----
    To summarize the recommendations, be sure to examine your data (e.g., scatterplot). This will help to determine the
    initial starting values for the root-finding procedure. Otherwise, you may come across a convergence error.

    >>> estr = MEstimator(psi, init=[(np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # ED(50)
    >>> estr.theta[1]    # steepness

    References
    ----------
    Ritz C, Baty F, Streibig JC, & Gerhard D. (2015). Dose-response analysis using R. *PloS One*, 10(12), e0146021.

    An H, Justin TL, Aubrey GB, Marron JS, & Dittmer DP. (2019). dr4pl: A Stable Convergence Algorithm for the 4
    Parameter Logistic Model. *R J.*, 11(2), 171.

    Inderjit, Streibig JC, & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
    allelopathy research. *Physiologia Plantarum*, 114(3), 422-428.
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
    Construction of a estimating equations for ED25 with ``ee_3p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_2p_logistic, ee_effective_dose_delta

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> dose_data = d[:, 1]   # Dose data
    >>> resp_data = d[:, 0]   # Response data

    Since there is a natural lower-bound of 0 for root growth, we set ``lower=0``. While a natural upper bound does not
    exist for this example, we set ``upper=8`` for illustrative purposes. Defining psi, or the stacked estimating
    equations

    >>> def psi(theta):
    >>>     pl_model = ee_3p_logistic(theta=theta, X=dose_data, y=resp_data,
    >>>                               lower=0)
    >>>     ed_25 = ee_effective_dose_delta(theta[3], y=resp_data, delta=0.20,
    >>>                                     steepness=theta[0], ed50=theta[1],
    >>>                                     lower=0, upper=theta[2])
    >>>     # Returning stacked estimating equations
    >>>     return np.vstack((pl_model,
    >>>                       ed_25,))

    Notice that the estimating equations are stacked in the order of the parameters in ``theta`` (the first 3 belong to
    3PL and the last belong to ED(25)).

    >>> estr = MEstimator(psi, init=[(np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              np.max(resp_data),
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # ED(50)
    >>> estr.theta[1]    # steepness
    >>> estr.theta[2]    # upper limit
    >>> estr.theta[3]    # ED(25)

    References
    ----------
    Ritz C, Baty F, Streibig JC, & Gerhard D. (2015). Dose-response analysis using R. *PloS One*, 10(12), e0146021.

    An H, Justin TL, Aubrey GB, Marron JS, & Dittmer DP. (2019). dr4pl: A Stable Convergence Algorithm for the 4
    Parameter Logistic Model. *R J.*, 11(2), 171.

    Inderjit, Streibig JC, & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
    allelopathy research. *Physiologia Plantarum*, 114(3), 422-428.
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

        \sum_i^n \psi_m(Y_i, X_i, \beta) = \sum_i^n (Y_i - expit(X_i^T \beta)) X_i = 0

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
    user-defined functions are defined as ``psi``.

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
    Construction of a estimating equation(s) with ``ee_gformula`` should be done similar to the following

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
    of a causal inference technique. *American Journal of Epidemiology*, 173(7), 731-738.

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


def ee_ipw(theta, y, A, W, truncate=None):
    r"""Default stacked estimating equation for inverse probability weighting in the time-fixed setting. The
    parameter of interest is the average causal effect. For estimation of the weights (or propensity scores), a
    logistic model is used.

    Note
    ----
    Unlike ``ee_gformula``, ``ee_ipw`` only provides the average causal effect (and the causal means for ``A=1`` and
    ``A=0``). In other words, the implementation of IPW does not support generic action plans off-the-shelf,
    unlike ``ee_gformula``.

    The first estimating equation for the logistic regression model is

    .. math::

        \sum_i^n \psi_g(A_i, W_i, \alpha) = \sum_i^n (A_i - expit(W_i^T \alpha)) W_i = 0

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
    user-defined functions are defined as ``psi``.

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
    W : ndarray, list, vector
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
    >>>                   W=d[['C', 'W']])

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
    >>> estr.theta[1]    # causal mean under A=1
    >>> estr.theta[2]    # causal mean under A=0
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
    W = np.asarray(W)                            # Convert to NumPy array
    A = np.asarray(A)                            # Convert to NumPy array
    y = np.asarray(y)                            # Convert to NumPy array
    beta = theta[3:]                             # Extracting out theta's for the regression model

    # Estimating propensity score
    preds_reg = ee_logistic_regression(theta=beta,    # Using logistic regression
                                       X=W,           # Plug-in covariates for X
                                       y=A)           # Plug-in treatment for Y

    # Estimating weights
    pi = inverse_logit(np.dot(W, beta))          # Getting Pr(A|W) from model
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


def ee_aipw(theta, y, A, W, X, X1, X0, truncate=None, force_continuous=False):
    r"""Default stacked estimating equation for augmented inverse probability weighting (AIPW) in the time-fixed
    setting. The parameter of interest is the average causal effect.

    Note
    ----
    Unlike ``ee_gformula``, ``ee_ipw`` only provides the average causal effect (and the causal means for ``A=1`` and
    ``A=0``). In other words, the implementation of IPW does not support generic action plans off-the-shelf,
    unlike ``ee_gformula``.

    AIPW consists of two nuisance models (the propensity score model and the outcome model). For estimation of the
    propensity scores, a logistic model is used.

    .. math::

        \sum_i^n \psi_g(A_i, W_i, \alpha) = \sum_i^n (A_i - expit(W_i^T \alpha)) W_i = 0

    where ``A`` is the treatment and ``W`` is the set of confounders.

    Next, an outcome model is specified. For continuous Y, the linear regression estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \beta) = \sum_i^n (Y_i - X_i^T \beta) X_i = 0

    and for logistic regression, the estimating equation is

    .. math::

        \sum_i^n \psi_m(Y_i, X_i, \beta) = \sum_i^n (Y_i - expit(X_i^T \beta)) X_i = 0

    By default, `ee_aipw` detects whether `y` is all binary (zero or one), and applies logistic regression if that
    happens. See the parameters for more details. Notice that ``X`` here should consists of both ``A`` and ``W`` (with
    possible interaction terms or other differences in functional forms from the propensity score model).

    For the implementation of the AIPW estimator, stacked estimating equations further include the mean had everyone
    been set to ``A=1``, the mean had everyone been set to ``A=0``, and the mean difference. Those estimating equations
    look like

    .. math::

        \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

        \sum_i^n \psi_1(Y_i, A_i, W_i, \pi_i, \theta_1) = \sum_i^n (\frac{A_i \times Y_i}{\pi_i} -
        \frac{\hat{Y^1}(A_i-\pi_i}{\pi_i}) - \theta_1 = 0

        \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n (\frac{(1-A_i) \times Y_i}{1-\pi_i} +
        \frac{\hat{Y^0}(A_i-\pi_i}{1-\pi_i})) - \theta_2 = 0

    where :math:`Y^a` is the predicted values of :math:`Y` from the outcome model under action
    assignment :math:`A=a`.

    Due to these 3 extra values and two nuisance models, the length of the theta vector is 3+b+c, where b is the number
    of columns in ``W``, and c is the number of columns in ``X``.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Here, theta corresponds to a variety of different quantities. The *first* value in theta vector is mean
    difference (or average causal effect), the *second* is the mean had everyone been given ``A=1``, the *third* is the
    mean had everyone been given ``A=0``. The remainder of the parameters correspond to the regression model
    coefficients, in the order input. The first 'chunk' of coefficients correspond to the propensity score model
    and the last 'chunk' correspond to the outcome model.

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
    W : ndarray, list, vector
        2-dimensional vector of n observed values for b variables to model the probability of ``A`` with. No missing
        data should be included (missing data may cause unexpected behavior).
    X : ndarray, list, vector
        2-dimensional vector of n observed values for c variables to model the outcome ``y``. No missing data should
        be included (missing data may cause unexpected behavior).
    X1 : ndarray, list, vector
        2-dimensional vector of n observed values for b variables under the action plan. If the action is indicated by
        ``A``, then ``X1`` will take the original data ``X`` and update the values of ``A`` to follow the deterministic
        plan where ``A=1`` for all observations. No missing data should be included (missing data may cause unexpected
        behavior).
    X0 : ndarray, list, vector, None, optional
        2-dimensional vector of n observed values for b variables under the action plan. This second argument is
        optional and should be specified if a causal mean difference between two action plans is of interest. If the
        action is indicated by ``A``, then ``X0`` will take the original data ``X`` and update the values of ``A`` to
        follow the deterministic plan where ``A=0`` for all observatons. No missing data should be included (missing
        data may cause unexpected behavior).
    truncate : None, list, set, optional
        Bounds to truncate the estimated probabilities of ``A`` at. For example, estimated probabilities above 0.99 or
        below 0.01 can be set to 0.99 or 0.01, respectively. This is done by specifying ``truncate=(0.01, 0.99)``. Note
        this step is done via ``numpy.clip(.., a_min=truncate[0], a_max=truncate[1])``, so order is important. Default
        is None, which applies to no truncation.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (3+b+c)-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_aipw`` should be done similar to the following

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

    Defining psi, or the stacked estimating equations. Note that ``A`` is the action of interest. First, we will apply
    some necessary data processing.  We will create an interaction term between ``A`` and ``W`` in the original data.
    Then we will generate a copy of the data and update the values of ``A=1``, and then generate another
    copy but set ``A=0`` in that copy.

    >>> d['AW'] = d['A']*d['W']
    >>> d1 = d.copy()   # Copy where all A=1
    >>> d1['A'] = 1
    >>> d1['AW'] = d1['A']*d1['W']
    >>> d0 = d.copy()   # Copy where all A=0
    >>> d0['A'] = 0
    >>> d0['AW'] = d0['A']*d0['W']

    Having setup our data, we can now define the psi function.

    >>> def psi(theta):
    >>>     return ee_aipw(theta,
    >>>                    y=d['Y'],
    >>>                    A=d['A'],
    >>>                    W=d[['C', 'W']],
    >>>                    X=d[['C', 'A', 'W', 'AW']],
    >>>                    X1=d1[['C', 'A', 'W', 'AW']],
    >>>                    X0=d0[['C', 'A', 'W', 'AW']])

    Calling the M-estimation procedure. AIPW has 3 parameters with 2 coefficients in the propensity score model, and
    4 coefficients in the outcome model, the total number of initial values should be 3+2+4=9. When Y is binary, it
    will be best to start with ``[0., 0.5, 0.5, ...]`` followed by all ``0.`` for the initial values. Otherwise,
    starting with all 0. as initials is reasonable.

    >>> estr = MEstimator(psi,
    >>>                   init=[0., 0.5, 0.5, 0., 0., 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the corresponding parameters are

    >>> estr.theta[0]     # causal mean difference of 1 versus 0
    >>> estr.theta[1]     # causal mean under A=1
    >>> estr.theta[2]     # causal mean under A=0
    >>> estr.theta[3:5]   # propensity score regression coefficients
    >>> estr.theta[5:]    # outcome regression coefficients

    References
    ----------
    Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
    *Journal of Epidemiology & Community Health*, 60(7), 578-586.

    Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust estimation of causal
    effects. *American Journal of Epidemiology*, 173(7), 761-767.

    Tsiatis AA. (2006). Semiparametric theory and missing data. Springer, New York, NY.
    """
    # Ensuring correct typing
    y = np.asarray(y)              # Convert to NumPy array
    A = np.asarray(A)              # Convert to NumPy array
    W = np.asarray(W)              # Convert to NumPy array
    X = np.asarray(X)              # Convert to NumPy array
    X1 = np.asarray(X1)            # Convert to NumPy array
    X0 = np.asarray(X0)            # Convert to NumPy array

    # Checking some shapes
    if X.shape != X1.shape:
        raise ValueError("The dimensions of X and X1 must be the same.")
    if X.shape != X0.shape:
        raise ValueError("The dimensions of X and X0 must be the same.")

    # Extracting theta value for ease
    mud = theta[0]                 # Parameter for average causal effect
    mu1 = theta[1]                 # Parameter for the mean under A=1
    mu0 = theta[2]                 # Parameter for the mean under A=0
    alpha = theta[3:3+W.shape[1]]  # Parameter(s) for the propensity score model
    beta = theta[3+W.shape[1]:]  # Parameter(s) for the outcome model

    # pi-model (logistic regression)
    pi_model = ee_logistic_regression(theta=alpha,    # Estimating logistic model
                                      X=W,
                                      y=A)
    pi = inverse_logit(np.dot(W, alpha))              # Estimating Pr(A|W)
    if truncate is not None:                          # Truncating Pr(A|W) when requested
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    # m-model (logistic regression)
    # Checking outcome variable type
    if np.isin(y, [0, 1]).all() and not force_continuous:
        regression = ee_logistic_regression         # Use a logistic regression model
        transform = inverse_logit                   # ... and need to inverse-logit transformation
    else:
        regression = ee_linear_regression           # Use a linear regression model
        transform = identity                        # ... and need to apply the identity (no) transformation

    m_model = regression(theta=beta,                # Estimating the outcome model
                         y=y, X=X)
    ya1 = transform(np.dot(X1, beta))               # Generating predicted values under X1
    ya0 = transform(np.dot(X0, beta))               # Generating predicted values under X0

    # AIPW estimator
    ace = np.ones(y.shape[0]) * (mu1 - mu0) - mud               # Calculating the ATE
    y1_star = (y*A/pi - ya1*(A-pi)/pi) - mu1                    # Calculating \tilde{Y}(a=1)
    y0_star = (y*(1-A)/(1-pi) + ya0*(A-pi)/(1-pi)) - mu0        # Calculating \tilde{Y}(a=0)

    # Output (3+b+c)-by-n array
    return np.vstack((ace,               # theta[0] is for the ATE
                      y1_star[None, :],  # theta[1] is for R1
                      y0_star[None, :],  # theta[2] is for R0
                      pi_model,          # theta[b] is for the treatment model coefficients
                      m_model))          # theta[c] is for the outcome model coefficients
