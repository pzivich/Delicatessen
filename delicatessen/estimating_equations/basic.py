import numpy as np

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
