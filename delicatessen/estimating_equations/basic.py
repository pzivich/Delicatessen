import warnings

import numpy as np
from delicatessen.utilities import robust_loss_functions


#################################################################
# Basic Estimating Equations


def ee_mean(theta, y):
    r"""Estimating equation for the mean. The estimating equation for the mean is

    .. math::

        \sum_{i=1}^n Y_i - \theta_1 = 0

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in the case of the mean consists of a single value. Therefore, an initial value like the form of
        ``[0, ]`` should be provided.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input ``theta`` and ``y``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_mean`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the estimating equation

    >>> def psi(theta):
    >>>     return ee_mean(theta=theta, y=y_dat)

    Calling the M-estimator

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


def ee_mean_robust(theta, y, k, loss='huber', lower=None, upper=None):
    r"""Estimating equation for the (unscaled) robust mean. The estimating equation for the robust
    mean is

    .. math::

        \sum_{i=1}^n f_k(Y_i - \theta) = 0

    where :math:`f_k(x)` is the corresponding robust loss function. Options for the loss function include: Huber,
    Tukey's biweight, Andrew's Sine, and Hampel. See ``robust_loss_function`` for further details on the loss
    functions for the robust mean.

    Note
    ----
    The estimating-equation is not non-differentiable everywhere for some loss functions. Therefore, it is assumed that
    no points occur exactly at the non-differentiable points. For truly continuous :math:`Y`, the probability of that
    occurring is zero.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in the case of the robust mean consists of a single value. Therefore, an initial value like the form of
        ``[0, ]`` is should be provided.
    y : ndarray, vector, list
        1-dimensional vector of n observed values.
    k : int, float
        Tuning or hyperparameter for the chosen loss function. Notice that the choice of hyperparameter depends on the
        loss function.
    loss : str, optional
        Robust loss function to use. Default is 'huber'. Options include 'andrew', 'hampel', 'huber', 'tukey'.
    lower : int, float, None, optional
        Lower parameter for the 'hampel' loss function. This parameter does not impact the other loss functions.
        Default is ``None``.
    upper : int, float, None, optional
        Upper parameter for the 'hampel' loss function. This parameter does not impact the other loss functions.
        Default is ``None``.

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

    Defining psi, or the stacked estimating equations for Huber's robust mean

    >>> def psi(theta):
    >>>     return ee_mean_robust(theta=theta, y=y_dat, k=9, loss='huber')

    Calling the M-estimation procedure

    >>> estr = MEstimator(stacked_equations=psi, init=[0, ])
    >>> estr.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> estr.theta
    >>> estr.variance
    >>> estr.asymptotic_variance

    References
    ----------
    Andrews DF. (1974). A robust method for multiple linear regression. *Technometrics*, 16(4), 523-531.

    Beaton AE & Tukey JW (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic
    data. *Technometrics*, 16(2), 147-185.

    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Hampel FR. (1971). A general qualitative definition of robustness. *The Annals of Mathematical Statistics*,
    42(6), 1887-1896.

    Huber PJ. (1964). Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101.

    Huber PJ, Ronchetti EM. (2009) Robust Statistics 2nd Edition. Wiley. pgs 98-100
    """
    # Calculate the robust loss function of the residuals for the mean
    return robust_loss_functions(residual=np.asarray(y) - theta,   # Convert y to array and calculate residual
                                 k=k,                              # ... hyperparameter for loss function
                                 loss=loss,                        # ... chosen loss function to use
                                 a=lower,                          # ... lower limit (Hampel only)
                                 b=upper)                          # ... upper limit (Hampel only)


def ee_mean_variance(theta, y):
    r"""Estimating equations for the mean and variance. The estimating equations for the mean and
     variance are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            Y_i - \theta_1 = 0 \\
            (Y_i - \theta_1)^2 - \theta_2
        \end{bmatrix}
        = 0

    Unlike ``ee_mean``, ``theta`` consists of 2 parameters. The output covariance matrix will also provide estimates
    for each of the ``theta`` values.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.


    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of two values. Therefore, initial values like the form of ``[0, 0]`` should be
        provided.
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

    Calling the M-estimator (note that `init` has 2 values)

    >>> estr = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> estr.theta
    >>> estr.variance
    >>> estr.asymptotic_variance

    For this estimating equation, ``mestimation.theta[1]`` and ``mestimation.asymptotic_variance[0][0]`` are expected
    to be equal.

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
    r"""Estimating equation for the q percentile. The estimating equation is

    .. math::

        \sum_{i=1}^n q - I(Y_i \le \theta) = 0

    where :math:`0 < q < 1` is the percentile. Notice that this estimating equation is non-smooth. Therefore,
    root-finding is difficult.

    Note
    ----
    As the derivative of the estimating equation is not defined at :math:`\hat{\theta}`, the bread (and sandwich)
    cannot be used to estimate the variance. This estimating equation is offered for completeness, but is not generally
    recommended for applications.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of one value. Therefore, initial values like the form of ``[0, ]`` should be
        provided.
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
    # Warning about the bread
    warnings.warn("The estimating equation is not differentiable at theta. Therefore, the bread matrix is not defined "
                  "for finite samples, and the sandwich should not be used to estimate the variance.",
                  UserWarning)

    if q >= 1 or q <= 0:
        raise ValueError("`q` must be (0, 1)")

    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Output 1-by-n array of the estimating equations
    return q - 1*(y_array <= theta)


def ee_positive_mean_deviation(theta, y):
    r"""Estimating equations for the positive mean deviation. The estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            2(Y_i - \theta_2)I(Y_i > \theta_2) - \theta_1 \\
            0.5 - I(Y_i \le \theta_2)
        \end{bmatrix}
        = 0

    where the first estimating equation is for the positive mean difference, and the second estimating equation is for
    the median. Notice that this estimating equation is non-smooth. Therefore, root-finding is difficult.

    Note
    ----
    As the derivative of the estimating equation for the median is not defined at :math:`\hat{\theta}`, the bread (and
    sandwich) cannot be used to estimate the variance. This estimating equation is offered for completeness, but is not
    generally recommended for applications.

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
    # Warning about the bread
    warnings.warn("The estimating equation for the median is not differentiable. Therefore, the bread matrix is not "
                  "defined for finite samples, and the sandwich should not be used to estimate the variance.",
                  UserWarning)

    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Calculating median with built-in estimating equation
    median = ee_percentile(theta=theta[1], y=y_array, q=0.5)

    # Output 2-by-n matrix of estimating equations
    return ((2*(y_array - theta[1])*(y_array > theta[1])) - theta[0],   # Estimating equation for positive mean dev
            median, )                                                   # Estimating equation for median
