#####################################################################################################################
# Estimating functions for basic statistics
#####################################################################################################################

import warnings

import numpy as np
from delicatessen.utilities import robust_loss_functions
from delicatessen.estimating_equations.processing import generate_weights


#################################################################
# Basic Estimating Equations


def ee_mean(theta, y, weights=None):
    r"""Estimating equation for the mean. The estimating equation for the mean is

    .. math::

        \sum_{i=1}^n (Y_i - \theta) = 0

    For the weighted mean, the difference in the previous estimating equation is multiplied by the corresponding weight.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in the case of the mean consists of a single value. Therefore, an initial value like the form of
        ``[0, ]`` should be provided.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta`` and ``y``

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

    # Allowing for a weighted mean
    w = generate_weights(weights=weights, n_obs=y_array.shape[0])

    # Output 1-by-n array estimating equation for the mean
    return w * (y_array - theta)


def ee_mean_robust(theta, y, k, loss='huber', lower=None, upper=None):
    r"""Estimating equation for the (unscaled) robust mean. The estimating equation for the robust
    mean is

    .. math::

        \sum_{i=1}^n f_k(Y_i - \theta) = 0

    where :math:`f_k(x)` is the corresponding robust loss function. Options for the loss function include: Huber,
    Tukey's biweight, Andrew's Sine, and Hampel. See ``robust_loss_function`` for further details on the loss
    functions for the robust mean.

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
        Robust loss function to use. Default is ``'huber'``. Options include ``'andrew'``, ``'hampel'``, ``'tukey'``.
    lower : int, float, None, optional
        Lower parameter for the Hampel loss function. This parameter does not impact the other loss functions.
        Default is ``None``.
    upper : int, float, None, optional
        Upper parameter for the Hampel loss function. This parameter does not impact the other loss functions.
        Default is ``None``.

    Returns
    -------
    array :
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta`` and ``y``.

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


def ee_mean_geometric(theta, y, weights=None, log_theta=True):
    r"""Estimating equations for the geometric mean. The geometric mean is defined as

    .. math::

        \bar{\mu} = \left( \prod_{i=1}^n Y_i \right)^{1/n}

    where :math:`Y_i` is within the positive reals. This expression can be rewritten as the following estimating
    equation

    .. math::

        \sum_{i=1}^n \left[ \log(Y_i) - \log(\hat{\mu}) \right] = 0

    For the weighted geometric mean, the difference in the previous estimating equation is multiplied by the
    corresponding weight.


    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in the case of the geomtric mean consists of a single value. Therefore, an initial value like the form of
        ``[1, ]`` should be provided.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.
    log_theta : bool, optional
        Whether to log-transform the input theta parameter internally. Default is True, which takes ``np.log(theta)``.
        The choice for this argument should not affect the point estimate, but it can change the confidence intervals.

    Returns
    -------
    array :
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta`` and ``y``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_mean_geometric`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean_geometric

    Some generic data to estimate the geometric mean for

    >>> y_dat = [10, 1, 2, 4, 1, 2, 3, 1, 5, 2, 33]

    Defining psi, or the stacked estimating equations for the geometric mean

    >>> def psi(theta):
    >>>     return ee_mean_geometric(theta=theta, y=y_dat)

    Calling the M-estimation procedure

    >>> estr = MEstimator(stacked_equations=psi, init=[2, ])
    >>> estr.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> estr.theta
    >>> estr.variance
    >>> estr.asymptotic_variance

    References
    ----------
    Lesage É. (2011). The use of estimating equations to perform a calibration on complex parameters.
    *Survey Methodology*, 37(1), 103-108.
    """
    # Convert input y values to NumPy array
    y_array = np.asarray(y)

    # Allowing for a weighted mean
    w = generate_weights(weights=weights, n_obs=y_array.shape[0])

    # Output 1-by-n array estimating equation for the mean
    if log_theta:
        return w * (np.log(y_array) - np.log(theta))
    else:
        return w * (np.log(y_array) - theta)


def ee_mean_variance(theta, y):
    r"""Estimating equations for the mean and variance. The estimating equations for the mean and
     variance are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            Y_i - \theta_1 \\
            (Y_i - \theta_1)^2 - \theta_2
        \end{bmatrix}
        = 0

    Unlike ``ee_mean``, ``theta`` consists of 2 parameters.

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
        Returns a 2-by-`n` NumPy array evaluated for the input ``theta`` and ``y``.

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

    For this estimating equation, ``estr.theta[1]`` and ``estr.asymptotic_variance[0][0]`` are expected to be equal.

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

        \sum_{i=1}^n \left\{ q - I(Y_i \le \theta) \right\} = 0

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
        Percentile to calculate. Must be :math:`(0, 1)`

    Returns
    -------
    array :
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta`` and ``y``.

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
    here. These changes generally work better for the percentile since the estimating equation is non-smooth.
    Furthermore, optimization is hard when only a few observations (<100) are available.

    >>> estr.theta

    Then displays the estimated percentile / median. In this example, there is a difference between the closed form
    solution (``-0.07978``) and M-Estimation (``-0.06022``).

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
        Returns a 2-by-`n` NumPy array evaluated for the input ``theta`` and ``y``.

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

    Inspecting the parameter estimates

    >>> estr.theta

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


def ee_meta_random(theta, point_est, var_est):
    r"""Estimating equation for random-effects meta-analysis using the Paule-Mandel method. This estimating equation
    allows for one to summarize multiple point estimates together by taking an inverse-variance weighted average.
    Importantly, this estimating equation also incorporates heterogeneity between studies via a random effect. The
    corresponding estimating equations are

    .. math::

        \sum_{j=1}^n
        \begin{bmatrix}
            \frac{1}{V_j + \tau^2} \times (E_j - \mu) \\
            \frac{(E_j - \mu)^2}{V_j + \tau^2} - \frac{k-1}{k}
        \end{bmatrix}
        = 0

    where :math:`\theta = (\mu, \tau^2)`, :math:`\mu` is the weighted mean across the studies, :math:`\tau^2` is the
    between-study heterogeneity term, :math:`E_j` is the point estimate from study :math:`j`, and :math:`V_j` is the
    variance estimate.

    Here, ``delicatessen`` solves for the log-transformed :math:`\tau^2` rather than :math:`\tau^2` directly. This is
    due to :math:`\tau^2 > 0` and the log constraint enforces this to be non-negative. Note that issues will arise
    if :math:`\tau^2 = 0`, but a random-effect model would not be appropriate in that setting.

    Note
    ----
    Estimates and variances are assumed to be on the same scale. Further, the estimates should be on a scale for which
    a linear combination is valid (e.g., log-transformed for ratio measures).


    Optimization can sometimes be a bit difficult due to the dependence on the inverse of the variance and
    :math:`\tau^2`. A piece of advice is to first estimate the inverse-variance-weighted mean for :math:`\mu` and then
    use that as the starting value.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 2 values.
    point_est : ndarray, list, vector
        1-dimensional vector of `s` point estimates. Note that these should be on a linear scale (e.g., risk or mean
        differences, log risk ratios)
    var_est : ndarray, list, vector
        1-dimensional vector of `s` corresponding variance estimates. Note that these should be on the same scale as
        those provided in ``point_est``.

    Returns
    -------
    array :
        Returns 2-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of estimating equations with ``ee_meta_random`` should be done similar to the following

    >>> import numpy as np
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_meta_random

    Some generic data to estimate a random-effects meta-analysis model with.

    >>> p_est = [-0.186, 0.235, 0.037, -0.904, 0.218, -0.135, 0.68, -1.254, -0.154, -1.12]
    >>> v_est = [0.183, 0.186, 0.054, 0.133, 0.182, 0.201, 0.154, 0.139, 0.115, 0.125]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_meta_random(theta=theta, point_est=p_est, var_est=v_est)

    Calling the M-estimation procedure (``init`` requires 2 values). As starting values, the mean of the point
    estimates for :math:`\mu` is reasonable. For :math:`\tau^2`, a starting value of 0 correspond to
    :math:`\exp(0) = 1`.

    >>> estr = MEstimator(psi, init=[np.mean(p_est), 0.])
    >>> estr.estimate()

    Inspecting the parameter estimates

    >>> estr.theta[0]  # Estimate for mu          (-0.269)
    >>> estr.theta[1]  # Estimate for log(tau^2)  (-1.346)

    When considering ratio measures, the meta-analysis model should be fit using the log-transformed point estimates
    (and the variance estimate for the log-ratio). ``delicatessen`` does not implement these transformations and
    expects the user to provide estimates in a format for which the mean is an appropriate summary.

    References
    ----------
    Paule RC & Mandel J. (1982). Consensus values and weighting factors.
    *Journal of research of the National Bureau of Standards*, 87(5), 377.
    """
    # Preparation of input shapes and object types
    est_p = np.asarray(point_est)                            # Converting to NumPy array
    est_v = np.asarray(var_est)                              # Converting to NumPy array
    mu, tau_sq = theta                                       # Subsetting parameters to informative names
    k = len(est_p)                                           # Number of observations
    tau_sq = np.exp(tau_sq)

    # Constructing study-specific weight
    weight = 1 / (est_v + tau_sq)                            # Weight the incorporates between-study heterogeneity

    # Computing estimating functions
    ee_beta = weight * (est_p - mu)                          # Estimating function for overall mean
    ee_tau = ((est_p - mu)**2 / (est_v + tau_sq)) - (k-1)/k  # Estimating function for between-study heterogeneity

    # Returning stacked estimating functions
    return np.vstack([ee_beta, ee_tau])
