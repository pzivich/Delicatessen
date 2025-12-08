#####################################################################################################################
# Functionality to compute the sandwich
#   This script allows for computation of the empirical sandwich variance estimator with just the
#   parameter values and estimating equations. This is to allow computing the sandwich quickly without
#   called the MEstimator procedure itself.
#####################################################################################################################

import warnings

import numpy as np
from scipy.optimize import approx_fprime
from scipy.stats import norm

from delicatessen.errors import check_alpha_level
from delicatessen.derivative import auto_differentiation, approx_differentiation


def compute_sandwich(stacked_equations, theta, deriv_method='approx', dx=1e-9, allow_pinv=True, small_n_adjust=None):
    r"""Compute the empirical sandwich variance estimator given a set of estimating equations and parameter estimates.
    Note that this functionality does not solve for the parameter estimates (unlike ``MEstimator``). Instead, it
    only computes the sandwich for the provided value.

    The empirical sandwich variance estimator is defined as

    .. math::

        V_n(O_i; \theta) = B_n(O_i; \theta)^{-1} F_n(O_i; \theta) \left[ B_n(O_i; \theta)^{-1} \right]^{T}

    where :math:`\psi(O_i; \theta)` is the estimating function,

    .. math::

        B_n(O_i; \theta) = \sum_{i=1}^n \frac{\partial}{\partial \theta} \psi(O_i; \theta),

    and

    .. math::

        F_n(O_i; \theta) = \sum_{i=1}^n \psi(O_i; \theta) \psi(O_i; \theta)^T .

    To compute the bread matrix, :math:`B_n`, the matrix of partial derivatives is computed by using either finite
    difference methods or automatic differentiation. For finite differences, the default is to use SciPy's
    ``approx_fprime`` functionality, which uses forward finite differences. However, you can also use the delicatessen
    homebrew version that allows for forward, backward, and center differences. Automatic differentiation is also
    supported by a homebrew version.

    To compute the meat matrix, :math:`F_n`, only linear algebra methods, implemented through NumPy, are necessary.
    The sandwich is then constructed from these pieces using linear algebra methods from NumPy.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    theta : list, set, array
        Parameter estimates to compute the empirical sandwich variance estimator at. Note that this function assumes
        that you have solved for the ``theta`` that correspond to the root of the input estimating equations.
    deriv_method : str, optional
        Method to compute the derivative of the estimating equations for the bread matrix. Options include numerical
        approximation via the forward difference method via SciPy (``'approx'``), forward difference implemented by-hand
        (`'fapprox'`), backward difference implemented by-hand (`'bapprox'`),  central difference implemented by-hand
        (`'capprox'`), or forward-mode automatic differentiation (``'exact'``). Default is ``'approx'``.
    dx : float, optional
        Spacing to use to numerically approximate the partial derivatives of the bread matrix. Here, a small value
        for ``dx`` should be used, since some large values can result in poor approximations. This argument is only
        used with numerical approximation methods. Default is ``1e-9``.
    allow_pinv : bool, optional
        Whether to allow for the pseudo-inverse (via ``numpy.linalg.pinv``) if the bread matrix is determined to be
        non-invertible. If you want to disallow the pseudo-inverse (i.e., use ``numpy.linalg.inv``), set this
        argument to ``False``. Default is ``True``, which  is more robust to the possible bread matrices.
    small_n_adjust : str, None, optional
        Whether to apply a finite-sample correction when computing the empirical sandwich variance estimator. Default is
        ``None``, which applies no correction. Corrections options include: HC1. The HC1 correction replaces the
        scaling by :math:`n` with :math:`n-p` where :math:`p` is the number of parameters.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``

    Examples
    --------
    Loading necessary functions and building a generic data set for estimation of the mean

    >>> import numpy as np
    >>> from delicatessen import MEstimator
    >>> from delicatessen import compute_sandwich
    >>> from delicatessen.estimating_equations import ee_mean_variance

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    The following is an illustration of how to compute sandwich covariance using only an estimating equation and the
    parameter values. The mean and variance (that correspond to ``ee_mean_variance``) can be computed using NumPy by

    >>> mean = np.mean(y_dat)
    >>> var = np.var(y_dat, ddof=0)

    For the corresponding estimating equation, we can use the built-in functionality as done below

    >>> def psi(theta):
    >>>     return ee_mean_variance(theta=theta, y=y_dat)

    Calling the sandwich computation procedure

    >>> sandwich_asymp = compute_sandwich(stacked_equations=psi, theta=[mean, var])

    The output sandwich is the *asymptotic* variance (or the variance that corresponds to the standard deviation). To
    get the variance (or the variance that corresponds to the standard error), we rescale ``sandwich`` by the number of
    observations

    >>> sandwich = sandwich_asymp / len(y_dat)

    The standard errors are then

    >>> se = np.sqrt(np.diag(sandwich))

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Ross RK, Zivich PN, Stringer JSA, & Cole SR. (2024). M-estimation for common epidemiological measures: introduction
    and applied examples. *International Journal of Epidemiology*, 53(2), dyae030.

    Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. The American Statistician, 56(1), 29-38.
    """
    # Evaluating at provided theta values
    evald_theta = np.asarray(stacked_equations(theta=theta))        # Evaluating EE at theta-hat
    if len(theta) == 1:                                             # Number of parameters
        n_obs = evald_theta.shape[0]                                # ... to get number of obs
        n_params = 1                                                # ... and number of parameters
    else:                                                           # Number of parameters
        n_obs = evald_theta.shape[1]                                # ... to get number of obs
        n_params = evald_theta.shape[0]                             # ... and number of parameters

    # Step 1: Compute the bread matrix
    bread = compute_bread(stacked_equations=stacked_equations,      # Call the bread matrix function
                          theta=theta,                              # ... at given theta-hat
                          deriv_method=deriv_method,                # ... with derivative method
                          dx=dx)                                    # ... and approximation
    bread = bread / n_obs                                           # Scale bread by number of obs

    # Step 2: Compute the meat matrix
    meat = compute_meat(stacked_equations=stacked_equations,        # Call the meat matrix function
                        theta=theta)                                # ... at given theta-hat
    meat = meat / n_obs                                             # Scale meat by number of obs

    if small_n_adjust is not None:
        n_adjust = small_n_adjust.upper()
        if n_adjust.upper() == "HC1":
            meat = meat * n_obs / (n_obs - n_params)
        else:
            raise ValueError("The requested finite-sample correction '" + str(small_n_adjust) + "' is not available. "
                             "Supported options include the following: None, HC1.")

    # Step 3: Construct sandwich from the bread and meat matrices
    sandwich = build_sandwich(bread=bread,                          # Call the sandwich constructor
                              meat=meat,                            # ... with bread and meat matrices above
                              allow_pinv=allow_pinv)                # ... and whether to allow pinv

    # Return the constructed empirical sandwich variance estimator
    return sandwich


def compute_bread(stacked_equations, theta, deriv_method, dx=1e-9):
    r"""Function to compute the bread matrix. The bread matrix is defined as

    .. math::

        B_n(O_i; \theta) = \sum_{i=1}^n \frac{\partial }{\partial \theta} \psi(O_i; \theta)

    where :math:`\psi(O_i; \theta)` is the estimating function.
    To compute the bread matrix, :math:`B_n`, the matrix of partial derivatives is computed by using either finite
    difference methods or automatic differentiation. For finite differences, the default is to use SciPy's
    ``approx_fprime`` functionality, which uses forward finite differences. However, you can also use the delicatessen
    homebrew version that allows for forward, backward, and center differences. Automatic differentiation is also
    supported by a homebrew version.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    theta : list, set, array
        Parameter estimates to compute the empirical sandwich variance estimator at. Note that this function assumes
        that you have solved for the ``theta`` that correspond to the root of the input estimating equations.
    deriv_method : str, optional
        Method to compute the derivative of the estimating equations for the bread matrix. Options include numerical
        approximation via the forward difference method via SciPy (``'approx'``), forward difference implemented by-hand
        (`'fapprox'`), backward difference implemented by-hand (`'bapprox'`),  central difference implemented by-hand
        (`'capprox'`), or forward-mode automatic differentiation (``'exact'``). Default is ``'approx'``.
    dx : float, optional
        Spacing to use to numerically approximate the partial derivatives of the bread matrix. Here, a small value
        for ``dx`` should be used, since some large values can result in poor approximations. This argument is only
        used when numerical approximation methods. Default is ``1e-9``.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``
    """
    def estimating_equation(input_theta):
        ef = np.asarray(stacked_equations(theta=input_theta))
        if ef.ndim == 1:
            return np.sum(ef)
        else:
            return np.sum(ef, axis=1)

    # Compute the derivative
    if deriv_method.lower() == 'approx':
        bread_matrix = approx_fprime(xk=theta,
                                     f=estimating_equation,
                                     epsilon=dx)
        if bread_matrix.ndim == 1:
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
    r"""Function to compute the meat matrix. The meat matrix is defined as

    .. math::

        F_n(O_i; \theta) = \sum_{i=1}^n \psi(O_i; \theta) \psi(O_i; \theta)^T

    where :math:`\psi(O_i; \theta)` is the estimating function.
    Rather than summing over all the individual contributions, this implementation takes a single dot product of the
    stacked estimating functions. This implementation is much faster than summing over :math:`n` matrices.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    theta : list, set, array
        Parameter estimates to compute the empirical sandwich variance estimator at. Note that this function assumes
        that you have solved for the ``theta`` that correspond to the root of the input estimating equations.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``
    """
    evald_theta = np.asarray(stacked_equations(theta=theta))  # Evaluating EE at theta-hat
    return np.dot(evald_theta, evald_theta.T)                 # Return the fast dot product calculation


def build_sandwich(bread, meat, allow_pinv=True):
    r"""Function to combine the sandwich elements together. This function takes the bread and meat matrices, does the
    inversions, and then combines them together. This function is separate from ``compute_sandwich`` as it is called
    by both ``compute_sandwich`` and ``MEstimator``.

    Parameters
    ----------
    bread : ndarray
        The bread matrix. The expected input is the output from the ``compute_bread`` function
    meat : ndarray
        The meat matrix. The expected input is the output from the ``compute_meat`` function
    allow_pinv : bool, optional
        Whether to allow for the pseudo-inverse (via ``numpy.linalg.pinv``) if the bread matrix is determined to be
        non-invertible. If you want to disallow the pseudo-inverse (i.e., use ``numpy.linalg.inv``), set this
        argument to ``False``. Default is ``True``, which  is more robust to the possible bread matrices.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where ``p = len(theta)``
    """
    # Check if there is an issue with the bread matrix
    if np.any(np.isnan(bread)):                                   # If bread contains NaN, breaks
        return None                                               # ... so give back a NaN

    # Compute the bread inversion
    if allow_pinv:                                                 # Allowing the pseudo-inverse
        bread_invert = np.linalg.pinv(bread)                       # ... then call pinv
    else:                                                          # Only allowing the actual inverse
        bread_invert = np.linalg.inv(bread)                        # ... then call inv

    # Compute the sandwich variance
    sandwich = np.dot(np.dot(bread_invert, meat), bread_invert.T)

    # Return the sandwich covariance matrix
    return sandwich


def delta_method(theta, g, covariance, deriv_method='exact', dx=1e-9):
    r"""Function to apply the Delta Method for a given parameter vector and transformation. The Delta Method is defined
    as

    .. math::

        Var \left[ g(\theta) \right] \approx g'(\theta) \; \Sigma_{\theta} \; g'(\theta)^T

    where :math:`\theta` is the parameter vector, :math:`g` is a vector-valued function that returns a 1 dimensional
    vector, :math:`g'` is the gradient (or partial derivatives), and :math:`\Sigma_{\theta}` is the covariance matrix
    for :math:`\theta`. In words, the variance of the transformation of the parameters is equal to covariance of the
    parameters sandwiched between the derivatives of the transformation functions. This expression then provides a
    variance estimator when replacing :math:`\theta` and :math:`\Sigma` with :math:`\hat{\theta}` and
    :math:`\hat{\Sigma}`, respectively.

    As described elsewhere, the sandwich variance estimator automates the Delta Method. Therefore, one can simply
    program the corresponding estimating equation to estimate the variance for that transformation. However, this can be
    computationally inefficient when :math:`g` outputs a large vector. This functionality offers a way to apply the
    Delta Method outside of ``MEstimator`` and ``GMMEstimator``. Internally, this function is used for some prediction
    functionalities.

    Parameters
    ----------
    theta : ndarray, list, set
        Parameter vector of dimension `v` to apply the transformation function ``g`` with.
    g : function, callable
        Vector function that transforms the `v` dimension parameter vector ``theta`` into a `w` dimensional vector.
    covariance : ndarray, list, set
        Covariance matrix for the parameter vector ``x``.
    deriv_method : str, optional
        Method to compute the derivative of the function ``g``. Default is ``'exact'``. Options include numerical
        approximation via the forward difference method via SciPy (``'approx'``), forward difference implemented by-hand
        (`'fapprox'`), backward difference implemented by-hand (`'bapprox'`),  central difference implemented by-hand
        (`'capprox'`), or forward-mode automatic differentiation (``'exact'``).
    dx : float, optional
        Spacing to use to numerically approximate the partial derivatives of the bread matrix. Here, a small value
        for ``dx`` should be used, since some large values can result in poor approximations. This argument is only
        used when numerical approximation methods. Default is ``1e-9``.

    Returns
    -------
    array :
        Returns a `p`-by-`p` NumPy array for the input ``theta``, where `p` is the length of the output vector from the
        function :math:`g`

    Examples
    --------
    Using ``delta_method`` to compute the variance should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator, delta_method
    >>> from delicatessen.estimating_equations import ee_ipw

    To illustrate how ``delta_method`` is intended to be used, we will first use an M-estimator to compute the point
    and variance estimates for the parameter vector :math:`\theta`. Here, we will replicate the example from the
    documentation for ``ee_ipw``.

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_ipw(theta, y=d['Y'], A=d['A'],
    >>>                   W=d[['C', 'W']])

    Calling the M-estimation procedure

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0.])
    >>> estr.estimate(solver='lm')

    Per the documentation for ``ee_ipw``, the average causal effect (or risk difference) is given by the following

    >>> estr.theta[0]        # causal mean difference of 1 versus 0
    >>> estr.variance[0, 0]  # corresponding variance estimate

    Suppose that ``ee_ipw`` did not directly provide the risk difference. Further, imagine we were interested in the
    risk ratio (log-transformed). Both of these quantities are transformations of the risk under action 1
    (i.e., ``theta[1]``) and the risk under action 0 (i.e., ``theta[2]``). The following is a function that applies and
    returns those transformations

    >>> def causal_contrasts(theta):
    >>>     risk1, risk0 = theta
    >>>     risk_diff = risk1 - risk0
    >>>     log_risk_ratio = np.log(risk1) - np.log(risk0)
    >>>     return risk_diff, log_risk_ratio

    To estimate the variance for this transformation, one can now use the *delta method*. While one could manually
    compute the derivatives for this function, ``delta_method`` automates this procedure for you (using either automatic
    or numerical approximation methods). Below is how ``delta_method`` can be applied to compute the variance for the
    risk difference and risk ratio

    >>> risks = estr.theta[1:3]                     # Risks
    >>> risks_c0var = estr.variance[1:3, 1:3]       # Variance for risks
    >>> rd, log_rr = causal_contrasts(theta=risks)  # RD, log(RR)

    >>> covar = delta_method(theta=risks,             # Delta method
    >>>                      g=causal_contrasts,      # ... function g
    >>>                      covariance=risks_c0var)  # ... covariance

    The output from ``delta_method`` is the corresponding covariance matrix for risk difference and risk ratio. We can
    get the corresponding 95% confidence intervals via

    >>> rd_stderr = np.sqrt(covar[0, 0])
    >>> rd_lcl, rd_ucl = rd - 1.96*rd_stderr, rd + 1.96*rd_stderr
    >>> rr_stderr = np.sqrt(covar[1, 1])
    >>> rr_lcl, rr_ucl = np.exp(log_rr - 1.96*rr_stderr), np.exp(log_rr + 1.96*rr_stderr)

    While these transformations are straightforward to stack as estimating functions, ``delta_method`` offers another
    option for estimating the variance of transformations that has computational benefits in some settings.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). Large Sample Theory: The Basics, In
    *Essential Statistical Inference: Theory and Methods*, 237-240.

    Cox C. (2005). Delta method. *Encyclopedia of Biostatistics*.

    Oehlert GW. (1992). A Note on the Delta Method. *The American Statistician*, 46(1), 27-29.
    """
    # Setup
    covariance = np.asarray(covariance)
    j = covariance.shape[0]
    k = covariance.shape[1]

    def g_array(theta):
        # Function to convert user function g to always return a NumPy array
        return np.asarray(g(theta))

    # Computing g(\theta)
    theta_star = g_array(theta)
    theta_dim = theta_star.shape
    v = len(theta)

    # Checking dimensions of parameter vector
    if len(theta_dim) > 1:
        raise ValueError("Output from function `g` must be a one-dimensional array")
    if j != k:
        raise ValueError("Input covariance matrix must be symmetric. Input matrix had dimensions " + str(j)
                         + " rows and " + str(k) + " columns")
    if j != v:
        raise ValueError("Input parameter vector and covariance matrix must share a dimension, but theta has " + str(v)
                         + " as its dimension and the matrix had dimension " + str(j))

    # Computing derivative for g(x)
    if deriv_method.lower() == "exact":
        g_prime = auto_differentiation(xk=theta, f=g_array)
    elif deriv_method.lower() == 'approx':
        g_prime = approx_fprime(xk=theta, f=g_array, epsilon=dx)
        if g_prime.ndim == 1:
            g_prime = np.asarray([g_prime, ])
    elif deriv_method.lower() in ['capprox', 'fapprox', 'bapprox']:
        g_prime = approx_differentiation(xk=theta, f=g_array, method=deriv_method.lower(), epsilon=dx)
    else:
        raise ValueError("The input for deriv_method was " + str(deriv_method)
                         + ", but only 'approx', 'fapprox', 'capprox', 'bapprox' and 'exact' are available.")

    # Applying delta method to get the variance
    covariance_g = np.dot(np.dot(g_prime, covariance), g_prime.T)

    # Returning the covariance matrix for g(x)
    return covariance_g


def compute_critical_value_bands(theta, covariance, alpha=0.05, method='supt', n_draws=100000, seed=None):
    """Function to compute the critical value for parameter vectors.

    Confidence bands are an extension of confidence intervals. Confidence intervals claim to cover the true parameter
    under infinite repetitions at the :math:`1-\alpha` proportion. However, when considering vectors of parameters,
    confidence interval coverage is below this proportion (can be thought of as a multiple-testing problem). Confidence
    bands instead work to ensure :math:`1-\alpha` coverage of *the parameter vector*. Many of the algorithms for
    confidence bands operate by adjusting the critical value. This function offers several methods to compute the
    critical value for confidence bands

    Parameters
    ----------
    theta : ndarray, list, set
        Parameter vector of dimension `v` to compute confidence bands for. This vector should consist of all parameters
        that inference is being jointly drawn for.
    covariance : ndarray, list, set
        Covariance matrix of dimension `v`. This should be the empirical sandwich covariance matrix for the parameter.
    alpha : float, optional
        The :math:`0 < \alpha < 1` level for the corresponding confidence bands. Default is 0.05, which
        corresponds to 95% confidence bands.
    method : str, optional
        Method to compute the confidence bands. Currently, only the sup-t and Bonferroni method are supported.
        Default is ``'supt'``
    n_draws : int, optional
        Number of random draws to use for any methods based on simulated approximation. Default is ``100000``.
    seed : int, optional
        Seed to intialize a pseudo RNG for methods based on simulated approximations. Default is ``None``
        which does not use a reproducible seed. To consistently obtain the exact same confidence bands, please use
        a seed.

    Returns
    -------
    float :
        Corresponding critical value for given method and inputs

    References
    ----------
    Montiel Olea JL & Plagborg‐Møller M. (2019). Simultaneous confidence bands: Theory, implementation, and an
    application to SVARs. *Journal of Applied Econometrics*, 34(1), 1-17.

    Zivich PN, Cole SR, Greifer N, Montoya LM, Kosorok MR, Edwards JK. (2025). Confidence Regions for Multiple
    Outcomes, Effect Modifiers, and Other Multiple Comparisons, *arXiv:2510.07076*
    """
    # Pre-processing inputs
    theta = np.asarray(theta)
    covariance = np.asarray(covariance)

    # Error checking for inputs
    if len(theta.shape) != 1:
        raise ValueError("The input theta vector must be a 1-dimensional vector. Instead, the input theta vector "
                         "has a dimension of " + str(len(theta.shape)))
    if len(covariance.shape) != 2:
        raise ValueError("The input covariance matrix must be a 2-dimensional matrix. Instead, the input covariance "
                         "matrix has a dimension of " + str(len(covariance.shape)))
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("The input covariance matrix must be square (i.e., have the same number of rows and columns). "
                         "Instead, the input covariance matrix has " + str(covariance.shape[0]) + " rows and "
                         + str(covariance.shape[1]) + " columns.")
    if theta.shape[0] != covariance.shape[0]:
        raise ValueError("The dimension of the theta vector and covariance matrix must be equal. Instead, the theta "
                         "vector has " + str(theta.shape[0]) + " elements and the covariance matrix is dimension "
                         + str(covariance.shape[0]))
    check_alpha_level(alpha=alpha)

    # Processing
    stderr = np.diag(covariance) ** 0.5

    # Approximate c
    rng = np.random.default_rng(seed=seed)
    k = len(theta)
    if method.lower() in ['supt', 'sup-t']:
        mvn = rng.multivariate_normal([0., ] * k, cov=covariance, size=n_draws)
        if (stderr <= 0).any():
            raise ValueError("There is at least one parameter with a standard error of zero or less. The sup-t method "
                             "cannot be applied as it would require division by zero for the rescaling process.")
        scaled_mvn = np.abs(mvn / stderr)
        ts = np.max(scaled_mvn, axis=1)
        critical_value = np.percentile(ts, q=(1 - alpha) * 100)
    elif method.lower() == 'bonferroni':
        critical_value = norm.ppf(1 - alpha / (2 * k), loc=0, scale=1)
    else:
        raise ValueError("The method '" + str(method) + "' was specified, but only the following "
                         "methods are supported: 'supt', 'bonferroni'.")

    # Returning the corresponding critical value
    return critical_value


def compute_confidence_bands(theta, covariance, alpha=0.05, method='supt', n_draws=100000, seed=None):
    r"""Function to compute the confidence bands for a given parameter vector and covariance matrix.

    Confidence bands are an extension of confidence intervals. Confidence intervals claim to cover the true parameter
    under infinite repetitions at the :math:`1-\alpha` proportion. However, when considering vectors of parameters,
    confidence interval coverage is below this proportion (can be thought of as a multiple-testing problem). Confidence
    bands instead work to ensure :math:`1-\alpha` coverage of *the parameter vector*. Many of the algorithms for
    confidence bands operate by adjusting the critical value.

    Parameters
    ----------
    theta : ndarray, list, set
        Parameter vector of dimension `v` to compute confidence bands for. This vector should consist of all parameters
        that inference is being jointly drawn for.
    covariance : ndarray, list, set
        Covariance matrix of dimension `v`. This should be the empirical sandwich covariance matrix for the parameter.
    alpha : float, optional
        The :math:`0 < \alpha < 1` level for the corresponding confidence bands. Default is 0.05, which
        corresponds to 95% confidence bands.
    method : str, optional
        Method to compute the confidence bands. Currently, only the sup-t and Bonferroni method are supported.
        Default is ``'supt'``
    n_draws : int, optional
        Number of random draws to use for any methods based on simulated approximation. Default is ``100000``.
    seed : int, optional
        Seed to intialize a pseudo RNG for methods based on simulated approximations. Default is ``None``
        which does not use a reproducible seed. To consistently obtain the exact same confidence bands, please use
        a seed.

    Returns
    -------
    array :
        b-by-2 array, where row 1 is the confidence intervals for :math:`\theta_1`, ..., and row b is the confidence
        intervals for :math:`\theta_b`

    Examples
    --------

    References
    ----------
    Montiel Olea JL & Plagborg‐Møller M. (2019). Simultaneous confidence bands: Theory, implementation, and an
    application to SVARs. *Journal of Applied Econometrics*, 34(1), 1-17.

    Zivich PN, Cole SR, Greifer N, Montoya LM, Kosorok MR, Edwards JK. (2025). Confidence Regions for Multiple
    Outcomes, Effect Modifiers, and Other Multiple Comparisons, *arXiv:2510.07076*
    """
    # Computing the critical value
    critical_value = compute_critical_value_bands(theta=theta, covariance=covariance,
                                                  alpha=alpha, method=method,
                                                  n_draws=n_draws, seed=seed)

    # Processing inputs
    theta = np.asarray(theta)
    covariance = np.asarray(covariance)
    stderr = np.diag(covariance) ** 0.5

    # Computing the corresponding confidence bands
    conf_bands = np.asarray([theta - critical_value * stderr,
                             theta + critical_value * stderr])

    # Returning confidence bands
    return conf_bands.T
