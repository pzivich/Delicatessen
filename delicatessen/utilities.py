import warnings
import numbers
import numpy as np
import scipy as sp
from scipy.stats import norm

from delicatessen.derivative import PrimalTangentPairs as PTPair
from delicatessen.errors import check_alpha_level
from delicatessen.sandwich import delta_method
from delicatessen.helper import convert_survival_measures


def logit(prob):
    r"""Logistic transformation. Used to transform probabilities into log-odds.

    .. math::

        \log \left( \frac{p}{1-p} \right)

    Parameters
    ----------
    prob : float, ndarray, list
        A single probability or an array of probabilities

    Returns
    -------
    array :
        logit-transformed values
    """
    p = np.asarray(prob)
    return np.log(p / (1 - p))


def inverse_logit(logodds):
    r"""Inverse logistic transformation. Used to transform log-odds into probabilities.

    .. math::

        \frac{1}{1 + \exp(o)}

    Parameters
    ----------
    logodds : float, ndarray, list
        A single log-odd or an array of log-odds

    Returns
    -------
    array :
        inverse-logit transformed values
    """
    lodds = np.asarray(logodds)
    return 1 / (1 + np.exp(-lodds))


def identity(value):
    """Identity transformation. Used to transform input into itself (i.e., no transformation in applied).

    Note
    ----
    This function doesn't actually apply any transformation. It is used for arbitrary function calls that apply
    transformations, and this is called when no transformation is to be applied

    Parameters
    ----------
    value : float, ndarray, list
        A single value or an array of values

    Returns
    -------
    value
    """
    return value


def polygamma(n, x):
    """Polygamma function. This is a wrapper function of ``scipy.special.polygamma`` meant to enable automatic
    differentation with ``delicatessen``. When the input is a ``PrimalTangentPairs`` object, then an internal function
    that implements the polygamma function is called. Otherwise, ``scipy.special.polygamma`` is called for the input
    object.

    Parameters
    ----------
    n : int
        Order of the derivative of the digamma function
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    if isinstance(x, np.ndarray):
        storage = []
        for xi in x:
            pgnxi = polygamma(n=n, x=xi)
            storage.append(pgnxi)
        return np.array(storage)
    elif isinstance(x, PTPair):
        return x.polygamma(n=n)
    else:
        return sp.special.polygamma(n=n, x=x)


def digamma(z):
    """Digamma function. This is a wrapper function of ``scipy.special.digamma`` meant to enable automatic
    differentation with ``delicatessen``. When the input is a ``PrimalTangentPairs`` object, then an internal function
    that implements the digamma function is called. Otherwise, ``scipy.special.digamma`` is called for the input
    object.

    Parameters
    ----------
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    return polygamma(n=0, x=z)


def standard_normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution. This is a wrapper function of
    ``scipy.stats.norm.cdf`` meant to enable automatic differentation with ``delicatessen``. When the input is a
    ``PrimalTangentPairs`` object, then an internal function that implements the CDF function is called. Otherwise,
    ``scipy.stats.norm.cdf`` is called for the input object.

    Parameters
    ----------
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    if isinstance(x, np.ndarray):
        storage = []
        for xi in x:
            normxi = standard_normal_cdf(x=xi)
            storage.append(normxi)
        return np.array(storage)
    elif isinstance(x, PTPair):
        return x.normal_cdf()
    else:
        return norm.cdf(x=x)


def standard_normal_pdf(x):
    """Probability density function for the standard normal distribution. This is a wrapper function of
    ``scipy.stats.norm.pdf`` meant to enable automatic differentation with ``delicatessen``. When the input is a
    ``PrimalTangentPairs`` object, then an internal function that implements the PDF function is called. Otherwise,
    ``scipy.stats.norm.pdf`` is called for the input object.

    Parameters
    ----------
    x : int, float, ndarray
        Real valued input

    Returns
    -------
    Return type depends on the input type (``PrimalTangentPairs`` will return ``PrimalTangentPairs``, otherwise will
    return ``ndarray``).
    """
    if isinstance(x, np.ndarray):
        storage = []
        for xi in x:
            normxi = standard_normal_pdf(x=xi)
            storage.append(normxi)
        return np.array(storage)
    elif isinstance(x, PTPair):
        return x.normal_pdf()
    else:
        return norm.pdf(x=x)


def robust_loss_functions(residual, loss, k, a=None, b=None):
    r"""Loss functions for robust mean and robust regression estimating equations. This function is called internally
    for ``ee_mean_robust`` and ``ee_robust_regression``. This function can also be accessed, so user's can easily adapt
    their own regression models into robust regression models using the pre-defined loss functions.

    Note
    ----
    The loss functions here are technically the first-order derivatives of the loss functions you see in the literature.


    The following score of the loss functions, :math:`f_k()`, are available.

    Andrew's Sine

    .. math::

        f_k(x) = I(k \pi \le x \le k \pi) \times \sin(x/k)

    Huber

    .. math::

        f_k(x) = x I(-k < x < k) + \text{sign}(x) k (1 - I(-k < x < k))

    Tukey's biweight

    .. math::

        f_k(x) = x I(-k < x < k) + x \left( 1 - (x/k)^2 \right)^2

    Fair

    .. math::

        f_k(x) = \frac{x}{1 + |x|/k}

    Cauchy

    .. math::

        f_k(x) = \frac{x}{1 + (x/k)^2}

    Ullah

    .. math::

        f_k(x) = x \left[ 1 + (x/k)^4 \right]^-2

    Welsch

    .. math::

        f_k(x) = x \exp(-x^2 / (2k^2))

    Hampel (Hampel's requires two additional parameters, :math:`a` and :math:`b`)

    .. math::

        f_{k,a,b}(x) =
        \begin{bmatrix}
            I(-a < x < a) \times x \\
            + I(a \le |x| < b) \times a \times \text{sign}(x) \\
            + I(b \le x < k) \times a \frac{k - x}{k - b} \\
            + I(-k \ge x > -b) \times -a \frac{-k + x}{-k + b} \\
            + I(|x| \ge k) \times 0
        \end{bmatrix}

    Parameters
    ----------
    residual : ndarray, vector, list
        1-dimensional vector of n observed values. Input should consists of the residuals (the difference between the
        observed value and the predicted value). For the robust mean, this is :math:`Y_i - \mu`. For robust regression,
        this is :math:`Y_i - X_i^T \beta`
    loss : str
        Loss function to use. Options include: `'andrew'`, `'huber'`, `'tukey'`, `'fair'`, `'cauchy'`, `'ullah'`,
        `'welsch'`, `'hampel'`
    k : int, float
        Tuning parameter for the corresponding loss function. Note: no default is provided, since each loss function
        has different recommendations.
    a : int, float, None, optional
        Lower parameter for the 'hampel' loss function
    b : int, float, None, optional
        Upper parameter for the 'hampel' loss function

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and residual

    Examples
    --------
    Using the robust loss function

    >>> import numpy as np
    >>> from delicatessen.utilities import robust_loss_functions

    Some generic data to stand-in for the residuals

    >>> residuals = np.random.standard_cauchy(size=20)

    Huber's loss function

    >>> robust_loss_functions(residuals, loss='huber', k=1.345)

    Andrew's Sine

    >>> robust_loss_functions(residuals, loss='andrew', k=1.339)

    Tukey's biweight

    >>> robust_loss_functions(residuals, loss='tukey', k=4.685)

    Fair

    >>> robust_loss_functions(residuals, loss='fair', k=1.3998)

    Cauchy

    >>> robust_loss_functions(residuals, loss='cauchy', k=2.3849)

    Ullah

    >>> robust_loss_functions(residuals, loss='ullah', k=3.2296)

    Welsch

    >>> robust_loss_functions(residuals, loss='welsch', k=2.9846)

    Hampel's loss function

    >>> robust_loss_functions(residuals, loss='hampel', k=8, a=2, b=4)


    References
    ----------
    Andrews DF. (1974). A robust method for multiple linear regression. *Technometrics*, 16(4), 523-531.

    Beaton AE & Tukey JW (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic
    data. *Technometrics*, 16(2), 147-185.

    Hampel FR. (1971). A general qualitative definition of robustness. *The Annals of Mathematical Statistics*,
    42(6), 1887-1896.

    Huber PJ. (1964). Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101.

    Huber PJ, Ronchetti EM. (2009) *Robust Statistics* 2nd Edition. Wiley. pgs 98-100

    de Menezes DQF, Prata DM, Secchi AR, & Pinto JC. (2021). A review on robust M-estimators for regression analysis.
    *Computers & Chemical Engineering*, 147, 107254.

    Rey WJ. (1983). Type M estimators. In *Introduction to Robust and Quasi-Robust Statistical Methods* (pp. 134-189).
    Berlin, Heidelberg: Springer Berlin Heidelberg.
    """
    # Checking type for later .lower() calls so informative error
    if not isinstance(loss, str):
        raise ValueError("The provided loss function should be a string.")

    loss_l = loss.lower()

    # Huber function
    if loss_l == "huber":
        xr = np.clip(residual, a_min=-k, a_max=k)

    # Tukey's biweight
    elif loss_l == "tukey":
        xr = np.where(np.abs(residual) <= k, residual * (1-(residual/k)**2)**2, 0)

    # Andrew's Sine
    elif loss_l == "andrew":
        xr = np.where(np.abs(residual) <= k*np.pi,
                      np.sin(residual/k), np.nan)
        xr = np.where(residual > k*np.pi, 0, xr)
        xr = np.where(residual < -k*np.pi, 0, xr)

    # Fair
    elif loss_l == "fair":
        xr = residual / (1 + np.abs(residual) / k)

    # Cauchy
    elif loss_l == "cauchy":
        xr = residual / (1 + (residual / k)**2)

    # Ullah
    elif loss_l == "ullah":
        xr = residual * (1 + (residual/k)**4)**(-2)

    # Welsch
    elif loss_l == "welsch":
        xr = residual * np.exp(-residual**2 / (2 * k**2))

    # Hampel
    elif loss_l == "hampel":
        if a is None or b is None:
            raise ValueError("The 'hampel' loss function requires the optional `a` and `b` arguments")
        if not a < b < k:
            raise ValueError("The 'hampel' loss function requires that a < b < k")
        xr = np.where(np.abs(residual) < a, residual, np.nan)
        xr = np.where((a <= residual) & (residual < b), a, xr)
        xr = np.where((-a >= residual) & (residual > -b), -a, xr)
        xr = np.where((b <= residual) & (residual < k), (k - residual)/(k - b)*a, xr)
        xr = np.where((-b >= residual) & (residual > -k), (-k - residual)/(-k + b)*-a, xr)
        xr = np.where(np.abs(residual) >= k, 0, xr)

    # Catch-all value error for the provided loss function
    else:
        raise ValueError("The loss function `"+str(loss)+"` is not available.")

    # Returning the updated values
    return xr


def aggregate_efuncs(est_funcs, group):
    r"""Aggregate estimating function contributions from the individual-level to the specified group-level. This
    function is intended to simply estimation with grouped or clustered data. Briefly, the input matrix of estimating
    function contributions is collapsed from `n` unit-level contributions into `m` group-level contributions under the
    assumption that observations within groups are independent (see later notes). This function should be used whenever
    observations are not independent and there is a group-level ID variable for appropriate statistical inference.

    This function is intended to be called after the estimating functions have been stacked, but before they are
    returned in a ``psi`` function. See the examples below for details on the intended use.

    Note
    ----
    Here, an independent working correlation structure is assumed.


    The assumption of an independent working correlation structure is done for two reasons: computational simplification
    and it does not rely on an extra assumption. Without needing to specify a more detailed structure, the aggregation
    of observations is straightforward for an arbitrary set of estimating functions. This means this procedure is
    flexible with any general input matrix of estimating function contributions. Regarding the second reason, as
    described in Pepe & Anderson (1994) and Pan et al. (2000), use of non-diagonal working correlation matrices (i.e.,
    other options than independent) relies on an additional assumption that may not hold. When this assumption does not
    hold, *point* estimates may be biased. Given that the empirical sandwich variance estimator is consistent under
    misspecification of the working correlation structure, the philosophy of this utility is to maintain unbiased point
    estimation at the potential cost of some statistical efficiency (when the correlation structure is correctly
    specified and the additional assumption holds).


    Parameters
    ----------
    est_funcs : ndarray, list, vector
        Input `p`-by-`n` matrix to collapse into a `p`-by-`m` matrix, where `n` is the number of units and `m` is the
        number of groups.
    group : ndarray, list, vector
        A vector of length `n` designating the group identifiers for each unit-level observation.

    Returns
    -------
    array :
        Returns a `p`-by-`m` NumPy array

    Examples
    --------
    Using the ``aggregate_efuncs`` utility for grouped or clustered data

    >>> import numpy as np
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean
    >>> from delicatessen.utilities import aggregate_efuncs

    Some generic data for clustered-observations

    >>> y = [1, -1, 0, 3, 2, -3, -2, -1, 1, 0]
    >>> group = [1, 1, 1, 2, 2, 3, 4, 4, 5, 6]

    Here, we are interested in estimating the mean of Y. There are 10 observations, but these observations only come
    from 6 unique groups (clusters). Therefore, we aggregate the estimating functions for the mean of Y by the group
    IDs. To apply this, we (1) compute the estimating functions at the unit-level, (2) aggregate the contributions at
    the group level (using ``aggregate_efuncs``), and (3) return the group-level estimating functions to
    ``MEstimator`` (or ``GMMEstimator``). The following blocks of code illustrate this process

    >>> def psi(theta):
    >>>     ee_ind_level = ee_mean(theta=theta, y=y)
    >>>     ee_group_level = aggregate_efuncs(ee_ind_level, group=group)
    >>>     return ee_group_level

    >>> estr = MEstimator(psi, init=[0., ])
    >>> estr.estimate()

    By aggregating estimating functions prior to providing to ``MEstimator``, we change the effective sample size and
    modify the inputs to the empirical sandwich variance estimator. This aggregation can be done for more than 1
    parameter and helps to simplify inference for grouped or clustered data.

    References
    ----------
    Pepe SM & Anderson GL (1994). A cautionary note on inference for marginal regression models with longitudinal data
    and general correlated response data. *Communications in Statistics-Simulation and Computation*, 23, 939-951.

    Pan W, Louis TA, & Connett JE. (2000). A note on marginal linear regression with correlated response data.
    *The American Statistician*, 54(3), 191-195.
    """
    id_vector = np.asarray(group)                           # Converting input into NumPy array
    est_funcs = np.asarray(est_funcs)                       # Converting input into NumPy array

    # Number of observations in the input estimating function and observations
    unique_id, inv = np.unique(group, return_inverse=True)  # Unique IDs present and their corresponding indices
    g = unique_id.size                                      # Number of unique group IDs
    if len(est_funcs.shape) == 1:                           # Checking if single parameter
        n_obs = est_funcs.shape[0]                          # ... because then first index is sample size
        n_prm = 1                                           # ... and only a single parameter
    else:                                                   # Otherwise more than 1 parameter
        n_obs = est_funcs.shape[1]                          # ... and the second index is the sample size
        n_prm = est_funcs.shape[0]                          # ... with the first as the number of parameters

    # Error checking for the input shapes (to give informative error)
    if id_vector.shape[0] != n_obs:
        raise ValueError("The length of the `group` vector must match the number of units in the provided estimating "
                         "functions. Instead, there were "+str(id_vector.shape[0])+" units and there were "
                         +str(n_obs)+" estimating function contributions.")

    # Create indices for bincount operation
    compact_index = inv[None, :] + np.arange(n_prm)[:, None] * g

    # Adding across the id_vector indices in the correct direction
    return np.bincount(compact_index.ravel(),                  # 1D array of the indices for everything
                       weights=est_funcs.ravel(),              # ... with the weights for the counts from the EFs
                       minlength=n_prm * g).reshape(n_prm, g)  # ... then shaping back into a 2D array


def regression_predictions(X, theta, covariance, offset=None, alpha=0.05):
    r"""Generate predicted values of the outcome given a design matrix, point estimates, and covariance matrix.
    This functionality computes :math:`\hat{Y}`, :math:`\hat{Var}\left(\hat{Y}\right)`, and corresponding Wald-type
    :math:`(1 - \alpha) \times` 100% confidence intervals from estimated coefficients and covariance of a regression
    model given a set of specific covariate values.

    This function is a helper function to compute the predictions from a regression model for a set of given :math:`X`
    values. Importantly, this method allows for the variance of :math:`\hat{Y}` to be estimated without having to expand
    the estimating equations. As such, this functionality is meant to be used after ``MEstimator`` has been used to
    estimate the coefficients (i.e., this function is for use after the M-estimator has computed the results for the
    chosen regression model). Therefore, this function should be viewed as a post-processing functionality for
    generating plots or tables.

    Note
    ----
    No tranformations are applied by this function. So, input from a logistic model will generate the *log-odds* of the
    outcome (not probability).


    Parameters
    ----------
    X : ndarray, list, vector
        2-dimensional vector of values to generate predicted variances for. The number of columns must match the number
        of coefficients / parameters in ``theta``.
    theta : ndarray
        Estimated coefficients from ``MEstimator.theta``.
    covariance : ndarray
        Estimated covariance matrix from ``MEstimator.variance``.
    offset : ndarray, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.
    alpha : float, optional
        The :math:`\alpha` level for the corresponding confidence intervals. Default is 0.05, which calculate the
        95% confidence intervals. Notice that :math:`0<\alpha<1`.

    Returns
    -------
    array :
        Returns a n-by-4 NumPy array with the 4 columns correspond to the predicted outcome, variance of the predictied
        outcome, lower confidence limit, and upper confidence limit.

    Examples
    --------
    The following is a simple example demonstrating how this function can be used to plot a regression line and
    corresponding 95% confidence intervals.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_regression
    >>> from delicatessen.utilities import regression_predictions

    Some generic data to estimate the regression model with

    >>> n = 50
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['C'] = 1

    Estimating the corresponding regression model parameters

    >>> def psi(theta):
    >>>     return ee_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'], model='linear')

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate(solver='lm')

    To create a line plot of our regression line, we need to first create a new set of covariate values that are evenly
    spaced across the range of the predictor values. Here, we will plot the relationship between ``Z`` and ``Y`` while
    holding ``X=0``.

    >>> pred = pd.DataFrame()
    >>> pred['Z'] = np.linspace(np.min(data['Z']), np.max(data['Z']), 200)
    >>> pred['X'] = 0
    >>> pred['C'] = 1

    Now the predicted values of the outcome, and confidence intervals to plot

    >>> Xp = pred[['C', 'X', 'Z']]
    >>> yhat = regression_predictions(X=Xp, theta=estr.theta, covariance=estr.variance)

    Finally, the predicted values can be plotted (using matplotlib)

    >>> plt.plot(pred['Z'], yhat[:, 0], '-', color='blue')
    >>> plt.fill_between(pred['Z'], yhat[:, 2], yhat[:, 3], alpha=0.25, color='blue')
    >>> plt.show()

    For predicting with a Poisson or logistic model, one may want to transform the predicted values and confidence
    intervals to another measure. For the logistic model, the predicted log-odds can easily be transformed using
    ``delicatessen.utilities.inverse_logit``. For the Poisson model, predictions can easily be transformed using
    ``numpy.exp``.
    """
    # Check valid alpha value is being provided
    check_alpha_level(alpha=alpha)

    # Process offset term
    if offset is None:                                 # When offset is None
        offset = 0                                     # ... modify by adding a zero (i.e., no mod)
    else:                                              # Otherwise
        offset = np.asarray(offset)[:, None]           # ... ensure that a NumPy array is passed forward

    # Setup inputs for matrix multiplications
    x = np.asarray(X)
    b = np.asarray(theta)
    c = np.asarray(covariance)

    # Predicted Y
    yhat = np.dot(x, b) + offset

    # Predicted Y variance / standard error
    yhat_var = np.sum(np.dot(x, c) * x,
                      axis=1)

    # Confidence limit of predictions
    yhat_se = np.sqrt(yhat_var)                        # Taking square root to get SE
    z_alpha = norm.ppf(1 - alpha/2, loc=0, scale=1)    # Z_alpha value for CI
    lower_ci = yhat - z_alpha*yhat_se                  # Lower CI
    upper_ci = yhat + z_alpha*yhat_se                  # Upper CI

    # Return estimates and variance
    return np.vstack([yhat, yhat_var, lower_ci, upper_ci]).T


def survival_predictions(times, theta, covariance, distribution, measure='survival', alpha=0.05):
    r"""Compute estimated functions for survival analysis measures from a parametric survival analysis model across
    a specified time period. This function is meant to be used with ``ee_survival_model`` and is a simple way to compute
    values of a survival analysis metric (and the corresponding point-wise confidence intervals) at user-specified
    time points. This functionality is meant to help with generating plots or describing results.

    To generate predicted values of the desired measure, the survival and hazard are computed using

    .. math::

        S(t) = \exp(- \lambda t^\gamma ) \\
        h(t) = - \gamma \lambda t^{\gamma - 1}

    From these two values, the specified measure is computed (see ``convert_survival_measures`` for details). The
    variance for the chosen measure is then computed using the Delta Method with automatic differentiation via the
    ``delta_method`` function.

    Parameters
    ----------
    times : float, int, ndarray, list, vector
        Either a single time point or a vector of time points to generate predicted measures at. This argument
        determines the shape of the output.
    theta : ndarray
        Estimated coefficients from ``MEstimator.theta`` with ``ee_survival``.
    covariance : ndarray
        Estimated covariance matrix from ``MEstimator.variance`` with ``ee_survival``.
    distribution : str
        Distribution of the AFT model, which should match the distribution specified in ``ee_survival_model``. See
        ``ee_survival_model`` for available options.
    measure : str, optional
        Measure to compute. Options include survival (``'survival'``), density (``'density'``), risk or the cumulative
        density (``'risk'``), hazard (``'hazard'``), or cumulative hazard (``'cumulative_hazard'``). Default is
        survival
    alpha : float, optional
        The :math:`\alpha` level for the corresponding confidence intervals. Default is 0.05, which calculate the
        95% confidence intervals. Notice that :math:`0 < \alpha < 1`.

    Returns
    -------
    array :
        Returns a `t`-by-`4` NumPy array of predictions, where the first column is the survival metric, the second is
        the corresponding variance, and the last two columns are the lower confidence limit and upper confidence limit,
        respectively.

    Examples
    --------
    The following illustrates how to use ``survival_predictions`` to generate a plot of the risk function. Other
    metrics can be plotted using a similar approach.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_survival_model
    >>> from delicatessen.utilities import survival_predictions

    Some generic data to estimate the regression model with

    >>> n = 100
    >>> d = pd.DataFrame()
    >>> d['C'] = np.random.weibull(a=1, size=n)
    >>> d['C'] = np.where(d['C'] > 5, 5, d['C'])
    >>> d['T'] = 0.8*np.random.weibull(a=0.8, size=n)
    >>> d['delta'] = np.where(d['T'] < d['C'], 1, 0)
    >>> d['t'] = np.where(d['delta'] == 1, d['T'], d['C'])

    Now we will use ``ee_survival_model`` to estimate the parameters of a Weibull model

    >>> def psi(theta):
    >>>     return ee_survival_model(theta=theta, t=d['t'], delta=d['delta'],
    >>>                              distribution='weibull')

    >>> estr = MEstimator(psi, init=[1., 1.])
    >>> estr.estimate()

    Now we can generate predicted values of the risk at specified times. Suppose we wanted the risk at a time of 5 and
    the corresponding confidence intervals. The following code gives us the risk at this time, the variance, and
    confidence intervals in an array

    >>> survival_predictions(times=5., theta=estr.theta, covariance=estr.variance,
    >>>                      distribution='weibull', measure='risk')

    Now, we will use these predictions to plot the risk function over the time period. We generate a vector of times
    for the plot. These should be chosen 'densely', so the plot appears smooth

    >>> # Generating predictions
    >>> times = np.linspace(0.01, 5, 100)
    >>> s_hat = survival_predictions(times=times, theta=estr.theta, covariance=estr.variance,
    >>>                              distribution='weibull', measure='risk')
    >>> # Plot
    >>> plt.fill_between(times, s_hat[:, 2], s_hat[:, 3], color='blue', alpha=0.3)
    >>> plt.plot(times, s_hat[:, 0], '-', color='blue', alpha=0.3)
    >>> plt.xlabel("Time")
    >>> plt.ylabel("Risk")
    >>> plt.show()

    Here, the ``fill_between` displays the point-wise 95% confidence intervals.
    """
    def predict_metric(times, theta, distribution):
        # Function to handle prediction process
        if isinstance(times, (numbers.Number, np.number)):
            # Preparing inputs
            if distribution == 'exponential':
                lambd = theta[0]
                gamma = 1
            else:
                lambd = theta[0]
                gamma = theta[1]
            # Computing predicted survival
            survival_t = np.exp(-lambd * (times ** gamma))     # Survival calculation from parameters
            hazard_t = lambd * gamma * (times ** (gamma - 1))  # hazard calculation from parameters
            metric = convert_survival_measures(survival=survival_t, hazard=hazard_t, measure=measure)
            return metric
        else:
            metrics = []
            for t in times:
                metric = predict_metric(times=t, theta=theta, distribution=distribution)
                metrics.append(metric)
            return np.asarray(metrics)

    def predict_function_differentiable(theta):
        # Function that is callable with the differentiation methods
        return predict_metric(times=times, theta=theta, distribution=distribution)

    # Check valid alpha value is being provided
    check_alpha_level(alpha=alpha)

    # Predicted measure at given times
    est = predict_function_differentiable(theta=theta)

    # Covariance for measure at given times
    covariance_m = delta_method(theta=theta, g=predict_function_differentiable, covariance=covariance)
    variance_m = np.diag(covariance_m)

    # Confidence limit of predictions
    yhat_se = np.sqrt(variance_m)                      # Taking square root to get SE
    z_alpha = norm.ppf(1 - alpha/2, loc=0, scale=1)    # Z_alpha value for CI
    lower_ci = est - z_alpha*yhat_se                   # Lower CI
    upper_ci = est + z_alpha*yhat_se                   # Upper CI

    # Return estimates and variance
    return np.vstack([est, variance_m, lower_ci, upper_ci]).T


def aft_predictions_individual(X, times, theta, distribution, measure='survival'):
    r"""Compute predicted survival analysis measures from an accelerated failure time (AFT) model for given a design
    matrix and times. This function is meant to be used with parametrization of the ``ee_aft`` to generate predicted
    survival (or other measures) at user-specified time points.

    Predictions are generated via

    .. math::

        S(t) = S_{\epsilon}\left( \frac{\log(t) - X \beta^T}{\sigma} \right) \\
        h(t) = (\sigma t)^{-1} h_{\epsilon}\left( \frac{\log(t) - X \beta^T}{\sigma} \right)

    where the corresponding function for the given AFT distribution is

    .. list-table::
       :widths: 25 25 25 25
       :header-rows: 1

       * - Distribution
         - Keyword
         - :math:`S_\epsilon(x)`
         - :math:`h_\epsilon(x)`
       * - Exponential
         - ``exponential``
         - :math:`\exp(-\exp(x))`
         - :math:`\exp(x)`
       * - Weibull
         - ``weibull``
         - :math:`\exp(-\exp(x))`
         - :math:`\exp(x)`
       * - Log-Logistic
         - ``log-logistic``
         - :math:`(1 - \exp(x))^{-1}`
         - :math:`(1 - \exp(-x))^{-1}`
       * - Log-Normal
         - ``log-normal``
         - :math:`1 - \Phi(x)`
         - :math:`\frac{\exp(-x^2 / 2)}{[1 - \Phi(x)] \sqrt{2 \pi }}`

    Note that one only needs to ensure that ``distribution`` is set to the same argument as the one used in ``ee_aft``

    Parameters
    ----------
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
    times : float, int, ndarray, list, vector
        Either a single time point or a vector of time points to generate predicted measures at. This argument
        determines the shape of the output.
    theta : ndarray, list, vector
        Estimated coefficients from ``MEstimator.theta`` with ``ee_aft``.
    distribution : str
        Distribution to use for the AFT model. See table for options.
    measure : str, optional
        Measure to compute. Options include survival (``'survival'``), density (``'density'``), risk or the cumulative
        density (``'risk'``), hazard (``'hazard'``), or cumulative hazard (``'cumulative_hazard'``). Default is
        survival

    Returns
    -------
    array :
        Returns a `n`-by-`t` NumPy array of predictions, where `n` is the number of rows in the design matrix and `t`
        is the number of time points.

    Examples
    --------
    The following illustrates how to use ``aft_predictions_individual`` to generate predicted survival probabilites at
    specific times for individuals.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_aft
    >>> from delicatessen.utilities import aft_predictions_individual
    >>> from delicatessen.data import load_breast_cancer

    Loading breast cancer data from Collett 2015

    >>> dat = load_breast_cancer()
    >>> delta = dat[:, 0]
    >>> t = dat[:, 1]
    >>> covars = np.asarray([np.ones(dat.shape[0]), dat[:, 0]]).T

    Estimating the parameters of a Weibull AFT model

    >>> def psi(theta):
    >>>     return ee_aft(theta=theta, t=t, delta=delta,
    >>>                   X=covars, distribution='weibull')

    >>> estr = MEstimator(psi, init=[5., 0., 0.])
    >>> estr.estimate()

    Now we can generate predicted values of survival for each observation. Suppose we wanted the survival at time 50
    for all units. The following code gives us predicted survival for all units

    >>> aft_predictions_individual(X=covars, times=50.,
    >>>                            theta=estr.theta,
    >>>                            distribution='weibull')

    Alternatively, we can request the predicted survival at multiple points at once. The following code computes the
    predicted survival at times 50, 100, 150, 200, 250 for all units.

    >>> aft_predictions_individual(X=covars, times=[50, 100, 150, 200, 250],
    >>>                            theta=estr.theta,
    >>>                            distribution='weibull')

    Different survival measures can be requested through the optional ``measure`` argument.

    References
    ----------
    Collett D. (2015). Accelerated failure time and other parametric models. In: Modelling survival data in medical
    research. CRC press. pg 242
    """
    if isinstance(times, (numbers.Number, np.number)):
        # Preparing inputs
        X = np.asarray(X)
        theta = np.asarray(theta)

        # Extracting parameters from input vector
        beta_dim = X.shape[1]
        beta = np.asarray(theta[:beta_dim])[:, None]
        if distribution == 'exponential':
            sigma = 1
        else:
            sigma = np.exp(-theta[-1])

        # Predicted survival at times, comes from Table 6.2 on page 242 of Collett 2015
        t_hat = np.dot(X, beta)
        log_t = np.log(times)
        epsilon_i = (log_t - t_hat) / sigma
        hazard_scaler = 1 / (sigma * times)
        if distribution in ['exponential', 'weibull']:
            survival_t = np.exp(-np.exp(epsilon_i))
            hazard_t = hazard_scaler * np.exp(epsilon_i)
        elif distribution in ['lognormal', 'log-normal']:
            survival_t = 1 - standard_normal_cdf(epsilon_i)
            hazard_t = hazard_scaler * np.exp(-epsilon_i**2 / 2) / (survival_t * np.sqrt(2 * np.pi))
        elif distribution in ['loglogistic', 'log-logistic']:
            survival_t = 1 / (1 + np.exp(epsilon_i))
            hazard_t = hazard_scaler * 1 / (1 + np.exp(-epsilon_i))
        else:
            raise ValueError("The specified distribution `" + str(distribution)
                             + "` is not supported")
        survival_t = survival_t.T[0]
        hazard_t = hazard_t.T[0]

        # Converting survival and hazard into desired metric
        metric = convert_survival_measures(survival=survival_t, hazard=hazard_t, measure=measure)
        return metric

    else:
        predictions = []
        for t in times:
            pred_t = aft_predictions_individual(X=X, times=t, theta=theta,
                                                distribution=distribution, measure=measure)
            predictions.append(pred_t)

        return np.asarray(predictions).T


def aft_predictions_function(X, times, theta, covariance, distribution, measure='survival', alpha=0.05):
    r"""Compute estimated functions for survival analysis measures from an accelerated failure time (AFT) model across
    a specified time period. This function is meant to be used with ``ee_aft`` and is a simple way to compute values of
    a survival analysis metric (and the corresponding point-wise confidence intervals) at user-specified
    time points for a given pattern of covariates. This functionality is meant to help with generating plots or
    describing results.

    To generate predicted values of the desired measure, the survival and hazard are computed using

    .. math::

        S(t) = S_{\epsilon}\left( \frac{\log(t) - X \beta^T}{\sigma} \right) \\
        h(t) = (\sigma t)^{-1} h_{\epsilon}\left( \frac{\log(t) - X \beta^T}{\sigma} \right)

    where the corresponding function for the given AFT distribution is

    .. list-table::
       :widths: 25 25 25 25
       :header-rows: 1

       * - Distribution
         - Keyword
         - :math:`S_\epsilon(x)`
         - :math:`h_\epsilon(x)`
       * - Exponential
         - ``exponential``
         - :math:`\exp(-\exp(x))`
         - :math:`\exp(x)`
       * - Weibull
         - ``weibull``
         - :math:`\exp(-\exp(x))`
         - :math:`\exp(x)`
       * - Log-Logistic
         - ``log-logistic``
         - :math:`(1 - \exp(x))^{-1}`
         - :math:`(1 - \exp(-x))^{-1}`
       * - Log-Normal
         - ``log-normal``
         - :math:`1 - \Phi(x)`
         - :math:`\frac{\exp(-x^2 / 2)}{[1 - \Phi(x)] \sqrt{2 \pi }}`

    Note that one only needs to ensure that ``distribution`` is set to the same argument as the one used in ``ee_aft``

    From these values, the specified measure is computed (see ``convert_survival_measures`` for details). The
    variance for the chosen measure is then computed using the Delta Method with automatic differentiation via the
    ``delta_method`` function. Corresponding :math:`1-\alpha`% Wald-type point-wise confidence intervals are then
    computed using this variance

    Parameters
    ----------
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
    times : float, int, ndarray, list, vector
        Either a single time point or a vector of time points to generate predicted measures at. This argument
        determines the shape of the output.
    theta : ndarray, list, vector
        Estimated coefficients from ``MEstimator.theta`` with ``ee_aft``.
    covariance : ndarray, list, vector
        Estimated covariance matrix from ``MEstimator.variance`` with ``ee_aft``.
    distribution : str
        Distribution to use for the AFT model. See table for options.
    measure : str, optional
        Measure to compute. Options include survival (``'survival'``), density (``'density'``), risk or the cumulative
        density (``'risk'``), hazard (``'hazard'``), or cumulative hazard (``'cumulative_hazard'``). Default is
        survival
    alpha : float, optional
        The :math:`\alpha` level for the corresponding confidence intervals. Default is 0.05, which calculate the
        95% confidence intervals. Notice that :math:`0 < \alpha < 1`.

    Returns
    -------
    array :
        Returns a `t`-by-`4` NumPy array of predictions, where the first column is the survival metric, the second is
        the corresponding variance, and the last two columns are the lower confidence limit and upper confidence limit,
        respectively.

    Examples
    --------
    The following illustrates how to use ``aft_predictions_function`` to generate a plot of the risk function. Other
    metrics can be plotted using a similar approach.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_aft
    >>> from delicatessen.utilities import aft_predictions_function
    >>> from delicatessen.data import load_breast_cancer

    Loading breast cancer data from Collett 2015

    >>> dat = load_breast_cancer()
    >>> delta = dat[:, 0]
    >>> t = dat[:, 1]
    >>> covars = np.asarray([np.ones(dat.shape[0]), dat[:, 0]]).T

    Estimating the parameters of a Weibull AFT model

    >>> def psi(theta):
    >>>     return ee_aft(theta=theta, t=t, delta=delta,
    >>>                   X=covars, distribution='weibull')

    >>> estr = MEstimator(psi, init=[5., 0., 0.])
    >>> estr.estimate()

    Now we can generate predicted values of the risk at specified times for a specific covariate pattern. Suppose we
    wanted the risk at a time of 50 for those with a negative stain and the corresponding confidence intervals. The
    following code gives us the risk, variance, and confidence intervals

    >>> aft_predictions_function(times=50., theta=estr.theta, covariance=estr.variance,
    >>>                          X=[[1, 0]],  # Intercept-only
    >>>                          distribution='weibull', measure='risk')

    We can do the same process for those with a positive stain by switching out the design matrix

    >>> aft_predictions_function(times=50., theta=estr.theta, covariance=estr.variance,
    >>>                          X=[[1, 1]],  # Intercept and positive stain
    >>>                          distribution='weibull', measure='risk')

    Now, we will use these predictions to plot the risk function over the time period. We generate a vector of times
    for the plot. These should be chosen 'densely', so the plot appears smooth

    >>> # Time steps to generate predictions for
    >>> times = np.linspace(0.01, 230, 100)
    >>> # Generating predictions
    >>> s0_hat = survival_predictions(times=times, theta=estr.theta, covariance=estr.variance,
    >>>                               X=[[1, 0]],  # Intercept only
    >>>                               distribution='weibull', measure='risk')
    >>> s1_hat = survival_predictions(times=times, theta=estr.theta, covariance=estr.variance,
    >>>                               X=[[1, 1]],  # Intercept and positive stain
    >>>                               distribution='weibull', measure='risk')
    >>> # Plot
    >>> plt.fill_between(times, s1_hat[:, 2], s1_hat[:, 3], color='blue', alpha=0.3)
    >>> plt.fill_between(times, s0_hat[:, 2], s0_hat[:, 3], color='red', alpha=0.3)
    >>> plt.plot(times, s1_hat[:, 0], '-', color='blue', alpha=0.3, label='Pos')
    >>> plt.plot(times, s0_hat[:, 0], '-', color='red', alpha=0.3, label='Neg')
    >>> plt.xlabel("Time")
    >>> plt.ylabel("Risk")
    >>> plt.legend()
    >>> plt.show()

    Here, the ``fill_between`` displays the point-wise 95% confidence intervals. Other survival measures can be
    requested through the optional ``measure`` argument.

    References
    ----------
    Collett D. (2015). Accelerated failure time and other parametric models. In: Modelling survival data in medical
    research. CRC press. pg 242
    """
    def predictions_aft(theta):
        # Wrapped prediction function for call with delta_method
        preds = aft_predictions_individual(X=X, times=times, theta=theta, distribution=distribution, measure=measure)
        return preds[0]

    # Check valid alpha value is being provided
    check_alpha_level(alpha=alpha)

    # Check X has only 1 row (error otherwise, since difficult to keep track of multiple covariate patterns and var)
    X = np.asarray(X)
    if X.shape[0] > 1:
        raise ValueError("Only one covariate pattern at a time can be specified, so `X` should only consist of a "
                         "single row. However, " + str(X.shape[0]) + " rows were provided.")

    # Predicted measure at given times
    est = predictions_aft(theta=theta)

    # Covariance for measure at given times
    covariance_m = delta_method(theta=theta, g=predictions_aft, covariance=covariance)
    variance_m = np.diag(covariance_m)

    # Confidence limit of predictions
    yhat_se = np.sqrt(variance_m)                      # Taking square root to get SE
    z_alpha = norm.ppf(1 - alpha/2, loc=0, scale=1)    # Z_alpha value for CI
    lower_ci = est - z_alpha*yhat_se                   # Lower CI
    upper_ci = est + z_alpha*yhat_se                   # Upper CI

    # Return estimates and variance
    return np.vstack([est, variance_m, lower_ci, upper_ci]).T


def plogit_predict(theta, t, delta, X, S=None, times_to_predict=None, measure='survival', unique_times=None):
    r"""Compute predicted survival analysis measures from a pooled logistic regression model for given a design matrix
    and times. This function is meant to be used with ``ee_pooled_logistic`` to generate predicted survival (or other
    measures) at designated time points.

    Given a specified covariate and time design matrix, the coefficients from a pooled logistic regression model are
    used to generate conditional probabilities of the event. These are then transformed into the desired survival
    measure. Predictions can be output for a selected set of times (``times_to_predict``).

    Note
    ----
    Specifications of ```theta``, ``t``, ``delta``, ``S``, and ``unique_times`` should match those provided to
    ``ee_pooled_logistic``.


    Parameters
    ----------
    theta : ndarray, list, vector
        Estimated parameter vector for the pooled logistic model. Composed of the parameters for the baseline
        covariates and the time coefficients. These should be the values optimized by ``ee_pooled_logistic``.
    t : ndarray, list, vector
        1-dimensional vector of `n` observed times. This should be the same values provided to ``ee_pooled_logistic``.
    delta : ndarray, list, vector
        1-dimensional vector of `n` event indicators, where 1 indicates an event and 0 indicates right censoring.
         This should be the same values provided to ``ee_pooled_logistic``.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables. Covariate values can be modified from those given
        to ``ee_pooled_logistic``, as is done with g-computation estimators.
    S : ndarray, list, vector, None, optional
        Optional argument for parametric function form specifications for time. Default is ``None``, which uses disjoint
        indicators to model time. Expected to have ``np.max(t)`` rows. This should match the specification provided
        to ``ee_pooled_logistic``.
    times_to_predict : int, float, ndarray, list, vector, None, optional
        Time(s) to generate predicted values for. Specified times must be :math:`[0, \tau]`. Default is ``None``, which
        generates predicted values at each unique event time (if ``S=None``) or at each unit-time interval (``S!=None``)
    measure : str, optional
        Measure to compute. Options include survival (``'survival'``), density (``'density'``), risk or the cumulative
        density (``'risk'``), hazard (``'hazard'``), or cumulative hazard (``'cumulative_hazard'``). Default is
        survival
    unique_times : None, ndarray, list, vector, optional
        Optional argument to compute the disjoint indicators for only a subset of terms. This argument is intended for
        use with disjoint indicators for time that are stratified by some external variable. This argument is ignored
        when ``S`` is not ``None``. This should match the specification provided to ``ee_pooled_logistic``.

    Returns
    -------
    array :
        Returns a `n`-by-`K` NumPy array of predictions, where `n` is the number of rows in the design matrix and `K`
        is the number of time points to compute the survival measure at.

    Examples
    --------

    The following illustrates how to use ``plogit_predictions_individual`` to generate predicted survival probabilites
    at specific times for individuals.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_plogit
    >>> from delicatessen.utilities import plogit_predict
    >>> from delicatessen.data import load_breast_cancer

    Here, we will illustrate pooled logistic regression with breast cancer from the Middlesex Hospital in July 1987.
    This data can be loaded as follows

    >>> d = pd.DataFrame(load_breast_cancer(), columns=['d', 't', 'statin'])

    To start, we estimate the coefficients of a pooled logistic regression where time is modeled disjoint indicators.
    See ``ee_plogit`` for further details

    >>> unique_event_times = list(np.unique(d.loc[d['d'] == 1, 't']))

    >>> def psi(theta):
    >>>     return ee_plogit(theta=theta, X=d[['statin', ]], delta=d['d'], t=d['t'])

    >>> inits = [0., ] + [-3., ] + [0., ]*(len(unique_times) - 1)
    >>> estr = MEstimator(stacked_equations=psi, init=inits)
    >>> estr.estimate()

    After estimating the parameters, predicted survival metrics can be computed. Here, we compute the risk function
    for all observations at all unique events times.

    >>> plogit_predict(theta=estr.theta, t=d['t'], delta=d['delta'], X=d[['statin', ]], S=None, measure='risk')

    Note that the shared arguments between ``ee_plogit`` and ``plogit_predict`` (besides ``X``, which can be modified)
    should match each other. If they do not, unexpected behaviors may occur.

    For further details on how to use ``plogit_predict``, see the Applied Examples.

    References
    ----------
    Abbott RD. (1985). Logistic regression in survival analysis. *American Journal of Epidemiology*, 121(3), 465-471.

    D'Agostino RB, Lee ML, Belanger AJ, Cupples LA, Anderson K, & Kannel WB. (1990). Relation of pooled logistic
    regression to time dependent Cox regression analysis: the Framingham Heart Study.
    *Statistics in Medicine*, 9(12), 1501-1515.

    Zivich PN, Cole SR, Shook-Sa BE, DeMonte JB, & Edwards JK. (2025). Estimating equations for survival analysis with
    pooled logistic regression. *arXiv:2504.13291*
    """
    # Pre-processing input data
    t = np.asarray(t)                 # Convert to NumPy array
    delta = np.asarray(delta)         # Convert to NumPy array
    X = np.asarray(X)                 # Convert to NumPy array
    xp = X.shape[1]                   # Get shape of X array to divide parameter vector
    beta_x = theta[:xp]               # Beta parameters for X design matrix
    beta_s = np.asarray(theta[xp:])   # Beta parameters for S design matrix

    if S is None:
        if unique_times is None:
            event_times = t[delta == 1]
            unique_times = np.unique(event_times)
        else:
            unique_times = np.asarray(unique_times)
        n_time_steps = unique_times.shape[0]

        # Creating design matrix for time
        time_design_matrix = np.identity(n=len(unique_times))
        time_design_matrix[:, 0] = 1
    else:
        time_design_matrix = np.asarray(S)
        unique_times = np.asarray(range(1, int(np.max(t))+1, 1))
        n_time_steps = len(unique_times)       #

    # Log-odds contributions for covariate and time
    log_odds_w = np.dot(X, beta_x)
    log_odds_t = np.dot(time_design_matrix, beta_s)

    # Computing full matrix of predicted values for each time
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))       # Stacked copies of X contributions for intervals
    y_pred = inverse_logit(log_odds_w_matrix + log_odds_t[:, None])  # Predicted event at time intervals matrix
    survival_prediction = np.cumprod(1 - y_pred, axis=0)
    prediction_matrix = convert_survival_measures(survival_prediction, hazard=y_pred, measure=measure)
    prediction_t0 = convert_survival_measures(1, 0, measure=measure)

    # Computing requested predictions
    if times_to_predict is None:
        return prediction_matrix
    else:
        predictions = []
        for time in times_to_predict:
            if time == 0 or time < unique_times[0]:
                prediction = np.ones(t.shape[0]) * prediction_t0
            elif time > np.max(t):
                raise ValueError("Cannot predict beyond the maximum observed time")
            else:
                if unique_times[-1] <= time:
                    pred_matrix_index = -1
                else:
                    further_times = unique_times[time < unique_times]
                    if len(further_times) < 1:
                        nearest = unique_times[time >= unique_times][-1]  # Looks at jump point before
                    else:
                        nearest = unique_times[time < unique_times][0]  # Looks at jump point after
                    pred_matrix_index = np.where(unique_times == nearest)[0][0] - 1
                prediction = prediction_matrix[pred_matrix_index, :]
            predictions.append(prediction)
        return np.asarray(predictions)


def spline(variable, knots, power=3, restricted=True, normalized=False):
    r"""Generate generic polynomial spline terms for a given NumPy array and pre-specified knots. Default is restricted
    cubic splines but unrestricted splines to different polynomial terms can also be specified.

    Unrestricted splines for knot :math:`k` are generated using the following formula

    .. math::

        s_k(X) = I(X > k) \left\{ X - k \right\}^a

    where :math:`a` is the power (:math:`a=3` for cubic splines).

    Restricted splines are generated via

    .. math::

        r_k(X) = I(X > k) \left\{ X - k \right\}^a - s_K(X)

    where :math:`K` is largest knot value.

    Splines are normalized by the upper knot minus the lower knot to the corresponding power. Normalizing the splines
    can be helpful for the root-finding procedure.

    Parameters
    ----------
    variable : ndarray, vector, list
        1-dimensional vector of observed values. Input should consists of the variable to generate spline terms for
    knots : ndarray, vector, list
        1-dimensional vector of pre-specified knot locations. All knots should be between the minimum and maximum
        values of the input variable
    power : int, float, optional
        Power or polynomial term to use for the splines. Default is 3, which corresponds to cubic splines
    restricted : bool, optional
        Whether to generate restricted or unrestricted splines. Default is True, which corresponds to restricted
        splines. Restricted splines return one less column than the number of knots, whereas unrestricted splines
        return the same number of columns as knots
    normalized : bool, optional
        Whether to normalize, or divide, the spline terms by the difference between the upper and lower knots. Default
        is ``False``, which corresponds to unnormalized splines.

    Returns
    -------
    ndarray :
        A 2-dimensional array of the spline terms in ascending order of the knots.

    Examples
    --------
    Construction of spline variables should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen.utilities import spline

    Some generic data to estimate a generalized additive model

    >>> x = np.random.normal(size=200)

    A restricted quadratic spline with 3 knots (at -1, 0, 1) can be generated using the following function call

    >>> spline(variable=x, knots=[-1, 0, 1], power=2, restricted=True)

    This function will return a 2 by 200 array here. Other knot specifications, other powers, and unrestricted splines
    can also be generated by updating the corresponding arguments.

    References
    ----------
    Mulla ZD (2007). Spline regression in clinical research. *West Indian Med J*, 56(1), 77.
    """
    # Processing input and output arrays
    knots = sorted(knots)                           # Sorting the order of the knots
    n_cols = len(knots)                             # Column number is based on the number of provided knots
    x = np.asarray(variable)                        # Converting input variable to another array
    spline_terms = np.empty((x.shape[0], n_cols))   # Creating the output spline array

    # Determining normalization
    if normalized:                                     # Rescale spline terms so less extreme ranges (helps root-finder)
        if n_cols == 1:                                # When only a single knot
            divisor = knots[0]**power                  # ... scale by knot to that power
        else:                                          # Otherwise
            divisor = (knots[-1] - knots[0])**power    # ... range of the knots to that power
    else:                                              # If not rescaling (only option up to v2.0)
        divisor = 1                                    # ... divisor is 1

    # Generating each spline with it's corresponding term
    for i in range(n_cols):
        spline_terms[:, i] = np.where(x > knots[i], (x - knots[i])**power, 0)
        spline_terms[:, i] = np.where(np.isnan(x), np.nan, spline_terms[:, i])

    # Logic for unrestricted and restricted splines
    if restricted is False:
        return spline_terms / divisor
    else:
        for i in range(n_cols - 1):
            spline_terms[:, i] = np.where(x > knots[i], spline_terms[:, i] - spline_terms[:, -1], 0)
            spline_terms[:, i] = np.where(np.isnan(x), np.nan, spline_terms[:, i])
        return spline_terms[:, :-1] / divisor


def additive_design_matrix(X, specifications, return_penalty=False):
    r"""Generate an additive design matrix for generalized additive models (GAM) given a set of spline specifications to
    apply.

    Note
    ----
    This function is interally called by ``ee_additive_regression``. This function can also be called to aid in easily
    generating predicted values.

    Parameters
    ----------
    X : ndarray, vector, list
        Input independent variable data.
    specifications : ndarray, vector, list
        A list of dictionaries that define the hyperparameters for the spline (e.g., number of knots, strength of
        penalty). For terms that should not have splines, ``None`` should be specified instead (see examples below).
        Each dictionary supports the following parameters:
        "knots", "natural", "power", "penalty"
        knots (list): controls the position of the knots, with knots are placed at given locations. There is no
        default, so must be specified by the user.
        natural (bool): controls whether to generate natural (restricted) or unrestricted splines.
        Default is ``True``, which corresponds to natural splines.
        power (float): controls the power to raise the spline terms to. Default is 3, which corresponds to cubic
        splines.
        penalty (float): penalty term (:math:`\lambda`) applied to each corresponding spline basis term. Default is 0,
        which applies no penalty to the spline basis terms.
        normalized (bool): whether to normalize the spline terms. Default is ``False``, with a default change coming
        with v3.0 release.
    return_penalty : bool, optional
        Whether the list of the corresponding penalty terms should also be returned. This functionality is used
        internally to create the list of penalty terms to provide the Ridge regression model, where only the spline
        terms are penalized. Default is False.

    Returns
    -------
    array :
        Returns a (b+k)-by-n design matrix as a NumPy array, where b is the number of columns in the input array and k
        is determined by the specifications of the spline basis functions.

    Examples
    --------
    Construction of a design matrix for an additive model should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen.utilities import additive_design_matrix

    Some generic data to estimate a generalized additive model

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['X'] = np.random.normal(size=n)
    >>> d['Z'] = np.random.normal(size=n)
    >>> d['W'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> d['C'] = 1

    To begin, consider the simple input design matrix of ``d[['C', 'X']]``. This initial design matrix consists of an
    intercept term and a continuous term. Here, we will specify a natural spline with 20 knots for the second term only

    >>> x_knots = np.linspace(np.min(d['X'])+0.1, np.max(d['X'])-0.1, 20)
    >>> specs = [None, {"knots": x_knots, "penalty": 10}]
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X']], specifications=specs)

    Other optional specifications are also available. Here, we will specify an unrestricted quadratic spline with a
    penalty of 5.5 for the second column of the design matrix.

    >>> specs = [None, {"knots": [-2, -1, 0, 1, 2], "natural": False, "power": 2, "penalty": 5.5}]
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X']], specifications=specs)

    Now consider the input design matrix of ``d[['C', 'X', 'Z', 'W']]``. This initial design matrix consists of an
    intercept, two continuous, and a categorical term. Here, we will specify splines for both continuous terms

    >>> x_knots = np.linspace(np.min(d['X'])+0.1, np.max(d['X'])-0.1, 20)
    >>> z_knots = np.linspace(np.min(d['Z'])+0.1, np.max(d['Z'])-0.1, 10)
    >>> specs = [None,                              # Intercept term
    >>>          {"knots": x_knots, "penalty": 25}, # X (continuous)
    >>>          {"knots": z_knots, "penalty": 15}, # Z (continuous)
    >>>          None]                              # W (categorical)
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X', 'Z', 'W']], specifications=specs)

    Notice that the two continuous terms have different spline specifications.

    Finally, we could opt to only generate a spline basis for one of the continuous variables

    >>> specs = [None,                              # Intercept term
    >>>          {"knots": x_knots, "penalty": 25}, # X (continuous)
    >>>          None,                              # Z (continuous)
    >>>          None]                              # W (categorical)
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X', 'Z', 'W']], specifications=specs)

    Specification of splines can be modified and paired in a variety of ways. These are determined by the object type
    in the specification list, and the input dictionary for the spline terms.
    """
    def generate_spline(variable, specification):
        """Internal function to call the spline functionality. This function merely calls the spline function with the
        corresponding specifications. This was built as an internal function to simply future maintenance.

        Parameters
        ----------
        variable : ndarray
            Column of variables to generate the splines for
        specification : dict
            Dictionary of the processed spline specification for the corresponding variable.

        Returns
        -------
        ndarray :
            Returns the object returned by the spline (the basis matrix of the spline terms for the column)
        """
        return spline(variable=variable,                        # Pass variable to spline function
                      knots=specification["knots"],             # ... with knot locations
                      power=specification["power"],             # ... to the power
                      restricted=specification["natural"],      # ... whether to restrict
                      normalized=specification["normalized"])   # ... and whether to normalize by knot range

    def generate_default_spline_parameters(specification):
        """Internal function to process the input specification dictionary of spline parameters. Namely, ensure that
        'knots' is specified, fill in any other empty parameters, and check for additional keys given.

        Parameters
        ----------
        specification : dict
            Dictionary of the input spline specifications to check and process.

        Returns
        -------
        dict :
            Processed dictionary of the spline hyperparameters
        """
        # Setup for dict processing
        keys = specification.keys()                                   # Extract keys from the input dictionary
        defaults = {"knots": None,                                    # Default values (knots must be provided)
                    "natural": True,                                  # ... default to restricted splines
                    "power": 3,                                       # ... default to cubic splines
                    "penalty": 0,                                     # ... default to NO penalty on spline terms
                    "normalized": False}                              # ... default to NO normalized splines

        # Checking the keys in the input dictionary against the expected keys
        expected_keys = ["knots", "natural", "power", "penalty", "normalized"]  # List of keys expected to occur
        extra_keys = [param for param in keys if param not in expected_keys]
        if len(extra_keys) != 0:
            warnings.warn("The following keys were found in the specification: " + str(extra_keys)
                          + ". These keys are not supported and are being ignored.",
                          UserWarning)

        # Managing knot keyword
        if "knots" not in keys:
            raise ValueError("`knots` must be specified.")
        # Managing spline keywords with defaults
        for kw in ["natural", "power", "penalty", "normalized"]:
            if kw not in keys:
                specification[kw] = defaults[kw]

        # Returning processed spline parameters dictionary
        return specification

    # Extract meta-data from input
    X = np.asarray(X)                               # Convert to NumPy array (when user interacts directly with func)
    n_cols = X.shape[1]                             # Number of columns in the input data
    n_obs = X.shape[0]                              # Number of observations in the input data
    if isinstance(specifications, dict):            # Expanding to list if only a single dict is provided
        specifications = [specifications, ]*n_cols
    elif specifications is None:                    # When None is given as specification, ignore splines (linear reg)
        specifications = [specifications, ]*n_cols
    else:                                           # Otherwise check dimensions of specifications and cols match
        if len(specifications) != n_cols:
            raise ValueError("The number of input specifications (" + str(len(specifications)) +
                             ") and the number of columns (" + str(n_cols) + ") do not match")

    # Generate spline terms for each column
    Xt = []                                              # Placeholder storage for generated spline terms
    penalties = []                                       # Placeholder storage for corresponding penalty terms
    for col_id in range(n_cols):                         # Loop through all the columns by their index number
        xvar = X[:, col_id]                              # ... extract corresponding column by ID
        xspec = specifications[col_id]                   # ... extract specification by ID

        # Linear term
        Xt.append(xvar.reshape(n_obs, 1))                # ... always add the corresponding linear term
        penalties.append(0)                              # ... always have penalty of zero for the linear term

        # Spline generation
        if xspec is not None:                            # ... when given a specification != None, generate splines
            # Processing input spline specifications to adhere to conventions
            spec_i = generate_default_spline_parameters(specification=xspec)
            # Generate the spline matrix using the spline function and specification
            spline_matrix = generate_spline(variable=X[:, col_id],
                                            specification=spec_i)
            # Add spline terms to storage
            Xt.append(spline_matrix)                                      # ... add matrix of splines to list
            penalties += [spec_i["penalty"], ]*spline_matrix.shape[1]     # ... add penalty terms from specifications

    # Return the transformed design matrix
    if return_penalty:                      # If return penalty
        return np.hstack(Xt), penalties     # ... return additive design matrix AND list of penalty terms
    else:                                   # Otherwise
        return np.hstack(Xt)                # ... only return the additive design matrix
