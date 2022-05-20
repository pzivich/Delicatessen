import numpy as np

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