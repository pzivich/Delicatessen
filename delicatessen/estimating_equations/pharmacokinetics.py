#####################################################################################################################
# Estimating functions for dose-response analyses
#####################################################################################################################

import warnings
import numpy as np


def ee_4p_logistic(theta, X, y):
    r"""Estimating equations for the 4-parameter logistic model (4PL).

    Note
    ----
    This estimating equation is being depreciated in favor of ``ee_loglogistic`` in v3.0.


    The estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            -2 (Y_i - \hat{Y}_i) (1 - 1/(1 + \rho)) \\
            2 (Y_i - \hat{Y}_i) (\theta_3 - \theta_0) \frac{\theta_2}{\theta_1} \frac{\rho}{(1 + \rho)^2} \\
            2 (Y_i - \hat{Y}_i) (\theta_3 - \theta_0) \log(D_i / \theta_1) \frac{\rho}{(1 + \rho)^2} \\
            2 (Y_i - \hat{Y}_i) (1 / (1 + \rho)))
        \end{bmatrix}
        = 0

    where :math:`R_i` is the response of individual :math:`i`, :math:`D_i` is the dose,
    :math:`\rho = \frac{D_i}{\theta_1}^{\theta_2}`, and
    :math:`\hat{Y_i} = \theta_0 + \frac{\theta_3 - \theta_0}{1+\rho}`.

    Here, theta is a 1-by-4 array. The first theta corresponds to lower limit
    (:math:`\theta_0`), the second corresponds to the effective dose (ED50) (:math:`\theta_1`), the third corresponds
    to the steepness of the curve (:math:`\theta_2`), and the fourth corresponds to the upper limit (:math:`\theta_3`).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 4 values. In general, starting values :math:`>0` are better choices for the
        4PL model
    X : ndarray, list, vector
        1-dimensional vector of `n` dose values.
    y : ndarray, list, vector
        1-dimensional vector of `n` response values.

    Returns
    -------
    array :
        Returns a 4-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_4p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_4p_logistic

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> resp_data = d[:, 0]   # Response data
    >>> dose_data = d[:, 1]   # Dose data

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_4p_logistic(theta=theta, X=dose_data, y=resp_data)

    The 4PL model and others are harder to solve compared to other estimating equations. Namely, the root-finder is
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
    warnings.warn("The `ee_#p_logistic` models will be removed in v3.0. Please use the new `ee_emax_sigmoid` model "
                  "instead. Also note that the parameter order has changed for `ee_emax_sigmoid`.", UserWarning)
    # Processing inputs
    X = np.asarray(X)        # Convert to NumPy array
    y = np.asarray(y)        # Convert to NumPy array

    # Creating rho to cut down on typing
    rho = (X / theta[1]) ** theta[2]

    # Generalized 4PL model function for y-hat
    fx = theta[0] + (theta[3] - theta[0]) / (1 + rho)

    # Using a special implementation of natural log here
    # nested_log = np.log(X / theta[1],             # ... to avoid dose=0 issues only take log
    #                     where=0 < X)              # ... where dose>0 (otherwise puts zero in place)
    nested_log = np.where(X > 0, np.log(X / theta[1]), 0)  # Handling when dose = 0

    # Calculate the derivatives for the gradient
    deriv = np.array((1 - 1/(1 + rho),                                          # Gradient for lower limit
                     (theta[3] - theta[0])*theta[2]/theta[1]*rho/(1 + rho)**2,  # Gradient for steepness
                     (theta[3] - theta[0])*nested_log*rho/(1 + rho)**2,         # Gradient for ED50
                     1 / (1 + rho)), )                                          # Gradient for upper limit

    # Compute gradient and return for each i
    return -2*(y - fx)*deriv


def ee_3p_logistic(theta, X, y, lower):
    r"""Estimating equations for the 3-parameter logistic model (3PL).

    Note
    ----
    This estimating equation is being depreciated in favor of ``ee_loglogistic`` in v3.0.


    The estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            -2 (Y_i - \hat{Y}_i) (1 - 1/(1 + \rho)) \\
            2 (Y_i - \hat{Y}_i) (\theta_3 - \theta_0) \frac{\theta_2}{\theta_1} \frac{\rho}{(1 + \rho)^2} \\
            2 (Y_i - \hat{Y}_i) (\theta_3 - \theta_0) \log(D_i / \theta_1) \frac{\rho}{(1 + \rho)^2} \\
        \end{bmatrix}
        = 0

    where :math:`R_i` is the response of individual :math:`i`, :math:`D_i` is the dose,
    :math:`\rho = \frac{D_i}{\theta_1}^{\theta_2}`, and
    :math:`\hat{Y_i} = \theta_0 + \frac{\theta_3 - \theta_0}{1+\rho}`.

    Here, theta is a 1-by-3 array for the 3PL. The first theta corresponds to the effective dose (ED50)
    (:math:`\theta_1`), the second corresponds to the steepness of the curve (:math:`\theta_2`), and the third
    corresponds to the upper limit (:math:`\theta_3`). The lower limit (:math:`\theta_0`, ``lower``) is pre-specified
    by the user (and is no longer estimated)

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 3 values. In general, starting values :math:`>0` are better choices for the
        3PL model
    X : ndarray, list, vector
        1-dimensional vector of `n` dose values.
    y : ndarray, list, vector
        1-dimensional vector of `n` response values.
    lower : int, float
        Set value for the lower limit.

    Returns
    -------
    array :
        Returns a 3-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_3p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_3p_logistic

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> resp_data = d[:, 0]   # Response data
    >>> dose_data = d[:, 1]   # Dose data

    Since there is a natural lower-bound of 0 for root growth, we set ``lower=0``. Defining psi, or the stacked
    estimating equations

    >>> def psi(theta):
    >>>     return ee_3p_logistic(theta=theta, X=dose_data, y=resp_data,
    >>>                           lower=0)

    The 3PL model and others are harder to solve compared to other estimating equations. See the advice provided in the
    ``ee_4p_logistic`` documentation.

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
    ee_dr = ee_4p_logistic(theta=[lower, theta[0], theta[1], theta[2]], X=X, y=y)
    return ee_dr[1:, :]


def ee_2p_logistic(theta, X, y, lower, upper):
    r"""Estimating equations for the 2-parameter logistic model (2PL).

    Note
    ----
    This estimating equation is being depreciated in favor of ``ee_loglogistic`` in v3.0.


    The estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            2 (Y_i - \hat{Y}_i) (\theta_3 - \theta_0) \frac{\theta_2}{\theta_1} \frac{\rho}{(1 + \rho)^2} \\
            2 (Y_i - \hat{Y}_i) (\theta_3 - \theta_0) \log(D_i / \theta_1) \frac{\rho}{(1 + \rho)^2} \\
        \end{bmatrix}
        = 0

    where :math:`R_i` is the response of individual :math:`i`, :math:`D_i` is the dose,
    :math:`\rho = \frac{D_i}{\theta_1}^{\theta_2}`, and
    :math:`\hat{Y_i} = \theta_0 + \frac{\theta_3 - \theta_0}{1+\rho}`.

    Here, theta is a 1-by-2 array for the 2PL. The first theta corresponds to the effective dose (ED50)
    (:math:`\theta_1`), and the second corresponds to the steepness of the curve (:math:`\theta_2`). The lower limit
    (:math:`\theta_0`, ``lower``) and upper limit (:math:`\theta_3`, ``upper``) are pre-specified by the user (and are
    no longer estimated)

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 2 values. In general, starting values :math:`>0` are better choices for the
        2PL model
    X : ndarray, list, vector
        1-dimensional vector of `n` dose values.
    y : ndarray, list, vector
        1-dimensional vector of `n` response values.
    lower : int, float
        Set value for the lower limit.
    upper : int, float
        Set value for the upper limit.

    Returns
    -------
    array :
        Returns a 2-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_2p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_2p_logistic

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> resp_data = d[:, 0]   # Response data
    >>> dose_data = d[:, 1]   # Dose data

    Since there is a natural lower-bound of 0 for root growth, we set ``lower=0``. While a natural upper bound does not
    exist for this example, we set ``upper=8`` for illustrative purposes. Defining psi, or the stacked estimating
    equations

    >>> def psi(theta):
    >>>     return ee_2p_logistic(theta=theta, X=dose_data, y=resp_data,
    >>>                           lower=0, upper=8)

    The 2PL model and others are harder to solve compared to other estimating equations. See the advice provided in the
    ``ee_4p_logistic`` documentation.

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
    ee_dr = ee_4p_logistic(theta=[lower, theta[0], theta[1], upper], X=X, y=y)
    return ee_dr[1:3, :]


def ee_effective_dose_delta(theta, y, delta, steepness, ed50, lower, upper):
    r"""Default stacked estimating equation to pair with the 4 parameter logistic model for estimation of the
    :math:`delta` effective dose.

    Note
    ----
    This estimating equation is being depreciated in favor of ``ee_loglogistic_ed`` in v3.0.


    The estimating equation is

    .. math::

        \sum_{i=1}^n \left\{ \theta_1 + \frac{\theta_4 - \theta_1}{1 + (\theta_5 / \theta_2)^{\theta_3}} -
        \theta_4(1-\delta) - \theta_1 \delta \right\} = 0

    where :math:`\theta_5` is the :math:`ED(\delta)`, and the other :math:`\theta` are from a PL model (1: lower limit,
    2: steepness, 3: ED(50), 4: upper limit). For proper uncertainty estimation, this estimating equation should be
    stacked with the correspond PL model.

    Parameters
    ----------
    theta : int, float
        Theta value corresponding to the ED(alpha).
    y : ndarray, list, vector
        1-dimensional vector of `n` response values, used to construct correct shape for output.
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
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equations for ED25 with ``ee_3p_logistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_2p_logistic, ee_effective_dose_delta

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> resp_data = d[:, 0]   # Response data
    >>> dose_data = d[:, 1]   # Dose data

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


def ee_emax(theta, dose, response):
    r"""Estimating equations for the (hyperbolic) E-max model, or Hill Equation. The E-max model describes the
    dose-response relationship as concave monotone definend by three parameters: the zero-dose response, the maximum
    response (E-max) and the dose producing half maximal effect (ED50). The assumed model is

    .. math::

        R = \theta_{z} + \frac{(\theta_{m} - \theta{z}) D}{\theta_{50} + D}

    where :math:`R` is the response and :math:`D` is the dose. Here, :math:`\theta_{z}` is the zero-dose response,
    :math:`\theta_{m}` is the maximum response and :math:`\theta_{50}` is the dose with 50% of maximal response. The
    corresponding estimating equations for this model are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            R_i - \theta_{z} - \frac{\theta_{m} D_i}{\theta_{50} + D_i} \\
            \left( R_i - \theta_{z} - \frac{\theta_{m} D_i}{\theta_{50} + D_i} \right) \times
            \left( \frac{D_i}{\theta_{50} + D_i} \right) \\
            \left( R_i - \theta_{z} - \frac{\theta_{m} D_i}{\theta_{50} + D_i} \right) \times
            \left( \frac{-\theta_{m} D_i}{\theta_{50} + D_i} \right) \\
        \end{bmatrix}
        = 0

    The first estimating equation is for the zero-dose response, the second estimating equations is for the maximum
    response and the third estimating equation is for 50% maximal response.

    Note
    ----
    This implementation supports both the E-max model (dose increases response) and the I-max model (dose decreases
    response). Depending on the relationship observed in the data, set the starting values for the `lower` and `upper`
    parameters according (i.e., E-max has `upper > lower` and I-max has `upper < lower`).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 2 values.
    dose : ndarray, list, vector
        1-dimensional vector of `n` dose values.
    response : ndarray, list, vector
        1-dimensional vector of `n` response values.

    Returns
    -------
    array :
        Returns a 2-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equations with ``ee_emax_model`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_emax

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly. Notice that here the response data is modified to correspond to the descrease in root length (since
    the E-max model assumes increase dose leads to increased response). This example is purely for illustration and
    one may not think this is the appopriately model for this context

    >>> d = load_inderjit()                   # Loading array of data
    >>> response = np.max(d[:, 0]) - d[:, 0]  # Response data
    >>> dose = d[:, 1]                        # Dose data

   Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_emax(theta=theta, dose=dose, response=response)

    This model can be difficult to solve. To make the solving process more stable, we provide starting values for the
    root-finding process based on the observed data

    >>> estr = MEstimator(psi, init=[np.min(response), np.max(response), np.median(dose)])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # Minimum response
    >>> estr.theta[1]    # Maximum response
    >>> estr.theta[2]    # Dose that results in 50% of max response

    References
    ----------
    Bonate PL. *Pharmacokinetic-Pharmacodynamic Modeling and Simulation* 2nd edition. pg 101.

    Felmlee MA, Morris ME, & Mager DE. (2012). Mechanism-based pharmacodynamic modeling. *Methods Mol Biol*,
    929, 583–600.

    Wagner JG. (1968). Kinetics of pharmacologic response I. Proposed relationships between response and drug
    concentration in the intact animal and man. *Journal of Theoretical Biology*, 20(2), 173-201.
    """
    # Processing inputs
    X = np.asarray(dose)                                         # Convert to NumPy array
    y = np.asarray(response)                                     # Convert to NumPy array
    e_0 = theta[0]                                               # Minimum effect parameter
    e_max = theta[1]                                             # Max effect parameter
    e_50 = theta[2]                                              # 50% max effect parameter

    # Computing estimating equations
    yhat = e_0 + (e_max * X) / (e_50 + X)                        # Predicted response
    r_contribution = y - yhat                                    # Response-contribution
    ee_0 = r_contribution                                        # E_min estimating equation
    ee_max = r_contribution * (X / (e_50 + X))                   # E_max estimating equation
    ee_ec50 = r_contribution * ((-e_max * X) / ((e_50 + X)**2))  # E_50 estimating equation

    # Returning stacked estimating equations
    return np.vstack([ee_0, ee_max, ee_ec50])


def ee_emax_ed(theta, dose, delta, ed50):
    """Estimating equation for the :math:`delta`-effective dose with the E-max model. The estimating
    equation is

    .. math::

        \sum_{i=1}^n \left\{ \theta_z + \frac{(\theta_m - \theta_l) \theta_d}{\theta{50} + \theta_d} -
        \theta_m(1 - \delta) - \theta_z \delta \right\} = 0

    where :math:`\theta_d` is the :math:`ED(\delta)`, and the other :math:`\theta` are from a E-max model. For
    proper uncertainty estimation, this estimating equation should be stacked with the E-max model.

    Parameters
    ----------
    theta : int, float
        Theta value corresponding to the ED(delta).
    dose : ndarray, list, vector
        1-dimensional vector of `n` response values, used to construct correct shape for output.
    delta : float
        The effective dose level of interest, ED(delta).
    ed50 : float
        Estimated parameter for the ED50 from the PL.

    Returns
    -------
    array :
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equations for ED25 with ``ee_loglogistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_emax, ee_emax_ed

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly. Notice that here the response data is modified to correspond to the descrease in root length (since
    the E-max model assumes increase dose leads to increased response). This example is purely for illustration and
    one may not think this is the appopriately model for this context

    >>> d = load_inderjit()                   # Loading array of data
    >>> response = np.max(d[:, 0]) - d[:, 0]  # Response data
    >>> dose = d[:, 1]                        # Dose data

   Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     e_model = ee_emax(theta=theta, X=dose, y=response)
    >>>     ed_25 = ee_emax_ed(theta[3], dose=dose, delta=0.25,
    >>>                        ed50=theta[2])
    >>>     return np.vstack((e_model,
    >>>                       ed_25,))

    Notice that the estimating equations are stacked in the order of the parameters in ``theta`` (the first 3 belong to
    3PL and the last belong to ED(25)).

    >>> estr = MEstimator(psi, init=[np.max(response),
    >>>                              (np.max(response)+np.min(response)) / 2,
    >>>                              (np.max(response)+np.min(response)) / 2,
    >>>                              (np.max(response)+np.min(response)) / 2])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # upper limit
    >>> estr.theta[1]    # ED(50)
    >>> estr.theta[2]    # steepness
    >>> estr.theta[3]    # ED(25)

    References
    ----------
    Bonate PL. *Pharmacokinetic-Pharmacodynamic Modeling and Simulation* 2nd edition. pg 153.
    """
    size = np.ones(np.asarray(dose).shape[0])
    ed_delta = delta / (1-delta) * ed50 - theta
    return size * ed_delta   # Returning estimating equations


def ee_loglogistic(theta, dose, response):
    """Estimating equations for the 4 parameter log-logistic dose-response model. The log-logistic model describes the
    dose-response relationship in terms of four parameters: the zero-dose response, the maximum response (E-max), the
    dose producing half maximal effect (ED50), and steepness of the dose-response curve. The assumed model is

    .. math::

        R_i = \theta_{z} + \frac{\theta_{m} - \theta_{z}}{1 + \exp\left[
        \theta_{s} (\log(D_i) - \log(\theta_{50})) \right]}

    where :math:`R` is the response and :math:`D` is the dose. Here, :math:`\theta_{z}` is the zero-dose response,
    :math:`\theta_{m}` is the maximum response, :math:`\theta_{50}` is the dose with 50% of maximal response, and
    :math:`\theta_{s}` is the slope. The corresponding estimating equations for this model are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            -2 (Y_i - \hat{Y}_i) (1 - 1/(1 + \rho)) \\
            2 (Y_i - \hat{Y}_i) (1 / (1 + \rho))) \\
            2 (Y_i - \hat{Y}_i) (\theta_m - \theta_z) \frac{\theta_2}{\theta_1} \frac{\rho}{(1 + \rho)^2} \\
            2 (Y_i - \hat{Y}_i) (\theta_m - \theta_z) \log(D_i / \theta_1) \frac{\rho}{(1 + \rho)^2}
        \end{bmatrix}
        = 0

    where :math:`R_i` is the response of individual :math:`i`, :math:`D_i` is the dose,
    :math:`\rho = \frac{D_i}{\theta_{50}}^{\theta_s}`, and
    :math:`\hat{Y_i} = \theta_z + \frac{\theta_m - \theta_z}{1+\rho}`.

    Here, theta is a 1-by-4 array. The first theta corresponds to lower limit (:math:`\theta_z`), the second
    corresponds to the upper limit (:math:`\theta_m`), the third corresponds to the effective dose (ED50)
    (:math:`\theta_{50}`), and the fourth corresponds to the steepness (:math:`\theta_s`).

    Note
    ----
    This implementation supports models where dose increases response and dose decreases response. Depending on the
    relationship observed in the data, set the starting values for the `lower` and `upper` parameters according (i.e.,
    increasing has `upper > lower` and descresing has `upper < lower`). Additionally, the `steepness` parameter should
    be positive for increasing and negative for decreasing. If starting parameters are not chosen well, the model
    may not converge or converge to nonsensical values.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 4 values. Outside of steepness (i.e., ``theta[2]``)In general, starting values
        :math:`>0` are recommended for the log-logistic model.
    dose : ndarray, list, vector
        1-dimensional vector of `n` dose values.
    response : ndarray, list, vector
        1-dimensional vector of `n` response values.

    Returns
    -------
    array :
        Returns a 4-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_emax_sigmoid`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_loglogistic

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> resp_data = d[:, 0]   # Response data
    >>> dose_data = d[:, 1]   # Dose data

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_loglogistic(theta=theta, dose=dose_data, response=resp_data)

    The sigmoid E-max model is harder to solve compared to other estimating equations. Namely, the root-finder is not
    aware of implicit bounds on the parameters. To reduce non-convergence issues, we can give the root-finder good
    starting values.

    Here, we use some general starting values that should perform well in many cases. For the lower-bound, give the
    minimum response value as the initial. For ED50, give the mid-point between the maximum response and the minimum
    response. From the scatter plot of the dose-response, we see that the response increases as dose increases. So,
    the starting value for steepness should be positive. For the upper-bound, give the maximum response value as the
    initial.

    >>> estr = MEstimator(psi, init=[np.min(resp_data),
    >>>                              np.max(resp_data),
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # lower limit
    >>> estr.theta[1]    # upper limit
    >>> estr.theta[2]    # ED50
    >>> estr.theta[3]    # steepness

    One can also use ``ee_emax_sigmoid`` to estimate 3-parameter and 2-parameter log-logistic models. For these models,
    a constant is added to the input array and only a subset of the output estimating equations are used. The following
    is an example of how to estimate a 3 parameter log-logistic model, which assumes that the lower limit of the
    response is zero (this makes sense in the context of this application).

    >>> def psi(theta):
    >>>     theta = [0, ] + list(theta)    # Setting lower-limit manually to zero
    >>>     ee = ee_loglogistic(theta=theta, dose=dose_data, response=resp_data)
    >>>     return ee[1:, :]               # Return all estimating equations besides lower limit

    Since we are no longer root-finding for the lower limit (i.e., it is manually set to zero in ``psi``), only three
    starting values need to be provided

    >>> estr = MEstimator(psi, init=[(np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              np.max(resp_data)])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates (notice that the indices differ from the usual sigmoid E-max, since we dropped
    the lower-limit estimating equation).

    >>> estr.theta[0]    # upper limit
    >>> estr.theta[1]    # ED50
    >>> estr.theta[2]    # steepness

    References
    ----------
    Ritz C, Baty F, Streibig JC, & Gerhard D. (2015). Dose-response analysis using R. *PloS One*, 10(12), e0146021.

    An H, Justin TL, Aubrey GB, Marron JS, & Dittmer DP. (2019). dr4pl: A Stable Convergence Algorithm for the 4
    Parameter Logistic Model. *R J.*, 11(2), 171.

    Inderjit, Streibig JC, & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
    allelopathy research. *Physiologia Plantarum*, 114(3), 422-428.
    """
    # Processing inputs
    X = np.asarray(dose)               # Convert to NumPy array
    X = np.where(X == 0, 1e-9, X)      # Removing dose of zero
    y = np.asarray(response)           # Convert to NumPy array
    e_0 = theta[0]                     # Minimum effect parameter
    e_max = theta[1]                   # Max effect parameter
    e_50 = theta[2]                    # 50% max effect parameter
    steep = theta[3]                   # Steepness parameter

    # Creating variables for computations
    rho = (X / e_50) ** steep                          # Short-hand for rho
    yhat = e_0 + (e_max - e_0) / (1 + rho)             # Predicted value of the response
    # This version does not work with autodiff. So, doing the np.where above instead
    # nested_log = np.log(X / e_50,                    # ... to avoid dose=0 issues only take log
    #                     where=0 < X)                 # ... where dose>0 (otherwise puts zero in place)
    nested_log = np.where(X > 0, np.log(X / e_50), 0)  # Handling when dose = 0

    # Score functions of the log-logistic model
    llimit = 1 - 1/(1 + rho)                                     # Gradient for lower limit
    ulimit = 1 / (1 + rho)                                       # Gradient for upper limit
    ed50 = (e_max - e_0) * steep / e_50 * rho / (1 + rho)**2     # Gradient for ED50
    steepness = (e_max - e_0) * nested_log * rho / (1 + rho)**2  # Gradient for steepness
    deriv = np.array([llimit, ulimit, ed50, steepness])          # Stacking the gradients together

    # Returning stacked estimating equations
    return (y - yhat) * deriv


def ee_loglogistic_ed(theta, dose, delta, lower, upper, ed50, steepness):
    """Estimating equation for the :math:`delta`-effective dose with the 4 parameter log-logistic model.
    The estimating equation is

    .. math::

        \sum_{i=1}^n \left\{ \theta_z + \frac{\theta_m - \theta_z}{1 + (\theta_d / \theta_{50})^{\theta_s}} -
        \theta_m(1-\delta) - \theta_z \delta \right\} = 0

    where :math:`\theta_d` is the :math:`ED(\delta)`, and the other :math:`\theta` are from a log-logistic model. For
    proper uncertainty estimation, this estimating equation should be stacked with the log-logistic model.

    Parameters
    ----------
    theta : int, float
        Theta value corresponding to the ED(delta).
    dose : ndarray, list, vector
        1-dimensional vector of `n` dose values, used to construct correct shape for output (not used in the
        estimating function).
    delta : float
        The effective dose level of interest, ED(delta).
    lower : int, float
        Estimated parameter or pre-specified constant for the lower limit. This should be a pre-specified constant for
        both the 3PL and 2PL.
    upper : int, float
        Estimated parameter or pre-specified constant for the lower limit. This should be a pre-specified constant for
        the 2PL.
    ed50 : float
        Estimated parameter for the ED50 from the PL.
    steepness : float
        Estimated parameter for the steepness from the PL.

    Returns
    -------
    array :
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equations for ED25 with ``ee_loglogistic`` should be done similar to the following

    >>> from delicatessen import MEstimator
    >>> from delicatessen.data import load_inderjit
    >>> from delicatessen.estimating_equations import ee_loglogistic, ee_loglogistic_ed

    For demonstration, we use dose-response data from Inderjit et al. (2002), which can be loaded from ``delicatessen``
    directly.

    >>> d = load_inderjit()   # Loading array of data
    >>> resp_data = d[:, 0]   # Response data
    >>> dose_data = d[:, 1]   # Dose data

    Since there is a natural lower-bound of 0 for root growth, we set ``lower=0``. While a natural upper bound does not
    exist for this example, we set ``upper=8`` for illustrative purposes. Defining psi, or the stacked estimating
    equations

    >>> def psi(theta):
    >>>     lower = 0
    >>>     pl_model = ee_loglogistic(theta=[lower, ]+list(theta), dose=dose_data, response=resp_data)
    >>>     ed_25 = ee_loglogistic_ed(theta[3], dose=dose_data, delta=0.25,
    >>>                               lower=lower, upper=theta[0],
    >>>                               ed50=theta[1], steepness=theta[2])
    >>>     return np.vstack((pl_model,
    >>>                       ed_25,))

    Notice that the estimating equations are stacked in the order of the parameters in ``theta`` (the first 3 belong to
    3PL and the last belong to ED(25)).

    >>> estr = MEstimator(psi, init=[np.max(resp_data),
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2,
    >>>                              (np.max(resp_data)+np.min(resp_data)) / 2])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting the parameter estimates

    >>> estr.theta[0]    # upper limit
    >>> estr.theta[1]    # ED(50)
    >>> estr.theta[2]    # steepness
    >>> estr.theta[3]    # ED(25)

    References
    ----------
    Ritz C, Baty F, Streibig JC, & Gerhard D. (2015). Dose-response analysis using R. *PloS One*, 10(12), e0146021.

    An H, Justin TL, Aubrey GB, Marron JS, & Dittmer DP. (2019). dr4pl: A Stable Convergence Algorithm for the 4
    Parameter Logistic Model. *R J.*, 11(2), 171.

    Inderjit, Streibig JC, & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
    allelopathy research. *Physiologia Plantarum*, 114(3), 422-428.
    """
    if steepness < 0:
        level = upper*delta - lower*(1-delta)
    else:
        level = upper*(1-delta) - lower*delta
    rho = (theta / ed50) ** steepness                          # Theta is the corresponds ED(alpha) value
    response = lower + (upper - lower) / (1 + rho)             # Formula for the response
    ed_delta = response - level                                # Effective dose
    return np.ones(np.asarray(dose).shape[0]) * ed_delta       # Return estimating equation
