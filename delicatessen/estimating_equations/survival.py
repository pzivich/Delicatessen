#####################################################################################################################
# Estimating functions for survival or time-to-event analyses
#####################################################################################################################

import warnings
import numpy as np

from delicatessen.estimating_equations.processing import generate_weights
from delicatessen.utilities import standard_normal_cdf, standard_normal_pdf


#################################################################
# Parametric Survival Estimating Equations


def ee_exponential_model(theta, t, delta):
    r"""Estimating equation for an exponential model. Let :math:`T_i` indicate the
    time of the event and :math:`C_i` indicate the time to right censoring. Therefore, the observable data consists of
    :math:`t_i = \min(T_i, C_i)` and :math:`\Delta_i = I(t_i = T_i)`. The estimating equation is

    .. math::

        \sum_{i=1}^n \left\{ \frac{\Delta_i}{\lambda} -  t_i \right\} = 0

    Here, :math:`\theta` is a single parameter that corresponds to the scale parameter for the exponential distribution.
    The hazard from the exponential model is parameterized as the following

    .. math::

        h(t) = \lambda

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in the case of the exponential model consists of a single value. Furthermore, the parameter will be
        non-negative. Therefore, an initial value like the ``[1, ]`` should be provided.
    t : ndarray, list, vector
        1-dimensional vector of `n` observed times.
    delta : ndarray, list, vector
        1-dimensional vector of `n` event indicators, where 1 indicates an event and 0 indicates right censoring.

    Returns
    -------
    array :
        Returns a 1-by-`n` NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_exponential_model`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_exponential_model

    Some generic survival data to estimate an exponential survival model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 5, 5, data['C'])
    >>> data['T'] = 0.8*np.random.weibull(a=1.0, size=n)
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_exponential_model(theta=theta,
    >>>                                     t=data['t'], delta=data['delta'])

    Calling the M-estimator

    >>> estr = MEstimator(stacked_equations=psi, init=[1.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting parameter the specific parameter estimates

    >>> estr.theta[0]     # lambda (scale)

    References
    ----------
    Collett D. (2015). Modelling survival data in medical research. CRC press.
    """
    # Converting input to NumPy arrays
    delta = np.asarray(delta)
    t = np.asarray(t)

    # Returning calculation for exponential distribution
    return (delta / theta) - t


def ee_weibull_model(theta, t, delta):
    r"""Estimating equation for a two-parameter Weibull model. Let :math:`T_i` indicate the time of the event and
    :math:`C_i` indicate the time to right censoring. Therefore, the observable data consists of
    :math:`t_i = min(T_i, C_i)` and :math:`\Delta_i = I(t_i = T_i)`. The estimating equations are

    .. math::

        \sum_{i=1}^n =
        \begin{bmatrix}
            \frac{\Delta_i}{\lambda} -  t_i^{\gamma} \\
            \frac{\Delta_i}{\gamma} + \Delta_i \log(t_i) - \lambda t_i^{\gamma} \log(t_i)
        \end{bmatrix}
        = 0

    Here, :math:`\theta` consists of two parameters for the Weibull model: the scale (:math:`\lambda`) and the shape
    (:math:`\gamma`). The hazard from the Weibull model is parameterized as the following

    .. math::

        h(t) = \lambda \gamma t^{\gamma - 1}

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in the case of the Weibull model consists of two values. Furthermore, the parameter will be
        non-negative. Therefore, an initial value like the ``[1, ]`` is recommended.
    t : ndarray, list, vector
        1-dimensional vector of `n` observed times. No missing data should be included (missing data may cause
        unexpected behavior).
    delta : ndarray, list, vector
        1-dimensional vector of `n` event indicators, where 1 indicates an event and 0 indicates right censoring. No
        missing data should be included (missing data may cause unexpected behavior).

    Returns
    -------
    array :
        Returns a 2-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_weibull_model`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_weibull_model

    Some generic survival data to estimate a Weibull survival model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 5, 5, data['C'])
    >>> data['T'] = 0.8*np.random.weibull(a=0.8, size=n)
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_weibull_model(theta=theta,
    >>>                                 t=data['t'], delta=data['delta'])

    Calling the M-estimator

    >>> estr = MEstimator(stacked_equations=psi, init=[1., 1.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting parameter the specific parameter estimates

    >>> estr.theta[0]     # lambda (scale)
    >>> estr.theta[1]     # gamma  (shape)

    References
    ----------
    Collett D. (2015). Modelling survival data in medical research. CRC press.
    """
    # Converting input to NumPy arrays
    delta = np.asarray(delta)
    t = np.asarray(t)

    # Extracting and naming parameters for my convenience
    lambd, gamma = theta[0], theta[1]             # Names / parameterization follow Collett

    # Calculating the contributions
    contribution_1 = (delta/lambd) - t**gamma     # Calculating estimating equation for lambda
    contribution_2 = ((delta/gamma)               # Calculating estimating equation for gamma
                      + (delta*np.log(t))
                      - (lambd * (t**gamma) * np.log(t)))

    # Returning stacked estimating equations
    return np.vstack((contribution_1,
                      contribution_2))


def ee_exponential_measure(theta, times, n, measure, scale):
    r"""Estimating equation to calculate a survival measure (survival, density, risk, hazard, cumulative hazard) from
    the exponential model. The estimating equation for the survival function at time :math:`t` is

    .. math::

        \sum_{i=1}^n \left\{ \exp(- \lambda t) - \theta \right\} = 0

    and the estimating equation for the hazard function at time :math:`t` is

    .. math::

        \sum_{i=1}^n \left\{ \lambda - \theta \right\} = 0

    For the other measures, we take advantage of the following transformations

    .. math::

        F(t) = 1 - S(t) \\
        H(t) = -\log(S(t)) \\
        f(t) = h(t) S(t)

    Note
    ----
    For proper uncertainty estimation, this estimating equation is meant to be stacked with ``ee_exponential_model``.


    Parameters
    ----------
    theta : ndarray, list, vector
        theta consists of t values. The initial values should consist of the same number of elements as provided in the
        ``times`` argument.
    times : int, float, ndarray, list, vector
        A single time or 1-dimensional collection of times to calculate the measure at. The number of provided times
        should consist of the same number of elements as provided in the ``theta`` argument.
    n : int
        Number of observations in the input data. This argument ensures that the dimensions of the estimating equation
        are correct given the number of observations in the data.
    measure : str
        Measure to calculate. Options include survival (``'survival'``), density (``'density'``), risk or the cumulative
        density (``'risk'``), hazard (``'hazard'``), or cumulative hazard (``'cumulative_hazard'``).
    scale : float, int
        The estimated scale parameter from the Weibull model. From ``ee_weibull_model``, will be the first element.

    Returns
    -------
    array :
        Returns a `t`-by-`n` NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_exponential_model`` and ``ee_exponential_measure`` should be done
    similar to the following. First, we will estimate the survival at time 5.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_exponential_model, ee_exponential_measure

    Some generic survival data to estimate an exponential model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 5, 5, data['C'])
    >>> data['T'] = 0.8*np.random.weibull(a=1.0, size=n)
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     exp = ee_exponential_model(theta=theta[0], t=data['t'],
    >>>                                delta=data['delta'])
    >>>     pred_surv_t = ee_exponential_measure(theta=theta[1], n=data.shape[0],
    >>>                                          times=5, measure='survival',
    >>>                                          scale=theta[0])
    >>>     return np.vstack((exp, pred_surv_t))

    Calling the M-estimator (note that `init` has 2 value, one for the scale and the other for :math:`S(t=5)`).

    >>> estr = MEstimator(stacked_equations=psi, init=[1., 0.5])
    >>> estr.estimate(solver='lm')

    Inspecting the estimate, variance, and confidence intervals for :math:`S(t=5)`

    >>> estr.theta[-1]                      # \hat{S}(t)
    >>> estr.variance[-1, -1]               # \hat{Var}(\hat{S}(t))
    >>> estr.confidence_intervals()[-1, :]  # 95% CI for S(t)

    Next, we will consider evaluating the survival function at multiple time points (so we can easily create a plot of
    the survival function and the corresponding confidence intervals)

    Note
    ----
    When calculate the survival (or other measures) at many time points, it is generally best to
    pre-wash the coefficients to reduce the number of iterations and total run-time.


    To make everything easier, we will generate a list of uniformly spaced values between the start and end points of
    our desired survival function. We will also generate initial values of the same length (to help the optimizer, we
    also start our starting values from near one and end near zero).

    >>> resolution = 50
    >>> time_spacing = list(np.linspace(0.01, 5, resolution))
    >>> fast_inits = list(np.linspace(0.99, 0.01, resolution))

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     exp = ee_exponential_model(theta=theta[0], t=data['t'],
    >>>                                delta=data['delta'])
    >>>     pred_surv_t = ee_exponential_measure(theta=theta[1:], n=data.shape[0],
    >>>                                          times=time_spacing, measure='survival',
    >>>                                          scale=theta[0])
    >>>     return np.vstack((exp, pred_surv_t))

    Calling the M-estimator

    >>> mestr = MEstimator(psi, init=list(estr.theta[0]) + fast_inits)
    >>> mestr.estimate(solver="lm")

    To plot the survival curves, we could do the following:

    >>> import matplotlib.pyplot as plt
    >>> ci = mestr.confidence_intervals()[1:, :]  # Extracting relevant CI
    >>> plt.fill_between(time_spacing, ci[:, 0], ci[:, 1], alpha=0.2)
    >>> plt.plot(time_spacing, mestr.theta[1:], '-')
    >>> plt.show()

    References
    ----------
    Collett D. (2015). Modelling survival data in medical research. CRC press.
    """
    # Lazy approach that just calls existing weibull measure function (exponential is a Weibull with shape=1
    return ee_weibull_measure(theta=theta, times=times,
                              n=n, measure=measure,
                              scale=scale, shape=1)


def ee_weibull_measure(theta, times, n, measure, scale, shape):
    r"""Estimating equation to calculate a survival measure (survival, density, risk, hazard, cumulative hazard) for
    the Weibull model. The estimating equation for the survival function at time :math:`t` is

    .. math::

        \sum_{i=1}^n \left\{ \exp(- \lambda t^{\gamma}) - \theta \right\} = 0

    and the estimating equation for the hazard function at time :math:`t` is

    .. math::

        \sum_{i=1}^n \left\{ \lambda \gamma t^{\gamma - 1} - \theta \right\} = 0

    For the other measures, we take advantage of the following transformation between survival measures

    .. math::

        F(t) = 1 - S(t) \\
        H(t) = -\log(S(t)) \\
        f(t) = h(t) S(t)

    Note
    ----
    For proper uncertainty estimation, this estimating equation is meant to be stacked with ``ee_weibull_model``.

    Parameters
    ----------
    theta : ndarray, list, vector
        theta consists of `t` values. The initial values should consist of the same number of elements as provided in
        the ``times`` argument.
    times : int, float, ndarray, list, vector
        A single time or 1-dimensional collection of times to calculate the measure at. The number of provided times
        should consist of the same number of elements as provided in the ``theta`` argument.
    n : int
        Number of observations in the input data. This argument ensures that the dimensions of the estimating equation
        are correct given the number of observations in the data.
    measure : str
        Measure to calculate. Options include survival (``'survival'``), density (``'density'``), risk or the cumulative
        density (``'risk'``), hazard (``'hazard'``), or cumulative hazard (``'cumulative_hazard'``).
    scale : float, int
        The estimated scale parameter from the Weibull model. From ``ee_weibull_model``, will be the first element.
    shape :
        The estimated shape parameter from the Weibull model. From ``ee_weibull_model``, will be the second (last)
        element.

    Returns
    -------
    array :
        Returns a `t`-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_weibull_model`` and ``ee_weibull_measure`` should be done
    similar to the following. First, we will estimate the survival at time 5.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_weibull_model, ee_weibull_measure

    Some generic survival data to estimate a Weibull model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 5, 5, data['C'])
    >>> data['T'] = 0.8*np.random.weibull(a=0.8, size=n)
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     exp = ee_weibull_model(theta=theta[0:2], t=data['t'],
    >>>                            delta=data['delta'])
    >>>     pred_surv_t = ee_weibull_measure(theta=theta[2], n=data.shape[0],
    >>>                                      times=5, measure='survival',
    >>>                                      scale=theta[0], shape=theta[1])
    >>>     return np.vstack((exp, pred_surv_t))

    Calling the M-estimator

    >>> estr = MEstimator(stacked_equations=psi, init=[1., 1., 0.5])
    >>> estr.estimate(solver='lm')

    Inspecting the estimate, variance, and confidence intervals for :math:`S(t=5)`

    >>> estr.theta[-1]                      # \hat{S}(t)
    >>> estr.variance[-1, -1]               # \hat{Var}(\hat{S}(t))
    >>> estr.confidence_intervals()[-1, :]  # 95% CI for S(t)

    Next, we will consider evaluating the survival function at multiple time points (so we can easily create a plot of
    the survival function and the corresponding confidence intervals)

    Note
    ----
    When calculate the survival (or other measures) at many time points, it is generally best to
    pre-wash the coefficients to reduce the number of iterations and total run-time.


    To make everything easier, we will generate a list of uniformly spaced values between the start and end points of
    our desired survival function. We will also generate initial values of the same length (to help the optimizer, we
    also start our starting values from near one and end near zero).

    >>> resolution = 50
    >>> time_spacing = list(np.linspace(0.01, 5, resolution))
    >>> fast_inits = list(np.linspace(0.99, 0.01, resolution))

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     exp = ee_weibull_model(theta=theta[0:2], t=data['t'],
    >>>                            delta=data['delta'])
    >>>     pred_surv_t = ee_weibull_measure(theta=theta[2:], n=data.shape[0],
    >>>                                      times=time_spacing, measure='survival',
    >>>                                      scale=theta[0], shape=theta[1])
    >>>     return np.vstack((exp, pred_surv_t))

    Calling the M-estimator

    >>> mestr = MEstimator(psi, init=list(estr.theta[0:2]) + fast_inits)
    >>> mestr.estimate(solver="lm")

    To plot the survival curves, we could do the following:

    >>> import matplotlib.pyplot as plt
    >>> ci = mestr.confidence_intervals()[2:, :]  # Extracting relevant CI
    >>> plt.fill_between(time_spacing, ci[:, 0], ci[:, 1], alpha=0.2)
    >>> plt.plot(time_spacing, mestr.theta[2:], '-')
    >>> plt.show()

    References
    ----------
    Collett D. (2015). Modelling survival data in medical research. CRC press.
    """
    lambd, gamma = scale, shape

    def calculate_metric(time, theta_t):
        # Intermediate calculations
        survival_t = np.exp(-lambd * (time ** gamma))*np.ones(n)     # Survival calculation from parameters
        hazard_t = lambd*gamma*(time**(gamma-1))*np.ones(n)          # hazard calculation from parameters

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
# Accelerated Failure Time Models


def ee_aft_weibull(theta, X, t, delta, weights=None):
    r"""Estimating equation for accelerated failure time (AFT) model with a Weibull distribution. Let :math:`T_i`
    indicate the time of the event and :math:`C_i` indicate the time to right censoring. Therefore, the observable data
    consists of :math:`t_i = min(T_i, C_i)` and :math:`\Delta_i = I(t_i = T_i)`. The estimating equations are

    .. math::

        \sum_{i=1}^n =
        \begin{bmatrix}
            \frac{\Delta_i}{\lambda} -  t_i^{\gamma} \exp(\beta X_i) \\
            \Delta_i X_i - (\lambda  t_i^{\gamma} \exp(\beta X_i))X_i \\
            \frac{\Delta_i}{\gamma} + \Delta_i \log(t) - \lambda t_i^{\gamma} \exp(\beta X_i) \log(t)
        \end{bmatrix}
        = 0

    The AFT consists of the following parameters: :math:`\mu, \beta, \sigma`. The above
    estimating equations use the proportional hazards form of the Weibull model. For the Weibull AFT, notice the
    following relation between the coefficients: :math:`\lambda = - \mu \gamma`,
    :math:`\beta_{PH} = - \beta_{AFT} \gamma`, and :math:`\gamma = \exp(\sigma)`.

    Here, :math:`\theta` is a 1-by-(2+`b`) array, where `b` is the distinct covariates included as part of ``X``. For
    example, if ``X`` is a 3-by-`n` matrix, then theta will be a 1-by-5 array. The code is general to allow for an
    arbitrary dimension of ``X``.

    Parameters
    ----------
    theta : ndarray, list, vector
        theta consists of 1+`b`+1 values. Therefore, initial values should consist of the same number as the number of
        columns present in ``X`` plus 2. This can easily be implemented via
        ``[0, ] + [0, ] * X.shape[1] + [0, ]``.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
    t : ndarray, list, vector
        1-dimensional vector of `n` observed times.
    delta : ndarray, list, vector
        1-dimensional vector of `n` values indicating whether the time was an event or censoring.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a 1+`b`+1-by-`n` NumPy array evaluated for the input ``theta``.

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
    >>>         return ee_aft_weibull(theta=theta, X=d_obs[['X', 'W']],
    >>>                               t=d_obs['t'], delta=d_obs['delta'])

    Calling the M-estimator (note that `init` has 4 values now, since ``X.shape[1]`` is 2).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0.])
    >>> estr.estimate(solver='hybr')

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
    warnings.warn("ee_aft_weibull will be removed in v4.0. Please use ee_aft instead.", FutureWarning)
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
    r"""Estimating equation to calculate a survival measure (survival, density, risk, hazard, cumulative hazard) given
    a specific covariate pattern and Weibull accelerated failure time (AFT) model. The estimating equation for the
    survival function at time :math:`t` is

    .. math::

        \sum_{i=1}^n \left\{ \exp(-1 \lambda_i t^{\gamma}) - \theta \right\} = 0

    and the estimating equation for the hazard function at time :math:`t` is

    .. math::

        \sum_{i=1}^n  \left\{ \lambda_i \gamma t^{\gamma - 1} - \theta \right\} = 0

    where

    .. math::

        \gamma = \exp(\sigma) \\
        \lambda_i = \exp(-1 (\mu + X \beta) * \gamma)

    For the other measures, we take advantage of the following transformation between survival meaures

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
        theta consists of `t` values. The initial values should consist of the same number of elements as provided in the
        ``times`` argument.
    times : int, float, ndarray, list, vector
        A single time or 1-dimensional collection of times to calculate the measure at. The number of provided times
        should consist of the same number of elements as provided in the ``theta`` argument.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
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
        Returns a `t`-by-`n` NumPy array evaluated for the input theta

    Examples
    --------
    Construction of a estimating equations for :math:`S(t=5)` with ``ee_aft_weibull_measure`` should be done similar to
    the following

    >>> import numpy as np
    >>> import pandas as pd
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

    Calling the M-estimator

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
    When calculate the survival (or other measures) at many time points, it is generally best to
    pre-wash the coefficients to reduce the number of iterations and total run-time.


    To make everything easier, we will generate a list of uniformly spaced values between the start and end points of
    our desired survival function. We will also generate initial values of the same length (to help the optimizer, we
    also start our starting values from near one and end near zero).

    >>> resolution = 50
    >>> time_spacing = list(np.linspace(0.01, 8, resolution))
    >>> fast_inits = list(np.linspace(0.99, 0.01, resolution))

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     aft = ee_aft_weibull(theta=theta[0:4],
    >>>                     t=d_obs['t'], delta=d_obs['delta'], X=d_obs[['X', 'W']])
    >>>     pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=d_coef[['X', 'W']],
    >>>                                          times=time_spacing, measure='survival',
    >>>                                          mu=theta[0], beta=theta[1:3], sigma=theta[3])
    >>>     return np.vstack((aft, pred_surv_t))

    Calling the M-estimator

    >>> estr = MEstimator(psi, init=list(estr.theta[0:4]) + fast_inits)
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


def ee_aft(theta, X, t, delta, distribution, weights=None):
    r"""Estimating equation for a generalized accelerated failure time (AFT) model. Let :math:`T_i` indicate the time
    of the event and :math:`C_i` indicate the time to right censoring. Therefore, the observable data consists of
    :math:`t_i = \min(T_i, C_i)` and :math:`\Delta_i = I(t_i = T_i)`. The estimating equations are

    .. math::

        \sum_{i=1}^n =
        \begin{bmatrix}
            - \sigma^{-1} \lambda_\epsilon X^T \\
            - \sigma^{-1} \lambda_\epsilon  Z_i - \delta \sigma^{-1} \\
        \end{bmatrix}
        = 0

    where :math:`\theta = (\beta, \sigma)`, :math:`Z_i = \frac{\log(t_i) - X \beta^T}{\sigma}` and

    .. math::

        \lambda_\epsilon = \Delta_i \frac{f_\epsilon'(Z_i)}{f_\epsilon(Z_i)}
                           - (1 - \Delta_i) \frac{S_\epsilon'(Z_i)}{S_\epsilon(Z_i)}.

    Here the choice of the distribution for :math:`f_\epsilon` and :math:`S_\epsilon` are determined by the specified
    distributions. Options include exponential, Weibull, log-logistic, and log-normal. The design matrix :math:`X`
    should include an intercept term. Note that for optimization, the starting values for the intercept term should be
    a positive number (e.g., :math:`5`).

    Note
    ----
    The parametrization of the AFT model is the same as R's ``survival`` library, except for scale parameter. Here,
    the inverse of the scale is equal to the R implementation.


    Here, :math:`\theta` is a 1-by-(`b`+1) array, where `b` is the distinct covariates included as part of ``X``. For
    example, if ``X`` is a 3-by-`n` matrix, then theta will be a 1-by-4 array. The code is general to allow for an
    arbitrary dimension of ``X``.

    Parameters
    ----------
    theta : ndarray, list, vector
        theta consists of `b`+1 values. Therefore, initial values should consist of the same number as the number of
        columns present in ``X`` plus 1. This can easily be implemented via ``[0, ] * X.shape[1] + [0, ]``. Note that
        if using an exponential model, only `b` values need to be provided.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
    t : ndarray, list, vector
        1-dimensional vector of `n` observed times.
    delta : ndarray, list, vector
        1-dimensional vector of `n` values indicating whether the time was an event or censoring.
    distribution : str
        Distribution to use for the AFT model. Options are ``'exponential'`` (exponential), ``'weibull'`` (Weibull),
        ``'log-logistic'`` (log-logistic), and ``'log-normal'`` (log-normal).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a `b`+1-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_aft`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_aft

    Some generic survival data to estimate a AFT regression model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> data['T'] = (1/1.25 + 1/np.exp(0.5)*data['X'])*np.random.weibull(a=0.75, size=n)
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 10, 10, data['C'])
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])
    >>> d_obs = data[['X', 't', 'delta']].copy()
    >>> d_obs['C'] = 1

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     aft = ee_aft(theta, t=d_obs['t'], delta=d_obs['delta'], X=d_obs[['X', 'W']], distribution='weibull')
    >>>     return aft

    Calling the M-estimator

    >>> estr = MEstimator(stacked_equations=psi, init=[2., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Inspecting parameter the specific parameter estimates

    >>> estr.theta[0]     # beta_0     (scale)
    >>> estr.theta[1]     # beta_X     (scale coefficients)
    >>> estr.theta[-1]    # log(sigma) (shape)

    References
    ----------
    Collett D. (2015). Accelerated failure time and other parametric models. In: Modelling survival data in medical
    research. CRC press. pg171-220
    """
    X = np.asarray(X)                          # Convert to NumPy array
    t = np.asarray(t)[:, None]                 # Convert to NumPy array and ensure correct shape for matrix algebra
    delta = np.asarray(delta)[:, None]         # Convert to NumPy array and ensure correct shape for matrix algebra
    beta_dim = X.shape[1]

    # Extract coefficients
    beta = np.asarray(theta[:beta_dim])[:, None]
    if distribution == 'exponential':
        sigma = 1
    else:
        sigma = np.exp(-theta[-1])

    # Computing error distribution for each observation
    z_i = (np.log(t) - np.dot(X, beta)) / sigma

    # Allowing for a weighted Weibull-AFT model
    if weights is None:                         # If weights is unspecified
        w = np.ones(X.shape[0])                 # ... assign weight of 1 to all observations
    else:                                       # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector

    # Handling different distribution specifications
    if distribution in ['exponential', 'weibull']:
        df_f = 1 - np.exp(z_i)
        dS_S = np.exp(z_i)
    elif distribution in ['log-logistic', 'loglogistic']:
        df_f = 1 - (2 * np.exp(z_i)) / (1 + np.exp(z_i))
        dS_S = np.exp(z_i) / (1 + np.exp(z_i))
    elif distribution in ['log-normal', 'lognormal']:
        df_f = -z_i
        dS_S = standard_normal_pdf(z_i) / (1 - standard_normal_cdf(z_i))
    else:
        raise ValueError("Invalid distribution: " + str(distribution))
    lambda_epsilon = delta*df_f - (1-delta)*dS_S

    # Contributions to the estimating functions
    score_scale = -1/sigma * lambda_epsilon * X
    if distribution == 'exponential':
        efunc = w * score_scale.T
    else:
        score_shape = (-1 / sigma * lambda_epsilon * z_i) - (delta / sigma)
        efunc = np.vstack((w * score_scale.T, w * score_shape.T))

    # Output b-by-n matrix
    return efunc
