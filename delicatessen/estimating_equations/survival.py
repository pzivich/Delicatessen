#####################################################################################################################
# Estimating functions for survival or time-to-event analyses
#####################################################################################################################

import warnings
import numpy as np

from delicatessen.estimating_equations.processing import generate_weights
from delicatessen.utilities import standard_normal_cdf, standard_normal_pdf


#################################################################
# Parametric Survival Estimating Equations

def ee_survival_model(theta, t, delta, distribution):
    r"""Estimating equation for a parametric survival models. Let :math:`T_i` indicate the time of the event and
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
    (:math:`\gamma`). The parameterization of the different survival analysis models are described in the following
    table

    .. list-table::
       :widths: 25 25 25 25
       :header-rows: 1

       * - Distribution
         - Keyword
         - Parameters
         - :math:`H(t)`
       * - Exponential
         - ``exponential``
         - :math:`\lambda`
         - :math:`\lambda t`
       * - Weibull
         - ``weibull``
         - :math:`\lambda, \gamma`
         - :math:`\lambda t^{\gamma}`


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
    distribution : str
        Distribution for the parametric survival model.

    Returns
    -------
    array :
        Returns a `p`-by-`n` NumPy array evaluated for the input ``theta``, where `p` is the number of parameters in
        the model.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_survival_model`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_survival_model

    Some generic survival data to estimate a parametric survival model with

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['C'] = np.random.weibull(a=1, size=n)
    >>> data['C'] = np.where(data['C'] > 5, 5, data['C'])
    >>> data['T'] = 0.8*np.random.weibull(a=0.8, size=n)
    >>> data['delta'] = np.where(data['T'] < data['C'], 1, 0)
    >>> data['t'] = np.where(data['delta'] == 1, data['T'], data['C'])

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_survival_model(theta=theta,
    >>>                                  t=data['t'], delta=data['delta'],
    >>>                                  distribution='weibull')

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

    To generate predictions from this model, please use ``delicatessen.utilities.survival_predictions``. See the
    corresponding documentation for further details.

    References
    ----------
    Collett D. (2015). Parametric proportional hazards models In: Modelling Survival Data in Medical Research.
    CRC press. pg 178-192
    """
    # Converting input to NumPy arrays
    delta = np.asarray(delta)
    t = np.asarray(t)
    distribution = distribution.lower()

    # Extracting and naming parameters for my convenience
    if distribution == 'exponential':
        lambd = theta[0]
        gamma = 1
    else:
        lambd, gamma = theta[0], theta[1]

    # Calculating the contributions
    contribution_1 = (delta/lambd) - t**gamma     # Calculating estimating equation for lambda
    contribution_2 = ((delta/gamma)               # Calculating estimating equation for gamma
                      + (delta*np.log(t))
                      - (lambd * (t**gamma) * np.log(t)))

    # Returning stacked estimating equations
    if distribution == 'exponential':
        return contribution_1
    else:
        return np.vstack((contribution_1,
                          contribution_2))


#################################################################
# Accelerated Failure Time Models

def ee_aft(theta, X, t, delta, distribution, weights=None):
    r"""Estimating equation for a generalized accelerated failure time (AFT) model. Let :math:`T_i` indicate the time
    of the event and :math:`C_i` indicate the time to right censoring. Therefore, the observable data consists of
    :math:`t_i = \min(T_i, C_i)` and :math:`\Delta_i = I(t_i = T_i)`. The estimating equations are

    .. math::

        \sum_{i=1}^n =
        \begin{bmatrix}
            - \frac{1}{\sigma} \lambda_\epsilon (Z_i) X^T \\
            - \frac{1}{\sigma} \lambda_\epsilon (Z_i) \times  Z_i - \frac{\Delta_i}{\sigma} \\
        \end{bmatrix}
        = 0

    where :math:`\theta = (\beta, \sigma)`, :math:`Z_i = \frac{\log(t_i) - X \beta^T}{\sigma}` and

    .. math::

        \lambda_\epsilon = \Delta_i \frac{f_\epsilon'(Z_i)}{f_\epsilon(Z_i)}
                           - (1 - \Delta_i) \frac{S_\epsilon'(Z_i)}{S_\epsilon(Z_i)}.

    Here the choice of the distribution for :math:`f_\epsilon` and :math:`S_\epsilon` are determined by the specified
    distributions. Options include exponential, Weibull, log-logistic, and log-normal. These functions are described in
    the following table

    .. list-table::
       :widths: 25 25 25 25
       :header-rows: 1

       * - Distribution
         - Keyword
         - :math:`f_\epsilon' / f_\epsilon`
         - :math:`S_\epsilon' / S_\epsilon`
       * - Exponential
         - ``exponential``
         - :math:`1-\exp(Z_i)`
         - :math:`\exp(Z_i)`
       * - Weibull
         - ``weibull``
         - :math:`1-\exp(Z_i)`
         - :math:`\exp(Z_i)`
       * - Log-Logistic
         - ``log-logistic``
         - :math:`1 - \frac{2 \exp(Z_i)}{1 + \exp(Z_i)}`
         - :math:`\frac{\exp(Z_i)}{1 + \exp(Z_i)}`
       * - Log-Normal
         - ``log-normal``
         - :math:`- Z_i`
         - :math:`\frac{\phi(Z_i)}{1 - \Phi(Z_i)}`

    The design matrix :math:`X` should include an intercept term. Note that for optimization, the starting values for
    the intercept term likely be a positive number (e.g., :math:`5`).

    Note
    ----
    The parametrization of the AFT model is the same as R's ``survival`` library, except for scale parameter. Here,
    the inverse of the scale is equal to the R implementation.


    Here, :math:`\theta` is a 1-by-(`b` + 1) array, where `b` is the distinct covariates included as part of ``X``. For
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
        Distribution to use for the AFT model. See table for options.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a (`b` + 1)-by-`n` NumPy array evaluated for the input ``theta``.

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

    Note
    ----
    Optimization of the AFT model can be difficult. It may help to fit an exponential AFT model first and then use
    those coefficients as starting values for a more general model.


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
    Collett D. (2015). Accelerated failure time and other parametric models. In: Modelling Survival Data in Medical
    Research. CRC press. pg 236-250
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
