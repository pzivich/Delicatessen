#####################################################################################################################
# Estimating functions for survival or time-to-event analyses
#####################################################################################################################

import warnings
import numpy as np

from delicatessen.errors import check_survival_data_valid
from delicatessen.estimating_equations.processing import generate_weights
from delicatessen.utilities import inverse_logit, standard_normal_cdf, standard_normal_pdf


#################################################################
# Parametric Survival Estimating Equations

def ee_survival_model(theta, t, delta, distribution):
    r"""Estimating equation for a parametric survival models. Let :math:`T_i` indicate the time of the event and
    :math:`C_i` indicate the time to right censoring. Therefore, the observable data consists of
    :math:`t_i = min(T_i, C_i)` and :math:`\Delta_i = I(t_i = T_i)`. The general estimating equations are

    .. math::

        \sum_{i=1}^n =
        \begin{bmatrix}
            \Delta_i \frac{f'(t_i; \theta)}{f(t_i; \theta)} + (1-\Delta_i) \frac{S'(t_i; \theta)}{S(t_i; \theta)}
        \end{bmatrix}
        = 0

    Here, :math:`\theta` consists of parameters for the corresponding model. Note that this estimating equation
    implicitly assumes that the event and censoring times are independent. See the table below for the different
    survival models and their parametrization in terms of the hazard function.

    .. list-table::
       :widths: 25 25 25 25
       :header-rows: 1

       * - Distribution
         - Keyword
         - Parameters
         - :math:`h(t)`
       * - Exponential
         - ``exponential``
         - :math:`\lambda`
         - :math:`\lambda`
       * - Weibull
         - ``weibull``
         - :math:`\lambda, \gamma`
         - :math:`\lambda \gamma t^{\gamma - 1}`
       * - Gompertz
         - ``weibull``
         - :math:`\lambda, \gamma`
         - :math:`\lambda \exp(\gamma t)`


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

    >>> estr.theta

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

    # Error checking for survival data formatting
    check_survival_data_valid(delta=delta, time=t)

    # Extracting and naming parameters for my convenience
    if distribution == 'exponential':
        lambd = theta[0]
        gamma = 1
    # When implementing generalized gamma, can use
    # elif distribution == 'generalized-gamma': lambd, gamma, [...] = theta[0], theta[1], theta[2]
    else:
        lambd, gamma = theta[0], theta[1]

    # Handling specified distribution
    if distribution in ['exponential', 'weibull']:
        # Calculating the contributions
        ef_lambda = (delta/lambd) - t**gamma     # Calculating estimating equation for lambda
        ef_gamma = ((delta/gamma)                # Calculating estimating equation for gamma
                    + (delta*np.log(t))
                    - (lambd * (t**gamma) * np.log(t)))
    elif distribution == 'gompertz':
        exp_gt = np.exp(gamma*t)
        ef_lambda = delta/lambd - (exp_gt - 1)/gamma
        ef_gamma = lambd/(gamma**2)*(exp_gt-1) + delta*t - (lambd/gamma)*exp_gt*t
    else:
        raise ValueError("The distribution '" + str(distribution) + "' was specified, but only the following "
                         "distributions are supported: 'exponential', 'weibull', 'gompertz'.")

    # Returning stacked estimating equations
    if distribution == 'exponential':
        return ef_lambda
    else:
        return np.vstack((ef_lambda,
                          ef_gamma))


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

    # Error checking for survival data formatting
    check_survival_data_valid(delta=delta, time=t)

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
        raise ValueError("The distribution '" + str(distribution) + "' was specified, but only the following "
                         "distributions are supported: 'exponential', 'weibull', 'log-logistic', 'log-normal'.")

    # Individual contributions
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


#################################################################
# Discrete-Time Models


def ee_plogit(theta, X, t, delta, S=None, unique_times=None, weights=None):
    r"""Estimating equation for pooled logistic regression with discrete-time survival data. One way to model survival
    data is to use survival stacking, where survival data is discretized into intervals. These intervals are expanded
    into a 'long' data set, where each row corresponds to a unique person-period. This standard implementation of
    pooled logistic regression can be computationally expensive. This estimating equation utilizes a re-expression that
    does not depend on the creation of a long data set. See the references for further details.

    The estimating equations for the implementation that does not require a long data set are

    .. math::

        \sum_{i=1}^n \sum_{k \in \mathcal{K}}
        \left\{ I(T_i^* > s_{k-1})
        \left( Y_{i,k} - \hat{Y}_{i,k} \right)
        \begin{bmatrix}
            \mathbb{X}_i^T \\
            \mathbb{S}_{i,k}^T \\
        \end{bmatrix}
        \right\}
        = 0

    where :math:`T_i^*` is the observed (continuous time),
    :math:`Y_{i,k} = I(s_{k-1} < T_i^* \le s_{k}) \Delta_i` is the indicator that the event happened in interval
    :math:`s_k`,
    :math:`\hat{Y}_{i,k} = \text{expit}(\mathbb{X}_i \theta_x^T + \mathbb{S}_{i,k} \theta_s^T)` is the predicted
    probability of the event for interval :math:`k` where :math:`\theta_x` are the coefficients for the covariates
    and :math:`\theta_s` are the coefficients for the times,
    :math:`\mathbb{X}` is the design matrix for baseline covariates,
    and :math:`\mathbb{S}` is the design matrix for time.

    Note
    ----
    Time-varying covariates are not currently supported for ``ee_plogit``.


    Here, :math:`\mathbb{S}` controls how the hazard is allowed to vary over time and is specified by the ``S``
    argument. The default is ``None``, which automatically uses disjoint indicators. This approach is highly flexible
    and makes no parametric constraints on the functional form for the underlying (discrete-time) hazard function.
    When specifying ``S``, a `p`-by-`K` matrix is expected, where `p` is the number of parameters and `K` is the number
    of discrete time points (as detected in the data). See the examples for advice on constructing this design matrix.

    Here, :math:`\theta` is a 1-by-(`b`+`K`) array, which corresponds to the coefficients in the corresponding pooled
    logistic regression model and `b` is the distinct covariates included as part of ``X`` and `K` is the number of
    unique time intervals in ``t``. Note that `K` varies by whether and how ``S`` is specified. See examples for
    further details.

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
    S : ndarray, list, vector, None, optional
        Design matrix for how the (discrete-time) hazard can vary over time. Default is ``None``, which uses a disjoint
        indicator matrix for time. Here, a computational trick described in Zivich et al. is used to reduce the
        computational burden. As such, the time design matrix is only dimension `K*` which corresponds to the number of
        unique *event* times.
        If providing a matrix, it is expected to have dimensions `p`-by-`K`, where `p` is the number of parameters
        (columns) and `K` is the number of unique times with a one-unit increase.
    unique_times : ndarray, list, vector, None, optional
        This argument is only used when ``S=None``, otherwise it is ignored. This function allows a user to restrict
        the automatic disjoint indicator matrix to a subset of times. This feature is only intended to be used when
        evaluating a stratified pooled logistic model and to maintain the computational trick described in Zivich et al.
        See examples for further details.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a (`b`+`K`)-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_plogit`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_plogit
    >>> from delicatessen.data import load_breast_cancer

    Here, we will illustrate pooled logistic regression with breast cancer from the Middlesex Hospital in July 1987.
    This data can be loaded as follows

    >>> d = pd.DataFrame(load_breast_cancer(), columns=['d', 't', 'statin'])

    To start, we will demonstrate pooled logistic regression where time is modeled using the automatically generated
    design matrix with disjoint indicators. First, we will assess how many unique event times exist in this data. This
    can easily be done as follows

    >>> unique_event_times = list(np.unique(d.loc[d['d'] == 1, 't']))

    Next, we specify the estimating function and our data.

    >>> def psi(theta):
    >>>     return ee_plogit(theta=theta, X=d[['statin', ]], delta=d['d'], t=d['t'])

    Next, M-estimator procedure can be called to estimate ``theta``. Note that ``init`` requires ``X.shape[1]`` plus
    ``len(unique_event_times)`` starting values. To help speed up the parameter solving process, it can be helpful to
    specify the intercept as a negative value (``-3`` as below). Note that the time coefficients follows after the
    covariate coefficients in the inputs.

    >>> inits = [0., ] + [-3., ] + [0., ]*(len(unique_times) - 1)
    >>> estr = MEstimator(stacked_equations=psi, init=inits)
    >>> estr.estimate()

    Inspecting the covariate parameter estimates and variance

    >>> estr.theta[0]
    >>> estr.variance[0, 0]

    As described elsewhere, the covariate coefficients can be broadly interpreted as (approximations of that) hazard
    ratios from a Cox proportional hazards model.

    Pooled logistic also allows one to specify parametric functional forms for how the discrete-time hazard evolves
    over time. Some of these specifications correspond with other well-known survival models. Here, we will model time
    as linear, which is analogous to a Gompertz model. Here, we need to manually create ``S``, which can be done
    using the following process

    >>> t_steps = np.asarray(range(1, np.max(d['t'])+1))
    >>> intercept = np.ones(t_steps.shape)[:, None]
    >>> s_matrix = np.concatenate([intercept, t_steps[:, None], ], axis=1)

    The first step creates an array of times corresponding to one-unit increases in time. The second creates the
    intercept term. Finally, we concatenate the intercept column and the linear term columns together into the design
    matrix for time. Now we can specify and fit the corresponding pooled logistic model. Note that only 3 starting
    values need to be provided, since the design matrix for time has two terms.

    >>> def psi(theta):
    >>>     return ee_plogit(theta=theta, X=d[['statin', ]], delta=d['d'], t=d['t'],
    >>>                               S=s_matrix)

    >>> inits = [0., ] + [-3., 0.]
    >>> estr = MEstimator(stacked_equations=psi, init=inits)
    >>> estr.estimate()

    Weighted models can be estimated by specifying the optional ``weights`` argument. This can be a set of weights
    fixed as baseline, provided as a 1D array (like other regression models). Unlike other regression models,
    ``ee_plogit`` also allows for weights to be time-varying (i.e., a 2D array can be input). Note that the
    dimensions of the 2D array need to correspond to the time intervals of :math:`\mathbb{S}`, so it should be
    `n`-by-`K` dimensions.

    References
    ----------
    Abbott RD. (1985). Logistic regression in survival analysis. *American Journal of Epidemiology*, 121(3), 465-471.

    D'Agostino RB, Lee ML, Belanger AJ, Cupples LA, Anderson K, & Kannel WB. (1990). Relation of pooled logistic
    regression to time dependent Cox regression analysis: the Framingham Heart Study.
    *Statistics in Medicine*, 9(12), 1501-1515.

    Murray EJ, Caniglia EC, & Petito LC. (2021). Causal survival analysis: a guide to estimating intention-to-treat
    and per-protocol effects from randomized clinical trials with non-adherence.
    *Research Methods in Medicine & Health Sciences*, 2(1), 39-49.

    Zivich PN, Cole SR, Shook-Sa BE, DeMonte JB, & Edwards JK. (2025). Estimating equations for survival analysis with
    pooled logistic regression. *arXiv:2504.13291*

    Zivich PN, Klose M, DeMonte JB, Shook-Sa BE, Cole SR, & Edwards JK. (2026). An Improved Pooled Logistic Regression
    Implementation. *Epidemiology*, In-Press.
    """
    # Pre-processing input data
    t = np.asarray(t)                      # Convert to NumPy array
    delta = np.asarray(delta)              # Convert to NumPy array
    X = np.asarray(X)                      # Convert to NumPy array
    xp = X.shape[1]                        # Get shape of X array to divide parameter vector
    beta_x = theta[:xp]                    # Beta parameters for X design matrix
    beta_s = np.asarray(theta[xp:])        # Beta parameters for S design matrix

    # Processing design matrix for time
    if S is None:                                                 # Built disjoint matrix when none is provided
        if unique_times is None:                                  # ... when no user-requested time points
            event_times = t[delta == 1]                           # ... look up event times
            unique_times = np.unique(event_times)                 # ... extract the unique ones
        else:                                                     # ... otherwise
            unique_times = np.asarray(unique_times)               # ... use unique event times provided by the user
        n_time_steps = unique_times.shape[0]                      # ... count up the number of unique event times
        time_design_matrix = np.identity(n=len(unique_times))     # ... create disjoint indicator design matrix
        time_design_matrix[:, 0] = 1                              # ... set first column to be the intercept
    else:                                                         # Otherwise use input design matrix from user
        time_design_matrix = np.asarray(S)                        # ... convert to NumPy array
        unique_times = np.asarray(range(1, int(np.max(t))+1, 1))  # ... unique time points in data
        n_time_steps = len(unique_times)                          # ... number of unique time points
        if n_time_steps != time_design_matrix.shape[0]:           # ... check dimension of input design matrix
            raise ValueError("A total of " + str(unique_times) +
                             " unit-time intervals were created based on the "
                             "input times, but the specific time design matrix has " +
                             str(time_design_matrix.shape[0]) +
                             " rows. These values are expected to match")

    # Log-odds contributions for covariate and time
    log_odds_w = np.dot(X, beta_x)                                   # Covariate contributions to the log odds
    log_odds_t = np.dot(time_design_matrix, beta_s)                  # Time-specific contributions to the log odds

    # Computing residuals
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))       # Stacked copies of X contributions for intervals
    y_obs = delta * (t == unique_times[:, None]).astype(int)         # Event indicator at time intervals matrix
    y_pred = inverse_logit(log_odds_w_matrix + log_odds_t[:, None])  # Predicted event at time intervals matrix
    in_risk_set = (t >= unique_times[:, None]).astype(int)           # Indicator if individual is in the risk set at k
    residual_matrix = (y_obs - y_pred) * in_risk_set                 # Computing residuals at time intervals matrix

    # Incorporating specified weights
    weights = generate_weights(weights, n_obs=t.shape[0])            # Pre-processing weight argument
    if weights.ndim == 2:
        if weights.shape[1] != n_time_steps or weights.shape[0] != t.shape[0]:
            raise ValueError("If a 2D weight matrix is provided, it must (1) have the same number of rows as "
                             "observations, and (2) match the number of time points. A total of "
                             + str(t.shape[0]) + " observations were provided with "
                             + str(n_time_steps) + " time intervals, but the weight "
                             "matrix was " + str(weights.shape))
        weights = weights.T
    residual_matrix = residual_matrix * weights     # Multiplying residuals by weights prior to sum

    # Getting score matrix for X
    n_ones = np.ones(shape=(1, n_time_steps))       # Vector of ones to ease cumulative sum across time intervals
    y_resid = np.dot(n_ones, residual_matrix)[0]    # Adding together residual contributions across all time intervals
    x_score = y_resid[:, None] * X                  # Compute the score for the current interval for X

    # Getting score matrix for S
    if S is None:                                   # If using disjoint indicators for time
        t_score = residual_matrix                   # ... simply return the residual_matrix
    else:                                           # Otherwise
        t_score = np.dot(S.T, residual_matrix)      # ... matrix multiplication of time design matrix with residuals

    # Returning the overall score function matrix stacked together
    return np.vstack([x_score.T, t_score])
