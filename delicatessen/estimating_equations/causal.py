#####################################################################################################################
# Estimating functions for causal inference applications
#####################################################################################################################

import numpy as np

from .regression import ee_regression, ee_glm
from delicatessen.utilities import logit, inverse_logit, identity


#################################################################
# Causal Inference (ATE) Estimating Equations


def ee_gformula(theta, y, X, X1, X0=None, force_continuous=False):
    r"""Estimating equations for the g-formula (or g-computation). The parameter of interest can either be the mean
    under a single policy or plan of action, or the mean difference between two policies. This is accomplished by
    providing the estimating equation the observed data (``X``, ``y``), and the same data under the actions (``X1``
    and optionally ``X0``).

    The stack of estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            \left\{ g({X_i^*}^T \beta) - \theta_1 \right\} \\
            \left\{ Y_i - g(X_i^T \beta) \right\} X_i
        \end{bmatrix}
        = 0

    where the first is the mean under the specified plan, with the plan setting the values of action :math:`A` (e.g.,
    exposure, treatment, vaccination, etc.), and the second equation is the outcome regression model.
    Here, :math:`g` indicates a transformation function. For linear regression, :math:`g` is the identity function.
    Logistic regression uses the inverse-logit function. By default, ``ee_gformula`` detects whether `y` is all binary
    (zero or one), and applies logistic regression if that is evaluated to be true.

    Note
    ----
    This variation includes 1+`b` parameters, where the first parameter is the causal mean, and the remainder are
    the parameters for the regression model.


    Alternatively, a causal mean difference is estimated when ``X0`` is specified. A common example of this would be
    the average causal effect, where the plans are all-action-one versus all-action-zero. Therefore, the estimating
    equations consist of the following three equations

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            (\theta_1 - \theta_2) - \theta_0 \\
            g({X_i^1}^T \beta) - \theta_1 \\
            g({X_i^0}^T \beta) - \theta_2 \\
            \left\{ Y_i - g(X_i^T \beta) \right\} X_i
        \end{bmatrix}
        = 0

    Note
    ----
    This variation includes 3+`b` parameters, where the first parameter is the causal mean difference, the second
    is the causal mean under plan 1, the third is the causal mean under plan 0, and the remainder are the parameters
    for the regression model.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 1+`b` values if ``X0`` is ``None``, and 3+`b` values if ``X0`` is not ``None``.
    y : ndarray, list, vector
        1-dimensional vector of `n` observed values.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables.
    X1 : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables under the action plan.
    X0 : ndarray, list, vector, None, optional
        2-dimensional vector of `n` observed values for `b` variables under the separate action plan. This second
        argument is optional and should be specified if the causal mean difference between two action plans is of
        interest.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (1+`b`)-by-`n` NumPy array if ``X0=None``, or returns a (3+`b`)-by-`n` NumPy array if ``X0!=None``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_gformula`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_gformula

    Some generic confounded data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    In the first example, we will estimate the causal mean had everyone been set to ``A=1``. Therefore, the optional
    argument ``X0`` is left as ``None``. Before creating the estimating equation, we need to do some data prep. First,
    we will create an interaction term between ``A`` and ``W`` in the original data. Then we will generate a copy of
    the data and update the values of ``A`` to be all ``1``.

    >>> d['AW'] = d['A']*d['W']
    >>> d1 = d.copy()
    >>> d1['A'] = 1
    >>> d1['AW'] = d1['A']*d1['W']

    Having setup our data, we can now define the psi function.

    >>> def psi(theta):
    >>>     return ee_gformula(theta,
    >>>                        y=d['Y'],
    >>>                        X=d[['C', 'A', 'W', 'AW']],
    >>>                        X1=d1[['C', 'A', 'W', 'AW']])

    Notice that ``y`` corresponds to the observed outcomes, ``X`` corresponds to the observed covariate data, and ``X1``
    corresponds to the covariate data *under the action plan*.

    Now we can call the M-Estimator. Since we are estimating the causal mean, and the regression parameters,
    the length of the initial values needs to correspond with this. Our linear regression model consists of 4
    coefficients, so we need 1+4=5 initial values. When the outcome is binary (like it is in this example), we can be
    nice to the optimizer and give it a starting value of 0.5 for the causal mean (since 0.5 is in the middle of that
    distribution).

    >>> estr = MEstimator(psi, init=[0.5, 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    The causal mean is

    >>> estr.theta[0]

    Continuing from the previous example, let's say we wanted to estimate the average causal effect. Therefore, we want
    to contrast two plans (all ``A=1`` versus all ``A=0``). As before, we need to create the reference data for ``X0``

    >>> d0 = d.copy()
    >>> d0['A'] = 0
    >>> d0['AW'] = d0['A']*d0['W']

    Having setup our data, we can now define the psi function.

    >>> def psi(theta):
    >>>     return ee_gformula(theta,
    >>>                        y=d['Y'],
    >>>                        X=d[['C', 'A', 'W', 'AW']],
    >>>                        X1=d1[['C', 'A', 'W', 'AW']],
    >>>                        X0=d0[['C', 'A', 'W', 'AW']], )

    Notice that ``y`` corresponds to the observed outcomes, ``X`` corresponds to the observed covariate data, ``X1``
    corresponds to the covariate data under ``A=1``, and ``X0`` corresponds to the covariate data under ``A=0``. Here,
    we need 3+4=7 starting values, since there are two additional parameters from the previous example. For the
    difference, a starting value of 0 is generally a good choice. Since ``Y`` is binary, we again provide 0.5 as
    starting values for the causal means

    >>> estr = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates

    >>> estr.theta[0]    # causal mean difference of 1 versus 0
    >>> estr.theta[1]    # causal mean under X1
    >>> estr.theta[2]    # causal mean under X0
    >>> estr.theta[3:]   # logistic regression coefficients

    References
    ----------
    Snowden JM, Rose S, & Mortimer KM. (2011). Implementation of G-computation on a simulated data set: demonstration
    of a causal inference technique. *American Journal of Epidemiology*, 173(7), 731-738.

    Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
    *Journal of Epidemiology & Community Health*, 60(7), 578-586.
    """
    # Ensuring correct typing
    X = np.asarray(X)                # Convert to NumPy array
    y = np.asarray(y)                # Convert to NumPy array
    X1 = np.asarray(X1)              # Convert to NumPy array

    # Error checking for misaligned shapes
    if X.shape != X1.shape:
        raise ValueError("The dimensions of X and X1 must be the same.")

    # Processing data depending on if two plans were specified
    if X0 is None:                   # If no reference was specified
        mu1 = theta[0]                  # ... only a single mean
        beta = theta[1:]                # ... immediately followed by the regression parameters
    else:                            # Otherwise difference and both plans are to be returned
        X0 = np.asarray(X0)             # ... reference data to NumPy array
        if X.shape != X0.shape:         # ... error checking for misaligned shapes
            raise ValueError("The dimensions of X and X0 must be the same.")
        mud = theta[0]                  # ... first parameter is mean difference
        mu1 = theta[1]                  # ... second parameter is mean under X1
        mu0 = theta[2]                  # ... third parameter is mean under X0
        beta = theta[3:]                # ... remainder are for the regression model

    # Checking outcome variable type
    if np.isin(y, [0, 1]).all() and not force_continuous:
        model = 'logistic'                          # Use a logistic regression model
        transform = inverse_logit                   # ... and need to inverse-logit transformation
    else:
        model = 'linear'                            # Use a linear regression model
        transform = identity                        # ... and need to apply the identity (no) transformation

    # Estimating regression parameters
    preds_reg = ee_regression(theta=beta,              # beta coefficients
                              X=X, y=y,                # ... along with observed X and observed y
                              model=model)             # ... and specified model type

    # Calculating mean under X1
    ya1 = transform(np.dot(X1, beta)) - mu1         # mean under X1

    if X0 is None:                                  # if no X0, then nothing left to do
        # Output (1+b)-by-n stacked array
        return np.vstack((ya1[None, :],     # theta[0] is the mean under X1
                          preds_reg))       # theta[1:] is the regression coefficients
    else:                                           # if X0, then need to predict mean under X0 and difference
        # Calculating mean under X0
        ya0 = transform(np.dot(X0, beta)) - mu0
        # Calculating mean difference between X1 and X0
        ace = np.ones(y.shape[0])*(mu1 - mu0) - mud
        # Output (3+b)-by-n stacked array
        return np.vstack((ace,            # theta[0] is the mean difference between X1 and X0
                          ya1[None, :],   # theta[1] is the mean under X1
                          ya0[None, :],   # theta[2] is the mean under X0
                          preds_reg))     # theta[3:] is for the regression coefficients


def ee_ipw(theta, y, A, W, truncate=None, weights=None):
    r"""Estimating equation for inverse probability weighting (IPW) estimator. The average causal effect is estimated by
    this implementation of the IPW estimator. For estimation of the propensity scores, a logistic model is used.

    The stacked estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            (\theta_1 - \theta_2) - \theta_0 \\
            \frac{A_i Y_i}{\pi_i} - \theta_1 - \theta_1 \\
            \frac{(1-A_i) Y_i}{1-\pi_i} - \theta_2 \\
            \left\{ A_i - \text{expit}(W_i^T \alpha) \right\} W_i
        \end{bmatrix}
        = 0

    where :math:`A` is the action, math:`W` is the set of confounders, and :math:`\pi_i = expit(W_i^T \alpha)`. The
    first estimating equation is for the average causal effect, the second is for the mean under :math:`A:=1`,
    the third is for the mean under :math:`A:=0`, and the last is the logistic regression model for the propensity
    scores. Here, the length of the theta vector is 3+`b`, where `b` is the number of parameters in the regression
    model.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 3+`b` values.
    y : ndarray, list, vector
        1-dimensional vector of `n` observed values.
    A : ndarray, list, vector
        1-dimensional vector of `n` observed values. The A values should all be 0 or 1.
    W : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables to model the probability of ``A`` with.
    truncate : None, list, set, ndarray, optional
        Bounds to truncate the estimated probabilities of ``A`` at. For example, estimated probabilities above 0.99 or
        below 0.01 can be set to 0.99 or 0.01, respectively. This is done by specifying ``truncate=(0.01, 0.99)``. Note
        this step is done via ``numpy.clip(.., a_min, a_max)``, so order is important. Default
        is ``None``, which applies no truncation.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ``None``, which assigns a weight of 1 to all observations. This
        argument is intended to support the use of missingness weights. The propensity score model is *not* fit using
        these weights.

    Returns
    -------
    array :
        Returns a (3+`b`)-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_ipw`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_ipw

    Some generic data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations. Note that ``'A'`` is the action.

    >>> def psi(theta):
    >>>     return ee_ipw(theta, y=d['Y'], A=d['A'],
    >>>                   W=d[['C', 'W']])

    Calling the M-estimation procedure. Since ``W`` is 2-by-n here and IPW has 3 additional parameters, the initial
    values should be of length 3+2=5. In general, it will be best to start with [0., 0.5, 0.5, ...] as the initials when
    ``Y`` is binary. Otherwise, starting with all 0. as initials is reasonable.

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the corresponding parameters are

    >>> estr.theta[0]    # causal mean difference of 1 versus 0
    >>> estr.theta[1]    # causal mean under A=1
    >>> estr.theta[2]    # causal mean under A=0
    >>> estr.theta[3:]   # logistic regression coefficients

    If you want to see how truncating the probabilities works, try repeating the above code but specifying
    ``truncate=(0.1, 0.9)`` as an optional argument in ``ee_ipw``.

    References
    ----------
    Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
    *Journal of Epidemiology & Community Health*, 60(7), 578-586.

    Cole SR, & Hernán MA. (2008). Constructing inverse probability weights for marginal structural models.
    *American Journal of Epidemiology*, 168(6), 656-664.
    """
    # Ensuring correct typing
    W = np.asarray(W)                            # Convert to NumPy array
    A = np.asarray(A)                            # Convert to NumPy array
    y = np.asarray(y)                            # Convert to NumPy array
    beta = theta[3:]                             # Extracting out theta's for the regression model

    # Estimating propensity score
    preds_reg = ee_regression(theta=beta,        # Using logistic regression
                              X=W,               # ... plug-in covariates for X
                              y=A,               # ... plug-in treatment for Y
                              model='logistic')  # ... use a logistic model

    # Estimating weights
    pi = inverse_logit(np.dot(W, beta))          # Getting Pr(A|W) from model
    if truncate is not None:                     # Truncating Pr(A|W) when requested
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    # Processing external weights argument
    if weights is None:
        weights = 1

    # Calculating Y(a=1)
    ya1 = (A * y) / pi * weights - theta[1]                # i's contribution is (AY) / \pi
    # Calculating Y(a=0)
    ya0 = ((1-A) * y) / (1-pi) * weights - theta[2]        # i's contribution is ((1-A)Y) / (1-\pi)
    # Calculating Y(a=1) - Y(a=0)
    ate = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

    # Output (3+b)-by-n stacked array
    return np.vstack((ate,             # theta[0] is for the ATE
                      ya1[None, :],    # theta[1] is for R1
                      ya0[None, :],    # theta[2] is for R0
                      preds_reg))      # theta[3:] is for the regression coefficients


def ee_ipw_msm(theta, y, A, W, V, distribution, link, hyperparameter=None, truncate=None, weights=None):
    r"""Estimating equation for parameters of a marginal structural model estimated using inverse probability weighting.
    For estimation of the propensity scores, a logistic model is used.

    The stacked estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            \frac{1}{\pi_i} \left\{ Y_i - g^{-1}(X_i^T \beta) \right\} \times \frac{D(\beta)}{v(\beta)} X_i \\
            \left\{ A_i - \text{expit}(W_i^T \alpha) \right\} W_i
        \end{bmatrix}
        = 0

    where :math:`A` is the action, math:`W` is the set of confounders, and :math:`\pi_i = \text{expit}(W_i^T \alpha)`.
    Here, :math:`X` is the design matrix for the marginal structural model (it includes :math:`A`, and possibly some
    covariates from :math:`W`). The first estimating equation is a weighted generalized linear model is used. See
    ``ee_glm`` for details on this estimating equation. The second estimating equation is the logistic model for the
    propensity scores.

    Here, ``theta`` corresponds to multiple quantities. The *first* set of values correspond to the parameters of the
    marginal structural model, and the *second* set correspond to the logistic regression model coefficients for the
    propensity scores.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of `c`+`b` values.
    y : ndarray, list, vector
        1-dimensional vector of `n` observed values.
    A : ndarray, list, vector
        1-dimensional vector of `n` observed values. The A values should all be 0 or 1.
    W : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables to model the probability of ``A`` with.
    V : ndarray, list, vector
        2-dimensional vector of `n` observed values for `c` variables in the marginal structural model.
    distribution : str
        Distribution for the generalized linear model. See ``ee_glm`` for options.
    link : str
        Link function for the generalized linear model. See ``ee_glm`` for options.
    truncate : None, list, set, ndarray, optional
        Bounds to truncate the estimated probabilities of ``A`` at. For example, estimated probabilities above 0.99 or
        below 0.01 can be set to 0.99 or 0.01, respectively. This is done by specifying ``truncate=(0.01, 0.99)``. Note
        this step is done via ``numpy.clip(.., a_min, a_max)``, so order is important. Default
        is ``None``, which applies no truncation.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is None, which assigns a weight of 1 to all observations. This
        argument is intended to support the use of missingness weights. The propensity score model is *not* fit using
        these weights.

    Returns
    -------
    array :
        Returns a (`c`+`b`)-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_ipw_msm`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_ipw_msm

    Some generic data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations for a logistic marginal structural model. Note that 'A' is the
    action.

    >>> def psi(theta):
    >>>     return ee_ipw_msm(theta, y=d['Y'], A=d['A'],
    >>>                       W=d[['C', 'W']], V=d[['C', 'A']],
    >>>                       link='logit', distribution='binomial')

    Calling the M-estimation procedure. Since ``W`` is 2-by-n here and the marginal structural model (``V``) consists
    of 2 parameters, the initial values should be of length 2+2=4. Starting values for the marginal structural model
    may need to be adjusted to lie within the domain of the chosen link-distribution functions.

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the corresponding parameters are

    >>> estr.theta[0:2]  # Marginal structural model paramters
    >>> estr.theta[2:]   # Nuisance model parameters

    If you want to see how truncating the probabilities works, try repeating the above code but specifying
    ``truncate=(0.1, 0.9)`` as an optional argument in ``ee_ipw_msm``.

    References
    ----------
    Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
    *Journal of Epidemiology & Community Health*, 60(7), 578-586.

    Cole SR, & Hernán MA. (2008). Constructing inverse probability weights for marginal structural models.
    *American Journal of Epidemiology*, 168(6), 656-664.
    """
    # Ensuring correct typing
    W = np.asarray(W)                            # Convert to NumPy array
    V = np.asarray(V)                            # Convert to NumPy array
    A = np.asarray(A)                            # Convert to NumPy array
    y = np.asarray(y)                            # Convert to NumPy array
    alpha = theta[:V.shape[1]]                   # Extracting out theta's for the MSM
    beta = theta[V.shape[1]:]                    # Extracting out theta's for the PS model

    # Estimating propensity score model
    preds_reg = ee_regression(theta=beta,        # Using logistic regression
                              X=W,               # ... plug-in covariates for X
                              y=A,               # ... plug-in treatment for Y
                              model='logistic',  # ... use a logistic model
                              weights=None)      # ... and no weights here

    # Estimating weights
    pi = inverse_logit(np.dot(W, beta))          # Getting Pr(A|W) from model
    if truncate is not None:                     # Truncating Pr(A|W) when requested
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    # Processing external weights argument
    ipw = np.where(A == 1, 1/pi, 1/(1-pi))
    if weights is not None:
        ipw = ipw * weights

    # Estimating the marginal structural model
    ee_msm = ee_glm(theta=alpha, X=V, y=y,                  # Regressing specified MSM
                    distribution=distribution, link=link,   # ... with given link, distribution
                    hyperparameter=hyperparameter,          # ... and GLM hyperparameter
                    weights=ipw, offset=None)               # ... weighted by the IPW

    # Output (c+b)-by-n stacked array
    return np.vstack((ee_msm,          # theta[:c] is the marginal structural model parameters
                      preds_reg))      # theta[c:] is for the regression coefficients


def ee_aipw(theta, y, A, W, X, X1, X0, truncate=None, force_continuous=False):
    r"""Estimating equation for augmented inverse probability weighting (AIPW) estimator. AIPW consists of two nuisance
    models (the propensity score model and the outcome model).

    The stacked estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            (\theta_1 - \theta_2) - \theta_0 \\
            \frac{A_i Y_i}{\pi_i} - \frac{\hat{Y^1}(A_i-\pi_i}{\pi_i} - \theta_1 \\
            \frac{(1-A_i) Y_i}{1-\pi_i} + \frac{\hat{Y^0}(A_i-\pi_i}{1-\pi_i} - \theta_2 \\
            \left\{ A_i - \text{expit}(W_i^T \alpha) \right\} W_i \\
            \left\{ Y_i - g(X_i^T \beta) \right\} X_i
        \end{bmatrix}
        = 0

    where :math:`A` is the action and :math:`W` is the set of confounders, :math:`Y` is the outcome, and
    :math:`\pi_i = \text{expit}(W_i^T \alpha)`. The first estimating equation is for the average causal effect, the
    second is for the mean under :math:`A:=1`, the third is for the mean under :math:`A:=0`, the fourth is the logistic
    regression model for the propensity scores, and the last is for the outcome model. Here, the length of the theta
    vector is 3+`b`+`c`, where `b` is the number of parameters in the propensity score model and `c` is the number
    of parameters in the outcome model.

    By default, `ee_aipw` detects whether `y` is all binary (zero or one), and applies logistic regression. Notice that
    ``X`` here should consists of both ``A`` and ``W`` (with possible interaction terms or other differences in
    functional forms from the propensity score model).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 3+`b`+`c` values.
    y : ndarray, list, vector
        1-dimensional vector of `n` observed values.
    A : ndarray, list, vector
        1-dimensional vector of `n` observed values. The A values should all be 0 or 1.
    W : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables to model the probability of ``A`` with.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `c` variables to model the outcome ``y``.
    X1 : ndarray, list, vector
        2-dimensional vector of `n` observed values for `c` variables under the action plan where ``A=1`` for all units.
    X0 : ndarray, list, vector, None, optional
        2-dimensional vector of `n` observed values for `c` variables under the action plan where ``A=0`` for all units.
    truncate : None, list, set, ndarray, optional
        Bounds to truncate the estimated probabilities of ``A`` at. For example, estimated probabilities above 0.99 or
        below 0.01 can be set to 0.99 or 0.01, respectively. This is done by specifying ``truncate=(0.01, 0.99)``. Note
        this step is done via ``numpy.clip(.., a_min, a_max)``, so order is important. Default
        is ``None``, which applies to no truncation.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (3+`b`+`c`)-by-`n` NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_aipw`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_aipw

    Some generic data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations. Note that ``A`` is the action of interest. First, we will apply
    some necessary data processing.  We will create an interaction term between ``A`` and ``W`` in the original data.
    Then we will generate a copy of the data and update the values of ``A=1``, and then generate another
    copy but set ``A=0`` in that copy.

    >>> d['AW'] = d['A']*d['W']
    >>> d1 = d.copy()   # Copy where all A=1
    >>> d1['A'] = 1
    >>> d1['AW'] = d1['A']*d1['W']
    >>> d0 = d.copy()   # Copy where all A=0
    >>> d0['A'] = 0
    >>> d0['AW'] = d0['A']*d0['W']

    Having setup our data, we can now define the psi function.

    >>> def psi(theta):
    >>>     return ee_aipw(theta,
    >>>                    y=d['Y'],
    >>>                    A=d['A'],
    >>>                    W=d[['C', 'W']],
    >>>                    X=d[['C', 'A', 'W', 'AW']],
    >>>                    X1=d1[['C', 'A', 'W', 'AW']],
    >>>                    X0=d0[['C', 'A', 'W', 'AW']])

    Calling the M-estimator. AIPW has 3 parameters with 2 coefficients in the propensity score model, and
    4 coefficients in the outcome model, the total number of initial values should be 3+2+4=9. When Y is binary, it
    will be best to start with ``[0., 0.5, 0.5, ...]`` followed by all ``0.`` for the initial values. Otherwise,
    starting with all 0. as initials is reasonable.

    >>> estr = MEstimator(psi,
    >>>                   init=[0., 0.5, 0.5, 0., 0., 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the corresponding parameters are

    >>> estr.theta[0]     # causal mean difference of 1 versus 0
    >>> estr.theta[1]     # causal mean under A=1
    >>> estr.theta[2]     # causal mean under A=0
    >>> estr.theta[3:5]   # propensity score regression coefficients
    >>> estr.theta[5:]    # outcome regression coefficients

    References
    ----------
    Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
    *Journal of Epidemiology & Community Health*, 60(7), 578-586.

    Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust estimation of causal
    effects. *American Journal of Epidemiology*, 173(7), 761-767.

    Tsiatis AA. (2006). Semiparametric theory and missing data. Springer, New York, NY.
    """
    # Ensuring correct typing
    y = np.asarray(y)              # Convert to NumPy array
    A = np.asarray(A)              # Convert to NumPy array
    W = np.asarray(W)              # Convert to NumPy array
    X = np.asarray(X)              # Convert to NumPy array
    X1 = np.asarray(X1)            # Convert to NumPy array
    X0 = np.asarray(X0)            # Convert to NumPy array

    # Checking some shapes
    if X.shape != X1.shape:
        raise ValueError("The dimensions of X and X1 must be the same.")
    if X.shape != X0.shape:
        raise ValueError("The dimensions of X and X0 must be the same.")

    # Extracting theta value for ease
    mud = theta[0]                 # Parameter for average causal effect
    mu1 = theta[1]                 # Parameter for the mean under A=1
    mu0 = theta[2]                 # Parameter for the mean under A=0
    alpha = theta[3:3+W.shape[1]]  # Parameter(s) for the propensity score model
    beta = theta[3+W.shape[1]:]    # Parameter(s) for the outcome model

    # pi-model (logistic regression)
    pi_model = ee_regression(theta=alpha,             # Estimating logistic model
                             X=W, y=A,
                             model='logistic')
    pi = inverse_logit(np.dot(W, alpha))              # Estimating Pr(A|W)
    if truncate is not None:                          # Truncating Pr(A|W) when requested
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    # m-model (logistic regression)
    # Checking outcome variable type
    if np.isin(y, [0, 1]).all() and not force_continuous:
        model = 'logistic'                             # Use a logistic regression model
        transform = inverse_logit                      # ... and need to inverse-logit transformation
    else:
        model = 'linear'                               # Use a linear regression model
        transform = identity                           # ... and need to apply the identity (no) transformation

    m_model = ee_regression(theta=beta,                # Estimating the outcome model
                            y=y, X=X,
                            model=model)
    ya1 = transform(np.dot(X1, beta))                  # Generating predicted values under X1
    ya0 = transform(np.dot(X0, beta))                  # Generating predicted values under X0

    # AIPW estimator
    ace = np.ones(y.shape[0]) * (mu1 - mu0) - mud               # Calculating the ATE
    y1_star = (y*A/pi - ya1*(A-pi)/pi) - mu1                    # Calculating \tilde{Y}(a=1)
    y0_star = (y*(1-A)/(1-pi) + ya0*(A-pi)/(1-pi)) - mu0        # Calculating \tilde{Y}(a=0)

    # Output (3+b+c)-by-n array
    return np.vstack((ace,               # theta[0] is for the ATE
                      y1_star[None, :],  # theta[1] is for R1
                      y0_star[None, :],  # theta[2] is for R0
                      pi_model,          # theta[b] is for the treatment model coefficients
                      m_model))          # theta[c] is for the outcome model coefficients


#################################################################
# Causal Inference (SMM) Estimating Equations

def ee_gestimation_snmm(theta, y, A, W, V, X=None, model='linear', weights=None):
    r"""Estimating equations for g-estimation of structural mean models (SMMs). The parameter(s) of interest are the
    parameter(s) of the corresponding SMM. Rather than estimating the average causal effect, g-estimation of SMM
    estimates the average causal effect in the acted on within strata of a set of covariates, :math:`V`. Options for
    SMM include the linear SMM and the log-linear SMM. The linear SMM is defined as

    .. math::

        E[Y^a - Y^{0} | A=a, V] = \beta_1 a + \beta_2 a V

    This model corresponds to the average causal effect among those with :math:`A=a` by :math:`V`. The
    log-linear SMM is defined as

    .. math::

        \frac{E[Y^a | A=a, V]}{E[Y^{0} | A=a, V]} = \exp(\beta_1 a + \beta_2 a V)

    This model corresponds to the causal mean ratio among those with :math:`A=a` by :math:`V`. Note that
    the log-linear SMM is only defined when :math:`Y > 0`. The parameters of either SMM are identified under the
    assumptions of  causal consistency, and exchangeability with positivity.

    Two different estimating equations are available for g-estimation. The first set is referred to at the 'inefficient'
    g-estimator. For the inefficient g-estimator we solve for :math:`\beta` in the following estimating equation

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            \left\{ H(\beta) \times (A - \pi_i) \right\}  \times V_i \\
            \left\{ A_i - \text{expit}(W_i^T \alpha) \right\} W_i
        \end{bmatrix}
        = 0

    where :math:`\pi_i = \text{expit}(W_i^T \alpha)`, and
    :math:`H(\beta) = Y - \beta A \mathbb{V}` for a linear SMM and
    :math:`H(\beta) = Y \times \exp(-A \beta \mathbb{V})` for a log-linear SMM, where .
    Note that :math:`V \subseteq W`, where :math:`W` is the set of confounding variables.
    The length of the parameter vector is `b`+`c`, where `b` is the number of columns in ``V``, and
    `c` is the number of columns in ``W``.

    The second implementation for g-estimation is the 'efficient' g-estimator. For the efficient g-estimator we replace
    :math:`H(\beta)` with :math:`\{H(\beta) - E[H(\beta) | W]\}` in the prior estimating equation and specify a model
    for :math:`E[H(\beta) | W]`. The corresponding stacked estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            \left\{ (H(\beta) - g^{-1}(W_i^T \gamma)) \times (A - \pi_i) \right\}  \times V_i \\
            \left\{ A_i - \text{expit}(W_i^T \alpha) \right\} W_i \\
            \left\{ H(\beta) - g^{-1}(W_i^T \gamma) \right\} W_i \\
        \end{bmatrix}
        = 0

    where :math:`g^{-1}` is the inverse transformation for the specified SMM. Therefore, there are b+c+d parameters
    for the efficient g-estimator, where `d` is the number of parameters in the model for :math:`E[H(\beta) | W]`.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 1+`b` values if ``X0`` is ``None``, and 3+b values if ``X0`` is not ``None``.
    y : ndarray, list, vector
        1-dimensional vector of `n` observed values of the outcome.
    A : ndarray, list, vector
        1-dimensional vector of `n` observed values of the action. The A values should all be 0 or 1.
    W : ndarray, list, vector
        2-dimensional vector of `n` observed values for b columns of a design matrix to model the expected value of
        ``A``.
    V : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` columns of a design matrix for the structural mean model.
        Note that the design matrix here is expected to not include the observed values of ``A``
    X : ndarray, list, vector, None, optional
        Default of this argument is ``None``, which implements the estimating equation for the inefficient g-estimator.
        To use the efficient g-estimator, a 2-dimensional vector of n observed values for `b` columns of a design matrix
        for the :math:`E[H(\beta) | W]` model should be provided here.
    model : str, optional
        Type of structural mean model to fit. Options are currently: ``linear``, ``poisson``. Default is ``linear``.
        The Poisson model specification can be used for positive continuous data, or with binary data in order to
        estimate causal risk ratios.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ``None``, which assigns a weight of 1 to all observations. This
        argument is intended to support the use of sampling or missingness weights.

    Returns
    -------
    array :
        Returns a (`b`+`c`)-by-`n` (inefficient) or (`b`+`c`+`d`)-by-`n` (efficient) NumPy array evaluated for the
        input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_gestimation_snmm`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_gestimation_snmm

    Some generic data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.normal(size=n)
    >>> d['V'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=logistic.cdf(0.25 + 0.5*d['V'] + d['W']), size=n)
    >>> d['Ya0'] = 12.75 - 3.5*d['V'] + d['W'] + np.random.normal(size=n)
    >>> d['Ya1'] = 10.75 - 0.8*d['V'] + d['W'] + np.random.normal(size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations. Note that ``A`` is the action of interest and ``Y`` is the
    outcome of interest. Here, we are interested in estimating the following linear SMM

    .. math::

        E[Y^a - Y^{0} | A=a, V] = \beta_1 a + \beta_2 a V

    >>> def psi(theta):
    >>>     return ee_gestimation_snmm(theta,
    >>>                                y=d['Y'], A=d['A'],
    >>>                                W=d[['C', 'V', 'W']],
    >>>                                V=d[['C', 'V']])

    Calling the M-estimator.  Since there are 2 coefficients in the SMM and 3 coefficients in the :math:`E[A|W]` model,
    the total number of initial values should be 2+3=5:

    >>> estr = MEstimator(psi,
    >>>                   init=[0., ]*5)
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the corresponding parameters are

    >>> estr.theta[0]     # beta_1 of SMM
    >>> estr.theta[1]     # beta_2 of SMM
    >>> estr.theta[2:]    # propensity score regression coefficients

    The efficient g-estimator can be implemented by providing a design matrix to the argument ``X``

    >>> def psi(theta):
    >>>     return ee_gestimation_snmm(theta,
    >>>                                y=d['Y'], A=d['A'],
    >>>                                W=d[['C', 'V', 'W']],
    >>>                                V=d[['C', 'V']],
    >>>                                X=d[['C', 'V', 'W']])

    Here, there are 2+3+3=8 parameters to estimate

    >>> estr = MEstimator(psi,
    >>>                   init=[0., ]*8)
    >>> estr.estimate(solver='lm')

    A log-linear SMM for this example can be estimated by specifying ``model='poisson'``.

    References
    ----------
    Dukes O, & Vansteelandt S (2018). A note on G-estimation of causal risk ratios. *American Journal of Epidemiology*,
    187(5), 1079-1084.

    Robins JM, Mark SD, Newey WK (1992). Estimating exposure effects by modelling the expectation of exposure
    conditional on confounders. *Biometrics*, 48(2), 479–495.

    Vansteelandt S, & Joffe M (2014). Structural nested models and G-estimation: the partially realized promise.
    *Statist Sci*, 29(4), 707-731.

    Vansteelandt S, & Sjolander A (2016). Revisiting g-estimation of the effect of a time-varying exposure subject to
    time-varying confounding. *Epidemiologic Methods*, 5(1), 37-56.
    """
    # Future consideration: add bias adjustment via b(A,W; \alpha) to h_psi from Vancak & Sjolander
    # Ensuring correct typing
    y = np.asarray(y)[:, None]                  # Convert to NumPy array and converting shape
    A = np.asarray(A)                           # Convert to NumPy array
    W = np.asarray(W)                           # Convert to NumPy array
    V = np.asarray(V)                           # Convert to NumPy array
    eq_add = []                                 # Storage for outcome model, default is empty (none)
    pdiv = V.shape[1]                           # Extracting number of SMM parameters
    qdiv = W.shape[1] + pdiv                    # Extracting number of E[A|W] parameters

    # Processing weights argument
    if weights is None:
        weight = 1
    else:
        weight = np.asarray(weights)

    # Extracting theta value for ease
    phi = np.asarray(theta[0: pdiv])[:, None]   # theta parameters for the SMM
    alpha = np.asarray(theta[pdiv:qdiv])        # theta parameters for the E[A|W] model
    if X is not None:                           # If given an input X
        beta = np.asarray(theta[qdiv:])         # ... theta parameters for the E[Y|W] model

    # # Option for the variations on the structural mean model
    if model.lower() == 'linear':                             # Linear structural mean model
        h_phi = y - np.dot(V*A[:, None], phi)                 # ... simply subtract
        y_transform = identity                                # ... transformation for E[h(phi) | X]
    elif model.lower() == 'poisson':                          # Log-linear structural mean model
        h_phi = y * np.exp(-1 * np.dot(V*A[:, None], phi))    # ... multiplication and exp transformation
        y_transform = np.exp                                  # ... transformation for E[h(phi) | X]
    # Add tanh(.) as a function for the risk difference?
    else:                                                     # Error checking
        raise ValueError("model='" + str(model) + "' is not a "
                         "supported option. Only the following "
                         "options are supported: linear, poisson")

    # Estimating the E[A | L] Model
    ee_log = ee_regression(theta=alpha,                              # Propensity score parameters
                           X=W, y=A,                                 # ... treatment and covariate design matrix
                           model='logistic',                         # ... logistic model
                           weights=weights)                          # ... with provided weights
    pi = inverse_logit(np.dot(W, alpha))                             # Converting log-odds to probability
    a_resid = (A - pi)[:, None]                                      # Calculating residuals for A

    # Estimating functions for the corresponding g-estimator of SMM
    if X is not None:                                            # Specifying an outcome model for efficient
        X = np.asarray(X)                                        # ... convert X to NumPy array
        ee_out = ee_regression(theta=beta,                       # ... outcome model with beta
                               X=X, y=h_phi[:, 0],               # ... for E[h(phi)|W]
                               model=model,                      # ... transformation to consider
                               weights=weights)                  # ... using provided weights
        yhat = y_transform(np.dot(X, beta)[:, None])             # ... get predicted h(phi)
        eq_add = [ee_out, ]                                      # ... adding outcome model estimating functions
    else:                                                        # Otherwise uses inefficient g-estimator
        yhat = 0                                                 # ... residual set manually to zero
        # This error should not be reached at this time. It is a placeholder for a potential future addition
        if model.lower() == 'logistic':                          # Error if logistic is requested
            raise ValueError("The g-estimator with X=None "
                             "does not support logistic structural mean models.")

    # Estimating function for the structural mean model
    y0_resid = h_phi - yhat
    ee_smm = weight * (a_resid * y0_resid * V).T

    # Output (b+c)-by-n array
    stacked_ee = [ee_smm,            # SMM parameters
                  ee_log] + eq_add   # Nuisance model parameters
    return np.vstack(stacked_ee)


#################################################################
# Causal Inference (Sensitivity Analysis) Estimating Equations

def ee_mean_sensitivity_analysis(theta, y, delta, X, q_eval, H_function):
    r"""Estimating equation for weighted sensitivity analysis estimator of the mean. This estimator can handle cases of
    missing completely at random, missing at random, and missing not at random. The sensitivity analysis consists of
    two sets of estimating equations. The first is for the mean of interest (:math:`\mu`), and the second is for the
    sensitivity analysis model for the estimable parameters of the missingness model (:math:`\gamma`):

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            \frac{S_i Y_i}{H[\gamma X_i + q(Y_i, \alpha)]} - \mu \\
            \left( \frac{S_i}{H[\gamma X_i + q(Y_i, \alpha)]} - 1 \right) X_i^T
        \end{bmatrix}
        = 0

    where :math:`Y_i` is the outcome of interest with missing data, :math:`X_i` is the corresponding design matrix, and
    :math:`H(b)` is a known, continuous, and monotone increasing distribution function that is bound by :math:`[0,1]`.
    Here, :math:`q(Y_i, \alpha)` is a user-specified sensitivity function. For example,
    :math:`q(Y_i, \alpha) = \alpha Y_i` Importantly, :math:`\alpha` is treated as known (i.e., this approach is not
    possible when :math:`\alpha` needs to be estimated).

    Note
    ----
    This estimator looks like the inverse probability weighting estimator, but the estimating equation for the mean
    is slightly different. When :math:`q(Y, \alpha) = 0`, the estimates between this estimator and the inverse
    probability weighting estimator will result in different (but similar) estimates.


    The length of the parameter vector, :math:`\theta`, is 1+`b`, where `b` is the number of columns in ``X``.
    The *first* value in the theta vector is the corrected mean of :math:`Y`. The remainder of the parameters
    correspond to the regression model coefficients.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of 1+`b` values. Therefore, initial values should consist of one plus the number of
        columns present in ``X``. This can easily be accomplished generally by ``[0, ] + [0, ] * X.shape[1]``.
    y : ndarray, list, vector
        1-dimensional vector of `n` values. Any values of ``y`` that are missing should be indicated by the ``delta``
        parameter.
    delta : ndarray, list, vector
        1-dimensional vector of `n` observed values indicating whether the observation has a value for ``y`` observed,
        where 1 indicates yes and 0 indicated no. This vector should not include any ``nan`` values.
    X : ndarray, list, vector
        2-dimensional vector of `n` observed values for `b` variables consider as predictors. At a minimum, a vector
        of ones (intercept) should be included. This matrix cannot include any ``nan`` values.
    q_eval : ndarray, list, vector
        1-dimensional vector of `n` values evaluated using the :math:`q(Y; \alpha)` function.
    H_function : callable
        Function use to bound the observations between :math:`[0,1]`. The function must be monotonic increasing and be
        bounded by :math:`[0,1]`. For example, the expit (``delicatessen.utilities.inverse_logit``) function meets
        this criteria.

    Returns
    -------
    array :
        Returns a (1+`b`)-by-`n` NumPy array evaluated for the input ``theta``.

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_mean_sensitivity_analysis`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean_sensitivity_analysis
    >>> from delicatessen.utilities import inverse_logit

    Some generic data with missing values for :math:`Y`

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['Y'] = 200. - 35*d['W'] + np.random.normal(scale=5, size=n)
    >>> d['M'] = np.random.binomial(1, p=inverse_logit(2 + d['W'] - 0.01 * d['Y']), size=n)
    >>> d['Y'] = np.where(d['M'] == 0, np.nan, d['Y'])
    >>> d['C'] = 1

    To apply the sensitivity analysis, we need to specify the corresponding sensitivity analysis function. The following
    is a simple model where missingness depends on a single parameter and the (possibly unobserved) outcome value. Note
    that the function replaces missing observations with zero.

    >>> def q_function(y_vals, alpha):
    >>>     # q(Y_i; \alpha) = alpha * Y_i
    >>>     y_no_miss = np.where(np.isnan(y_vals), 0, y_vals)
    >>>     return alpha * y_no_miss

    We can now define psi, or the stacked estimating equations.

    >>> def psi(theta):
    >>>     yhat = q_function(d['Y'], alpha=-0.01)
    >>>     return ee_mean_sensitivity_analysis(theta=theta,
    >>>                                         y=d['Y'], delta=d['M'], X=d[['C', 'W']],
    >>>                                         q_eval=yhat, H_function=inverse_logit)

    Calling the M-estimator.

    >>> estr = MEstimator(psi, init=[200., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    More specifically, the corresponding parameters are

    >>> estr.theta[0]     # mean of interest
    >>> estr.theta[1:]    # weighting model parameters given alpha

    Proportions (binary outcomes) are also natively supported

    >>> y_bin = np.where(d['Y'] <= 200., 1, 0)    # Binary conversion of outcome
    >>> def psi(theta):
    >>>     yhat = q_function(d['Y'], alpha=-0.1)
    >>>     return ee_mean_sensitivity_analysis(theta=theta,
    >>>                                         y=y_bin, delta=d['M'], X=d[['C', 'W']],
    >>>                                         q_eval=yhat, H_function=inverse_logit)

    >>> estr = MEstimator(psi, init=[0.5, 0., 0.])
    >>> estr.estimate(solver='lm')

    Often, we will want to conduct the sensitivity analysis for a range of different values of alpha. The following is
    code to accomplish this.

    >>> def psi(theta):
    >>>     yhat = q_function(d['Y'], alpha=alpha_current)
    >>>     return ee_mean_sensitivity_analysis(theta=theta,
    >>>                                         y=d['Y'], delta=d['M'], X=d[['C', 'W']],
    >>>                                         q_eval=yhat, H_function=inverse_logit)

    >>> alphas = np.linspace(0, 0.5, 40)
    >>> est, lcl, ucl = [], [], []
    >>> prev_optim = [200., 0., 0.]
    >>> for alpha_current in alphas:
    >>>     mest = MEstimator(psi, init=prev_optim)
    >>>     mest.estimate(solver='lm')
    >>>     prev_optim = mest.theta
    >>>     est.append(mest.theta[0])
    >>>     ci = mest.confidence_intervals()
    >>>     lcl.append(ci[0][0])
    >>>     ucl.append(ci[0][1])

    >>> # plotting
    >>> plt.fill_between(alphas, lcl, ucl, color='blue', alpha=0.2)
    >>> plt.plot(alphas, est, '-', color='blue')
    >>> plt.show()

    Note
    ----
    Note that we use the previous iteration as the starting values for the next alpha as a computational trick to
    speed up the root-finding process and prevent convergence issues.


    The corresponding plot provides a visualization of how the estimated mean changes as :math:`\alpha` changes. This
    can be useful to help judge the extent of bias for the mean due to data missing not at random for a specific model.

    References
    ----------
    Cole SR, Zivich PN, Edwards JK, Shook-Sa BE, & Hudgens MG. (2023). Sensitivity Analyses for Means or Proportions
    with Missing Outcome Data. *Epidemiology*

    Robins JM, Rotnitzky A, & Scharfstein DO. (2000). Sensitivity analysis for selection bias and unmeasured
    confounding in missing data and causal inference models. In
    *Statistical models in epidemiology, the environment, and clinical trials* (pp. 1-94). New York, NY:
    Springer New York.
    """
    delta = np.asarray(delta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra
    y = np.asarray(y)[:, None]               # Convert to NumPy array and ensure correct shape for matrix algebra
    X = np.asarray(X)                        # Convert to NumPy array
    qy = np.asarray(q_eval)[:, None]         # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta[1:])[:, None]    # Convert to NumPy array and ensure correct shape for matrix algebra

    # Predicted values from design matrix and nuisance coefficients
    pred_values = np.dot(X, beta)                                # dot product for speed (like regression)

    # Solving for the sensitivity analysis mean
    numerator = delta * y                                        # Numerator for the mean estimating equation
    denominator = H_function(pred_values + qy)                   # Denominator for the mean estimating equation
    ym_ind = np.where(delta == 1, numerator / denominator, 0)    # Setting missing Y as zero (don't contribute to EE)
    ef_mean = ym_ind - theta[0]                                  # Sensitivity analysis estimating equation

    # Solving for intercept and coefficients of model
    ef_H = (delta / H_function(pred_values + qy) - 1) * X        # Multiply by X at end to keep dims the same

    # Returning stacked estimating equations
    return np.vstack((ef_mean.T,                                 # theta[0] is the sensitivity analysis mean
                      ef_H.T))                                   # theta[1:] is (are) the nuisance parameter(s)
