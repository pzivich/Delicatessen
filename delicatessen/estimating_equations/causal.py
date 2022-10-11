import numpy as np

from .regression import ee_regression
from delicatessen.utilities import logit, inverse_logit, identity


#################################################################
# Causal Inference (ATE) Estimating Equations


def ee_gformula(theta, y, X, X1, X0=None, force_continuous=False):
    r"""Estimating equations for the g-computation. The parameter of interest can either be the mean under a single
    policy or plan of action, or the mean difference between two policies. This is accomplished by providing the
    estimating equation the observed data (``X``, ``y``), and the same data under the actions (``X1`` and optionally
    ``X0``).

    The outcome regression estimating equation is

    .. math::

        \sum_{i=1}^n (Y_i - g(X_i^T \beta)) X_i = 0

    where :math:`g` indicates a transformation function. For linear regression, :math:`g` is the identity function.
    Logistic regression uses the inverse-logit function. By default, `ee_gformula` detects whether `y` is all binary
    (zero or one), and applies logistic regression if that is evaluated to be true.

    There are two variations on the parameter of interest. The first could be the mean under a plan, where the plan sets
    the values of action :math:`A` (e.g., exposure, treatment, vaccination, etc.). The estimating equation for this
    causal mean is

    .. math::

        \sum_{i=1}^n g({X_i^*}^T \beta) - \theta_1 = 0

    Note
    ----
    This variation includes :math:`1+b` parameters, where the first parameter is the causal mean, and the remainder are
    the parameters for the regression model.

    The alternative parameter of interest could be the mean difference between two plans. A common example of this would
    be the average causal effect, where the plans are all-action-one versus all-action-zero. Therefore, the estimating
    equations consist of the following three equations

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            (\theta_1 - \theta_2) - \theta_0
            g({X_i^1}^T \beta) - \theta_1
            g({X_i^0}^T \beta) - \theta_2
        \end{bmatrix}
        = 0

    Note
    ----
    This variation includes :math:`3+b` parameters, where the first parameter is the causal mean difference, the second
    is the causal mean under plan 1, the third is the causal mean under plan 0, and the remainder are the parameters
    for the regression model.


    The parameter of interest is designated by the user via whether the optional argument ``X0`` is left as ``None``
    (which estimates the causal mean) or is given an array (which estimates the causal mean difference and the
    corresponding causal means).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 1+b values if ``X0`` is ``None``, and 3+b values if ``X0`` is not ``None``.
    y : ndarray, list, vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    X1 : ndarray, list, vector
        2-dimensional vector of n observed values for b variables under the action plan.
    X0 : ndarray, list, vector, None, optional
        2-dimensional vector of n observed values for b variables under the separate action plan. This second argument
        is optional and should be specified if the causal mean difference between two action plans is of interest.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (1+b)-by-n NumPy array if ``X0=None``, or returns a (3+b)-by-n NumPy array if ``X0!=None``

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


def ee_ipw(theta, y, A, W, truncate=None):
    r"""Estimating equation for inverse probability weighting estimator. For estimation of the weights (or propensity
    scores), a logistic model is used. The first estimating equations for the logistic regression model are

    .. math::

        \sum_{i=1}^n (A_i - expit(W_i^T \alpha)) W_i = 0

    where A is the action and W is the set of confounders.

    For the implementation of the inverse probability weighting estimator, stacked estimating equations are used
    for the mean had everyone been set to ``A=1``, the mean had everyone been set to ``A=0``, and the mean difference
    between the two causal means. The estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            (\theta_1 - \theta_2) - \theta_0
            \frac{A_i \times Y_i}{\pi_i} - \theta_1 - \theta_1
            \frac{(1-A_i) \times Y_i}{1-\pi_i} - \theta_2
        \end{bmatrix}
        = 0

    where :math:`\pi_i = expit(W_i^T \alpha)`. Due to these 3 extra values, the length of the theta vector is 3+b,
    where b is the number of parameters in the regression model.

    Note
    ----
    Unlike ``ee_gformula``, ``ee_ipw`` always provides the average causal effect, and causal means for ``A=1`` and
    ``A=0``.


    Here, theta corresponds to a variety of different quantities. The *first* value in theta vector is the causal mean
    difference, the *second* is the mean had everyone been set to ``A=1``, the *third* is the mean had everyone been
    set to ``A=0``. The remainder of the parameters correspond to the logistic regression model coefficients.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 3+b values.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    A : ndarray, list, vector
        1-dimensional vector of n observed values. The A values should all be 0 or 1.
    W : ndarray, list, vector
        2-dimensional vector of n observed values for b variables to model the probability of ``A`` with.
    truncate : None, list, set, ndarray, optional
        Bounds to truncate the estimated probabilities of ``A`` at. For example, estimated probabilities above 0.99 or
        below 0.01 can be set to 0.99 or 0.01, respectively. This is done by specifying ``truncate=(0.01, 0.99)``. Note
        this step is done via ``numpy.clip(.., a_min=truncate[0], a_max=truncate[1])``, so order is important. Default
        is ``None``, which applies no truncation.

    Returns
    -------
    array :
        Returns a (3+b)-by-n NumPy array evaluated for the input ``theta``

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

    Defining psi, or the stacked estimating equations. Note that 'A' is the action.

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

    # Calculating Y(a=1)
    ya1 = (A * y) / pi - theta[1]                # i's contribution is (AY) / \pi
    # Calculating Y(a=0)
    ya0 = ((1-A) * y) / (1-pi) - theta[2]        # i's contribution is ((1-A)Y) / (1-\pi)
    # Calculating Y(a=1) - Y(a=0)
    ate = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

    # Output (3+b)-by-n stacked array
    return np.vstack((ate,             # theta[0] is for the ATE
                      ya1[None, :],    # theta[1] is for R1
                      ya0[None, :],    # theta[2] is for R0
                      preds_reg))      # theta[3:] is for the regression coefficients


def ee_aipw(theta, y, A, W, X, X1, X0, truncate=None, force_continuous=False):
    r"""Estimating equation for augmented inverse probability weighting (AIPW) estimator. AIPW consists of two nuisance
    models (the propensity score model and the outcome model). For estimation of the propensity scores, the estimating
    equations are

    .. math::

        \sum_{i=1}^n (A_i - expit(W_i^T \alpha)) W_i = 0

    where ``A`` is the treatment and ``W`` is the set of confounders. The estimating equations for the outcome model
    are

    .. math::

        \sum_{i=1}^n (Y_i - g(X_i^T \beta)) X_i = 0

    By default, `ee_aipw` detects whether `y` is all binary (zero or one), and applies logistic regression. Notice that
    ``X`` here should consists of both ``A`` and ``W`` (with possible interaction terms or other differences in
    functional forms from the propensity score model).

    The AIPW estimating equations include the causal mean difference, mean had everyone been set to ``A=1``, and the
    mean had everyone been set to ``A=0``

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            (\theta_1 - \theta_2) - \theta_0 \\
            \frac{A_i \times Y_i}{\pi_i} - \frac{\hat{Y^1}(A_i-\pi_i}{\pi_i} - \theta_1 \\
            \frac{(1-A_i) \times Y_i}{1-\pi_i} + \frac{\hat{Y^0}(A_i-\pi_i}{1-\pi_i} - \theta_2
        \end{bmatrix}
        = 0

    where :math:`\hat{Y}^a = g({X_i^*}^T \beta)`.

    Note
    ----
    Unlike ``ee_gformula``, ``ee_aipw`` always provides the average causal effect, and causal means for ``A=1`` and
    ``A=0``.


    Due to these 3 extra values and two nuisance models, the length of the parameter vector is 3+b+c, where b is the
    number of columns in ``W``, and c is the number of columns in ``X``. The *first* value in theta vector is the
    causal mean difference (or average causal effect), the *second* is the mean had everyone been given ``A=1``, the
    *third* is the mean had everyone been given ``A=0``. The remainder of the parameters correspond to the regression
    model coefficients, in the order input. The first 'chunk' of  coefficients correspond to the propensity score model
    and the last 'chunk' correspond to the outcome model.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 3+b+c values.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    A : ndarray, list, vector
        1-dimensional vector of n observed values. The A values should all be 0 or 1.
    W : ndarray, list, vector
        2-dimensional vector of n observed values for b variables to model the probability of ``A`` with.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for c variables to model the outcome ``y``.
    X1 : ndarray, list, vector
        2-dimensional vector of n observed values for b variables under the action plan where ``A=1`` for all units.
    X0 : ndarray, list, vector, None, optional
        2-dimensional vector of n observed values for b variables under the action plan where ``A=0`` for all units.
    truncate : None, list, set, ndarray, optional
        Bounds to truncate the estimated probabilities of ``A`` at. For example, estimated probabilities above 0.99 or
        below 0.01 can be set to 0.99 or 0.01, respectively. This is done by specifying ``truncate=(0.01, 0.99)``. Note
        this step is done via ``numpy.clip(.., a_min=truncate[0], a_max=truncate[1])``, so order is important. Default
        is ``None``, which applies to no truncation.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (3+b+c)-by-n NumPy array evaluated for the input theta and y

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
