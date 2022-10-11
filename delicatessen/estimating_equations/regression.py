import warnings
import numpy as np

from delicatessen.utilities import logit, inverse_logit, identity, robust_loss_functions

#################################################################
# Basic Regression Estimating Equations


def ee_regression(theta, X, y, model, weights=None):
    r"""Estimating equation for regression. Options include: linear, logistic, and Poisson regression. The general
    estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - g(X_i^T \theta)) X_i = 0

    where :math:`g` indicates a transformation function. For linear regression, :math:`g` is the identity function.
    Logistic regression uses the inverse-logit function, :math:`expit(u) = 1 / (1 + exp(u))`. Finally, Poisson
    regression is :math:`\exp(u)`.

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughout, these
    user-defined functions are defined as ``psi``.


    Here, :math:`\theta` corresponds to the coefficients in the corresponding regression model

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be implemented by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is None, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_regression`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_regression

    Some generic data to estimate the regression models

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y1'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['Y2'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
    >>> data['Y3'] = np.random.poisson(np.exp(lam=0.5 + 2*data['X'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    To start, we will demonstrate linear regression for the outcome ``Y1``. Defining psi, or the stacked estimating
    equations

    >>> def psi(theta):
    >>>     return ee_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y1'], model='linear')

    Calling the M-estimator (note that ``init`` requires 3 values, since ``X.shape[1]`` is 3).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Next, we can estimate the parameters for a logistic regression model as follows

    >>> def psi(theta):
    >>>         return ee_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y2'], model='logistic')

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Finally, we can estimate the parameters for a Poisson regression model as follows

    >>> def psi(theta):
    >>>         return ee_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y3'], model='poisson')

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Weighted models can be estimated by specifying the optional ``weights`` argument.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Preparation of input shapes and object types
    X, y, beta = _prep_inputs_(X=X, y=y, theta=theta, penalty=None)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)   # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta))          # Generating predicted values via speedy matrix calculation

    # Allowing for a weighted linear model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Output b-by-n matrix
    return w*((y - pred_y) * X).T           # Return weighted regression score function


def ee_linear_regression(theta, X, y, weights=None):
    r"""Estimating equation for linear regression.

    Note
    ----
    The function ``ee_linear_regression`` is deprecated. Please use ``ee_regression`` instead.
    """
    warnings.warn("Regression estimating equations should be implemented using `ee_regression`. The specific type of "
                  "regression estimating equations will be removed in v1.0", DeprecationWarning)
    return ee_regression(theta=theta, X=X, y=y, model='linear', weights=weights)


def ee_logistic_regression(theta, X, y, weights=None):
    r"""Estimating equation for logistic regression.

    Note
    ----
    The function ``ee_linear_regression`` is deprecated. Please use ``ee_regression`` instead.
    """
    warnings.warn("Regression estimating equations should be implemented using `ee_regression`. The specific type of "
                  "regression estimating equations will be removed in v1.0", DeprecationWarning)
    return ee_regression(theta=theta, X=X, y=y, model='logistic', weights=weights)


def ee_poisson_regression(theta, X, y, weights=None):
    r"""Estimating equation for Poisson regression.

    Note
    ----
    The function ``ee_linear_regression`` is deprecated. Please use ``ee_regression`` instead.
    """
    warnings.warn("Regression estimating equations should be implemented using `ee_regression`. The specific type of "
                  "regression estimating equations will be removed in v1.0", DeprecationWarning)
    return ee_regression(theta=theta, X=X, y=y, model='poisson', weights=weights)


#################################################################
# Robust Regression Estimating Equations


def ee_robust_regression(theta, X, y, model, k, loss='huber', weights=None, upper=None, lower=None):
    r"""Estimating equations for (unscaled) robust regression. Robust linear regression is robust to outlying
    observations of the outcome variable (``y``). Currently, only linear regression is supported by
    ``ee_robust_regression``. The estimating equation is

    .. math::

        \sum_{i=1}^n f_k(Y_i - X_i^T \theta) X_i = 0

    where :math:`f_k(x)` is the corresponding robust loss function. Options for the loss function include: Huber,
    Tukey's biweight, Andrew's Sine, and Hampel. See ``robust_loss_function`` for further details on the loss
    functions for the robust mean.

    Note
    ----
    The estimating-equation is not non-differentiable everywhere for some loss functions. Therefore, it is assumed that
    no points occur exactly at the non-differentiable points. For truly continuous :math:`Y`, the probability of that
    occurring is zero.

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be implemented via ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    model : str
        Type of regression model to estimate. Options include: ``'linear'`` (linear regression).
    k : int, float
        Tuning or hyperparameter for the chosen loss function. Notice that the choice of hyperparameter should depend
        on the chosen loss function.
    loss : str, optional
        Robust loss function to use. Default is 'huber'. Options include 'andrew', 'hampel', 'huber', 'tukey'.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is None, which assigns a weight of 1 to all observations.
    lower : int, float, None, optional
        Lower parameter for the 'hampel' loss function. This parameter does not impact the other loss functions.
        Default is ``None``.
    upper : int, float, None, optional
        Upper parameter for the 'hampel' loss function. This parameter does not impact the other loss functions.
        Default is ``None``.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_robust_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_robust_regression

    Some generic data to estimate a robust linear regression model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, scale=3, size=n)
    >>> data['C'] = 1

    >>> X = data[['C', 'X', 'Z']]
    >>> y = data['Y']

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations for Huber's robust regression

    >>> def psi(theta):
    >>>         return ee_robust_regression(theta=theta, X=X, y=y, model='linear', k=1.345, loss='huber')

    Calling the M-estimator procedure (note that ``init`` has 3 values now, since ``X.shape[1]`` is 3).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

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
    # Preparation of input shapes and object types
    X, y, beta = _prep_inputs_(X=X, y=y, theta=theta, penalty=None)

    # Allowing for a weighted linear model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model,                # Looking up corresponding transformation
                                  assert_linear_model=True)   # ... and make sure it is a linear model
    pred_y = transform(np.dot(X, beta))                       # Generating predicted values

    # Generating predictions and applying Huber function for robust
    residual = robust_loss_functions(residual=y - pred_y,     # Calculating robust residuals
                                     k=k,                     # ... hyperparameter for loss function
                                     loss=loss,               # ... chosen loss function
                                     a=lower,                 # ... upper limit (Hampel only)
                                     b=upper)                 # ... lower limit (Hampel only)

    # Output b-by-n matrix
    return w*(residual * X).T    # Score function


def ee_robust_linear_regression(theta, X, y, k, weights=None):
    r"""Estimating equations for robust linear regression.

    Note
    ----
    The function ``ee_robust_linear_regression`` is deprecated. Please use ``ee_robust_regression`` instead.
    """
    warnings.warn("Robust regression estimating equations should be implemented using `ee_robust_regression`. The "
                  "specific type of regression estimating equations will be removed in v1.0", DeprecationWarning)
    return ee_robust_regression(theta=theta, X=X, y=y, model='linear', loss='huber', k=k, weights=weights)


#################################################################
# Penalized Regression Estimating Equations


def ee_ridge_regression(theta, y, X, model, penalty, weights=None, center=0.):
    r"""Estimating equations for ridge regression. Ridge regression applies an L2-regularization through a squared
    magnitude penalty. The estimating equation(s) is

    .. math::

        \sum_{i=1}^n (Y_i - X_i^T \theta) X_i - \lambda \theta = 0

    where :math:`\lambda` is the penalty term.

    Note
    ----
    The 'strength' of the penalty term is indicated by :math:`\lambda`, which is the ``penalty`` argument scaled (or
    divided by) the number of observations.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be implemented via ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    penalty : int, float, ndarray, list, vector
        Penalty term to apply to all coefficients (if only a integer or float is provided) or the corresponding
        coefficient (if a list or vector of integers or floats is provided). Note that the penalty term should either
        consists of a single value or b values (to match the length of ``theta``). The penalty is scaled by n.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ``None``, which assigns a weight of 1 to all observations.
    center : int, float, ndarray, list, vector, optional
        Center or reference value to penalized estimated coefficients towards. Default is ``0``, which penalized
        coefficients towards the null. Other center values can be specified for all coefficients (by providing an
        integer or float) or covariate-specific centering values (by providing a vector of values of the same length as
        X).

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_ridge_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_ridge_regression

    Some generic data to estimate Ridge regression models

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['V'] = np.random.normal(size=n)
    >>> data['W'] = np.random.normal(size=n)
    >>> data['X'] = data['W'] + np.random.normal(scale=0.25, size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y1'] = 0.5 + 2*data['W'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['Y2'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['Y3'] = np.random.poisson(lam=np.exp(1 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations. Note that the penalty is a list of values. Here, we are *not*
    penalizing the intercept (which is generally recommended when the intercept is unlikely to be zero). The remainder
    of covariates have a penalty of 10 applied.

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y1']
    >>>     return ee_ridge_regression(theta=theta, X=x, y=y, model='linear', penalty=penalty_vals)

    Calling the M-estimator (note that ``init`` has 5 values now, since ``X.shape[1]`` is 5).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Next, we can estimate the parameters for a logistic regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y2']
    >>>     return ee_ridge_regression(theta=theta, X=x, y=y, model='logistic', penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Finally, we can estimate the parameters for a Poisson regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y3']
    >>>     return ee_ridge_regression(theta=theta, X=x, y=y, model='poisson', penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0., 0.])
    >>> estr.estimate(solver='lm')

    Weighted models can be estimated by specifying the optional ``weights`` argument.

    References
    ----------
    Fu WJ. (1998). Penalized regressions: the Bridge versus the LASSO. Journal of Computational and Graphical
    Statistics, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. Biometrics, 59(1), 126-132.
    """
    # Preparation of input shapes and object types
    X, y, beta, penalty, center = _prep_inputs_(X=X, y=y, theta=theta, penalty=penalty, center=center)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)      # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta))             # Generating predicted values

    # Allowing for a weighted penalized regression model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Creating penalty term for ridge regression (bridge with gamma=2 is the special case of ridge)
    penalty_terms = _bridge_penalty_(theta=theta, n_obs=X.shape[0], penalty=penalty, gamma=2, center=center)

    # Output b-by-n matrix
    return w*(((y - pred_y) * X).T - penalty_terms[:, None])    # Score function with penalty term subtracted off


def ee_lasso_regression(theta, y, X, model, penalty, epsilon=3.e-3, weights=None, center=0.):
    r"""Estimating equation for an approximate LASSO (least absolute shrinkage and selection operator) regressor. LASSO
    regression applies an L1-regularization through a magnitude penalty.

    Note
    ----
    As the derivative of the estimating equation for LASSO is not defined, the bread (and sandwich) cannot be used to
    estimate the variance in all settings.


    The estimating equation for the approximate LASSO is

    .. math::

        \sum_i^n (Y_i - X_i^T \theta) X_i - \lambda (1 + \epsilon) | \theta |^{\epsilon} sign(\theta) = 0

    where :math:`\lambda` is the penalty term.

    Here, an approximation based on the bridge penalty for the LASSO is used. For the bridge penalty, LASSO is the
    special case where :math:`\epsilon = 0`. By making :math:`\epsilon > 0`, we can approximate the LASSO. The true
    LASSO may not be possible to implement due to the existence of multiple solutions

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of
    X's (as long as there is enough support in the data).

    Note
    ----
    The 'strength' of the penalty term is indicated by :math:`\lambda`, which is the ``penalty`` argument scaled (or
    divided by) the number of observations.


    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be implemented via ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    penalty : int, float, ndarray, list, vector
        Penalty term to apply to all coefficients (if only a integer or float is provided) or the corresponding
        coefficient (if a list or vector of integers or floats is provided). Note that the penalty term should either
        consists of a single value or b values (to match the length of ``theta``).  The penalty is scaled by n.
    epsilon : float, optional
        Approximation error to use for the LASSO approximation. Default argument is ``0.003``, which results in a
        bridge penalty of 1.0003.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ``None``, which assigns a weight of 1 to all observations.
    center : int, float, ndarray, list, vector, optional
        Center or reference value to penalized estimated coefficients towards. Default is ``0``, which penalized
        coefficients towards the null. Other center values can be specified for all coefficients (by providing an
        integer or float) or covariate-specific centering values (by providing a vector of values of the same length as
        X).

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_lasso_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_lasso_regression

    Some generic data to estimate a LASSO regression model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['V'] = np.random.normal(size=n)
    >>> data['W'] = np.random.normal(size=n)
    >>> data['X'] = data['W'] + np.random.normal(scale=0.25, size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y1'] = 0.5 + 2*data['W'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['Y2'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['Y3'] = np.random.poisson(lam=np.exp(1 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations. Note that the penalty is a list of values. Here, we are *not*
    penalizing the intercept (which is generally recommended when the intercept is unlikely to be zero). The remainder
    of covariates have a penalty of 10 applied.

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y1']
    >>>     return ee_lasso_regression(theta=theta, X=x, y=y, model='linear', penalty=penalty_vals)

    Calling the M-estimator (note that ``init`` has 5 values now, since ``X.shape[1]`` is 5).

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate(solver='lm', maxiter=20000)

    Inspecting the parameter estimates

    >>> estr.theta

    Next, we can estimate the parameters for a logistic regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y2']
    >>>     return ee_lasso_regression(theta=theta, X=x, y=y, model='logistic', penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate(solver='lm', maxiter=20000)

    Finally, we can estimate the parameters for a Poisson regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y3']
    >>>     return ee_lasso_regression(theta=theta, X=x, y=y, model='poisson', penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate(solver='lm', maxiter=20000)

    Weighted models can be estimated by specifying the optional ``weights`` argument.

    References
    ----------
    Fu WJ. (1998). Penalized regressions: the bridge versus the lasso. Journal of Computational and Graphical
    Statistics, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. Biometrics, 59(1), 126-132.
    """
    # Preparation of input shapes and object types
    X, y, beta, penalty, center = _prep_inputs_(X=X, y=y, theta=theta, penalty=penalty, center=center)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)      # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta))             # Generating predicted values

    # Allowing for a weighted penalized regression model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Creating penalty term for ridge regression (bridge with gamma=2 is the special case of ridge)
    if epsilon < 0:
        raise ValueError("epsilon must be greater than zero for the approximate LASSO")
    penalty_terms = _bridge_penalty_(theta=theta, n_obs=X.shape[0], penalty=penalty, gamma=1+epsilon, center=center)

    # Output b-by-n matrix
    return w*(((y - pred_y) * X).T - penalty_terms[:, None])    # Score function with penalty term subtracted off


def ee_elasticnet_regression(theta, y, X, model, penalty, ratio, epsilon=3.e-3, weights=None, center=0.):
    r"""Estimating equations for Elastic-Net regression. Elastic-Net applies both L1- and L2-regularization at a
    pre-specified ratio. Notice that the L1 penalty is based on an approximation. See ``ee_lasso_regression`` for
    further details on the approximation for the L1 penalty.

    Note
    ----
    As the derivative of the estimating equation for Elastic-Net is not defined, the bread (and sandwich) cannot be used to
    estimate the variance in all settings.


    The estimating equation for Elastic-Net with the approximate L1 penalty is

    .. math::

        \sum_{i=1}^n (Y_i - X_i^T \theta) X_i - \lambda r (1 + \epsilon)
        | \theta |^{\epsilon} sign(\theta) - \lambda (1-r) \theta = 0

    where :math:`\lambda` is the penalty term and :math:`r` is the ratio for the L1 vs L2 penalty.

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of
    X's (as long as there is enough support in the data).

    Note
    ----
    The 'strength' of the penalty term is indicated by :math:`\lambda`, which is the ``penalty`` argument scaled (or
    divided by) the number of observations.


    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be implemented via ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    penalty : int, float, ndarray, list, vector
        Penalty term to apply to all coefficients (if only a integer or float is provided) or the corresponding
        coefficient (if a list or vector of integers or floats is provided). Note that the penalty term should either
        consists of a single value or b values (to match the length of ``theta``). The penalty is scaled by n.
    ratio : float
        Ratio for the L1 vs L2 penalty in Elastic-net. The ratio must be be :math:`0 \le r \le 1`. Setting ``ratio=1``
        results in LASSO and ``ratio=0`` results in ridge regression.
    epsilon : float, optional
        Approximation error to use for the LASSO approximation. Default argument is ``0.003``, which results in a
        bridge penalty of 1.0003.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ```None``, which assigns a weight of 1 to all observations.
    center : int, float, ndarray, list, vector, optional
        Center or reference value to penalized estimated coefficients towards. Default is ``0``, which penalized
        coefficients towards the null. Other center values can be specified for all coefficients (by providing an
        integer or float) or covariate-specific centering values (by providing a vector of values of the same length as
        X).

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_elasticnet_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_elasticnet_regression

    Some generic data to estimate a Elastic-Net regression model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['V'] = np.random.normal(size=n)
    >>> data['W'] = np.random.normal(size=n)
    >>> data['X'] = data['W'] + np.random.normal(scale=0.25, size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y1'] = 0.5 + 2*data['W'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['Y2'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['Y3'] = np.random.poisson(lam=np.exp(1 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations. Note that the penalty is a list of values. Here, we are *not*
    penalizing the intercept (which is generally recommended when the intercept is unlikely to be zero). The remainder
    of covariates have a penalty of 10 applied.

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y1']
    >>>     return ee_elasticnet_regression(theta=theta, X=x, y=y, model='linear', ratio=0.5, penalty=penalty_vals)

    Calling the M-estimator (note that ``init`` has 5 values now, since ``X.shape[1]`` is 5).

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate()

    Inspecting the parameter estimates

    >>> estr.theta

    Next, we can estimate the parameters for a logistic regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y2']
    >>>     return ee_elasticnet_regression(theta=theta, X=x, y=y, model='logistic', ratio=0.5, penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate(solver='lm', maxiter=20000)

    Finally, we can estimate the parameters for a Poisson regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y3']
    >>>     return ee_elasticnet_regression(theta=theta, X=x, y=y, model='poisson', ratio=0.5, penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate(solver='lm', maxiter=20000)

    Weighted models can be estimated by specifying the optional ``weights`` argument.

    References
    ----------
    Fu WJ. (1998). Penalized regressions: the bridge versus the lasso. Journal of Computational and Graphical
    Statistics, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. Biometrics, 59(1), 126-132.
    """
    # Preparation of input shapes and object types
    X, y, beta, penalty, center = _prep_inputs_(X=X, y=y, theta=theta, penalty=penalty, center=center)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)  # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta))  # Generating predicted values

    # Allowing for a weighted penalized regression model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Creating penalty term for ridge regression (bridge with gamma=2 is the special case of ridge)
    if epsilon < 0:
        raise ValueError("epsilon must be greater than zero for the approximate LASSO")
    if not 0 <= ratio <= 1:
        raise ValueError("The elastic-net penalty is only defined for 0 <= ratio <= 1. The input L1:L2 ratio was "
                         + str(ratio))
    penalty_l1 = _bridge_penalty_(theta=theta, n_obs=X.shape[0], penalty=penalty, gamma=1+epsilon, center=center)
    penalty_l2 = _bridge_penalty_(theta=theta, n_obs=X.shape[0], penalty=penalty, gamma=2, center=center)
    penalty_terms = ratio*penalty_l1 + (1-ratio)*penalty_l2

    # Output b-by-n matrix
    return w * (((y - pred_y) * X).T - penalty_terms[:, None])  # Score function with penalty term subtracted off


def ee_bridge_regression(theta, y, X, model, penalty, gamma, weights=None, center=0.):
    r"""Estimating equation for bridge penalized regression. The bridge penalty is a generalization of penalized
    regression, that includes L1 and L2-regularization as special cases.

    Note
    ----
    While the bridge penalty is defined for :math:`\gamma > 0`, the provided estimating equation only supports
    :math:`\gamma \ge 1`. Additionally, the derivative of the estimating equation is not defined when
    :math:`\gamma<2`. Therefore, the bread (and sandwich) cannot be used to estimate the variance in those settings.


    The estimating equation for bridge penalized regression is

    .. math::

        \sum_{i=1}^n (Y_i - X_i^T \theta) X_i - \lambda \gamma | \theta |^{\gamma - 1} sign(\theta) = 0

    where :math:`\lambda` is the penalty term and :math:`\gamma` is a tuning parameter.

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of
    X's (as long as there is enough support in the data).

    Note
    ----
    The 'strength' of the penalty term is indicated by :math:`\lambda`, which is the ``penalty`` argument scaled (or
    divided by) the number of observations.


    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be implemented via ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    penalty : int, float, ndarray, list, vector
        Penalty term to apply to all coefficients (if only a integer or float is provided) or the corresponding
        coefficient (if a list or vector of integers or floats is provided). Note that the penalty term should either
        consists of a single value or b values (to match the length of ``theta``). The penalty is scaled by n.
    gamma : float
        Hyperparameter for the bridge penalty, defined for :math:`\gamma > 0`. However, only :math:`\gamma \ge 1` are
        supported.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ``None``, which assigns a weight of 1 to all observations.
    center : int, float, ndarray, list, vector, optional
        Center or reference value to penalized estimated coefficients towards. Default is ``0``, which penalized
        coefficients towards the null. Other center values can be specified for all coefficients (by providing an
        integer or float) or covariate-specific centering values (by providing a vector of values of the same length as
        X).

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_bridge_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_bridge_regression

    Some generic data to estimate a bridge regression model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['V'] = np.random.normal(size=n)
    >>> data['W'] = np.random.normal(size=n)
    >>> data['X'] = data['W'] + np.random.normal(scale=0.25, size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y1'] = 0.5 + 2*data['W'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['Y2'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['Y3'] = np.random.poisson(lam=np.exp(1 + 2*data['W'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations. Note that the penalty is a list of values. Here, we are *not*
    penalizing the intercept (which is generally recommended when the intercept is unlikely to be zero). The remainder
    of covariates have a penalty of 10 applied.

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y']
    >>>     return ee_bridge_regression(theta=theta, X=x, y=y, model='linear', gamma=2.3, penalty=penalty_vals)

    Calling the M-estimator (note that ``init`` has 5 values now, since ``X.shape[1]`` is 5).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0., 0.])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Next, we can estimate the parameters for a logistic regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y2']
    >>>     return ee_bridge_regression(theta=theta, X=x, y=y, model='logistic', gamma=2.3, penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate(solver='lm', maxiter=5000)

    Finally, we can estimate the parameters for a Poisson regression model as follows

    >>> penalty_vals = [0., 10., 10., 10., 10.]
    >>> def psi(theta):
    >>>     x, y = data[['C', 'V', 'W', 'X', 'Z']], data['Y3']
    >>>     return ee_bridge_regression(theta=theta, X=x, y=y, model='poisson', gamma=2.3, penalty=penalty_vals)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.01, 0.01, 0.01, 0.01, 0.01])
    >>> estr.estimate(solver='lm', maxiter=5000)

    Weighted models can be estimated by specifying the optional ``weights`` argument.

    References
    ----------
    Fu WJ. (1998). Penalized regressions: the bridge versus the lasso. Journal of Computational and Graphical
    Statistics, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. Biometrics, 59(1), 126-132.
    """
    # Preparation of input shapes and object types
    X, y, beta, penalty, center = _prep_inputs_(X=X, y=y, theta=theta, penalty=penalty, center=center)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)  # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta))  # Generating predicted values

    # Allowing for a weighted penalized regression model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Creating penalty term for ridge regression (bridge with gamma=2 is the special case of ridge)
    penalty_terms = _bridge_penalty_(theta=theta, n_obs=X.shape[0], penalty=penalty, gamma=gamma, center=center)

    # Output b-by-n matrix
    return w * (((y - pred_y) * X).T - penalty_terms[:, None])  # Score function with penalty term subtracted off


#################################################################
# Utility functions for regression equations

def _prep_inputs_(X, y, theta, penalty=None, center=None):
    """Internal use function to simplify variable transformations for regression. This function is used on the inputs
    to ensure they are the proper shapes

    Parameters
    ----------
    X : ndarray
        Dependent variables
    y : ndarray
        Independent variable
    theta : ndarray
        Input parameters
    penalty : ndarray, None, optional
        Input penalty term(s)

    Returns
    -------
    transformed parameters
    """
    X = np.asarray(X)                       # Convert to NumPy array
    y = np.asarray(y)[:, None]              # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra

    if penalty is None:                     # Return the transformed objects
        return X, y, beta
    else:                                   # Convert penalty term then return all
        penalty = np.asarray(penalty)       # Convert to NumPy array
        center = np.asarray(center)         # Convert to NumPy array
        return X, y, beta, penalty, center


def _model_transform_(model, assert_linear_model=False):
    """Internal use function to simplify the checking procedure for the model form to use. Takes the input string and
    returns the corresponding function for the variable transformation.

    Parameters
    ----------
    model : str
        Model identifier to calculate the transformation for

    Returns
    -------
    function
    """
    # Checking object type (and convert to lower-case)
    if isinstance(model, str):              # If string, convert to lower-case for internal handling
        model = model.lower()
    else:
        raise ValueError("The model argument must be a str object.")

    # forcing model to be 'linear' (used by ee_robust_regression)
    if assert_linear_model and model != 'linear':
        raise ValueError("The selected estimating equation only supports linear regression.")

    # Process the model transformations
    if model == 'linear':                   # If linear regression
        transform = identity                    # ... no transformation needed
    elif model == 'logistic':               # If logistic regression
        transform = inverse_logit               # ... expit (inverse_logit) transformation
    elif model == 'poisson':                # If Poisson regression
        transform = np.exp                      # ... exponential transformation
    else:                                   # Else results in error
        raise ValueError("Invalid input:", model,
                         ". Please select: 'linear', 'logistic', or 'poisson'.")
    return transform


def _generate_weights_(weights, n_obs):
    """Internal use function to return the weights assigned to each observation. Returns a vector of 1's when no
    weights are provided. Otherwise, converts provided vector into a numpy array.

    Parameters
    ----------
    weights : None, ndarray, list
        Vector of weights, or None if no weights are provided
    n_obs : int
        Number of observations in the data

    Returns
    -------
    ndarray
    """
    if weights is None:                     # If weights is unspecified
        w = np.ones(n_obs)                      # ... assign weight of 1 to all observations
    else:                                   # Otherwise
        w = np.asarray(weights)                 # ... set weights as input vector
    return w


def _bridge_penalty_(theta, gamma, penalty, n_obs, center):
    r"""Internal use function to calculate the corresponding penalty term. The penalty term formula is based on the
    bridge penalty, where LASSO is :math:`\gamma = 1` and ridge is :math:`\gamma = 2`. The penalty term is defined for
    :math:`\gamma > 0` but :math:`\gamma < 1` requires special optimization.

    Note
    ----
    All penalties are scaled by the number of observations.

    The penalty term for the score function (first derivative) is:

    .. math::

        \lambda \gamma | \theta |^{\gamma - 1} sign(\theta)

    where :math:`\lambda` is the (scaled) penalty, :math:`\gamma` is the hyperparameter for the bridge penalty, and
    :math:`\theta` are the regression coefficients.

    Parameters
    ----------
    theta : ndarray, list, vector
        Regression coefficients to penalize. ``theta`` in this case consists of b values.
    gamma : float, int
        Hyperparameter for the bridge penalty, defined for :math:`\gamma > 0`. Notice that :math:`\gamma = 1`
        corresponds to LASSO, and :math:`\gamma = 2` corresponds to ridge.
    penalty : int, float, ndarray, list, vector
        Penalty term to apply to all coefficients (if only a integer or float is provided) or the corresponding
        coefficient (if a list or vector of integers or floats is provided). Note that the penalty term should either
        consists of a single value or b values (to match the length of ``theta``).
    n_obs : int
        Number of observations. Used to rescale the penalty terms

    Returns
    -------
    ndarray
    """
    # Checking the penalty term is non-negative
    if penalty.size != 1:
        if penalty.shape[0] != len(theta):
            raise ValueError("The penalty term must be either a single number or the same length as theta.")
    if center.size != 1:
        if center.shape[0] != len(theta):
            raise ValueError("The center term must be either a single number or the same length as theta.")

    # Checking a valid hyperparameter is being provided
    if gamma < 1:
        raise ValueError("L_{gamma} for `gamma` < 1 is not currently able to be supported with estimating equations "
                         "evaluated using numerical methods.")
    # Warning about the bread for problematic hyperparameters
    if gamma < 2:
        warnings.warn("The estimating equation for chosen penalized regression model is not always differentiable. "
                      "Therefore, the bread matrix is not always defined for finite samples, and the sandwich should "
                      "not be used to estimate the variance.",
                      UserWarning)

    # Calculating the penalties
    penalty_scaled = penalty / (gamma * n_obs)
    penalty_terms = penalty_scaled * gamma * (np.abs(theta - center)**(gamma-1)) * np.sign(theta - center)
    return penalty_terms
