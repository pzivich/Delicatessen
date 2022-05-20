import warnings
import numpy as np

from delicatessen.utilities import logit, inverse_logit, identity

#################################################################
# Basic Regression Estimating Equations


def ee_regression(theta, X, y, model, weights=None):
    r"""Default stacked estimating equation for regression, with available options including: linear, logistic, and
    Poisson regression. The general estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - g(X_i^T \theta)) X_i = 0

    where :math:`g` indicates a general transformation function. For linear regression, :math:`g` is the identity
    function. Logistic regression uses the expit, or the inverse-logit function, :math:`expit(u) = 1 / (1 + exp(u))`.
    Finally, Poisson regression is :math:`g(u) = \exp(u)`.

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughout, these
    user-defined functions are defined as ``psi``.

    Here, :math:`\theta` corresponds to the coefficients in the corresponding regression model

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

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
    >>> data['Y3'] = np.random.poisson(lam=10.5 + 2*data['X'] - 1*data['Z'], size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    To start, we will demonstrate linear regression for the outcome ``Y1``. Defining psi, or the stacked estimating
    equations

    >>> def psi(theta):
    >>>         return ee_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y1'], model='linear')

    Calling the M-estimation procedure (note that ``init`` requires 3 values, since ``X.shape[1] = 3``).

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

    Additionally, weighted versions of all the previous models can be estimated by specifying the optional ``weights``
    argument.

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
    r"""Default stacked estimating equation for linear regression without the homoscedastic assumption.

    Note
    ----
    The function ``ee_linear_regression`` is deprecated. Please use ``ee_regression`` instead.

    The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i = 0

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Here, theta corresponds to the coefficients in a linear regression model

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely OLS). Therefore, optimization of OLS via a separate functionality can be done then those
    estimated parameters are fed forward as the initial values (which should result in a more stable optimization).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_linear_regression`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_linear_regression

    Some generic data to estimate a linear regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])

    Calling the M-estimation procedure (note that ``init`` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    warnings.warn("Regression estimating equations should be implemented using `ee_regression`. The specific type of "
                  "regression estimating equations will be removed in v1.0", DeprecationWarning)
    return ee_regression(theta=theta, X=X, y=y, model='linear', weights=weights)


def ee_logistic_regression(theta, X, y, weights=None):
    r"""Default stacked estimating equation for logistic regression.

    Note
    ----
    The function ``ee_linear_regression`` is deprecated. Please use ``ee_regression`` instead.

    The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    where expit, or the inverse logit is

    .. math::

        expit(u) = 1 / (1 + exp(u))

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Here, theta corresponds to the coefficients in a logistic regression model, and therefore are the log-odds.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely logistic regression). Therefore, optimization of logistic regression via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).


    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_logistic_regression`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_logistic_regression

    Some generic data to estimate a logistic regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_logistic_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])


    Calling the M-estimation procedure (note that `init` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    warnings.warn("Regression estimating equations should be implemented using `ee_regression`. The specific type of "
                  "regression estimating equations will be removed in v1.0", DeprecationWarning)
    return ee_regression(theta=theta, X=X, y=y, model='logistic', weights=weights)


def ee_poisson_regression(theta, X, y, weights=None):
    r"""Default stacked estimating equation for Poisson regression.

    Note
    ----
    The function ``ee_linear_regression`` is deprecated. Please use ``ee_regression`` instead.

    The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - \exp(X_i^T \theta)) X_i = 0

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely logistic regression). Therefore, optimization of logistic regression via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y
    """
    warnings.warn("Regression estimating equations should be implemented using `ee_regression`. The specific type of "
                  "regression estimating equations will be removed in v1.0", DeprecationWarning)
    return ee_regression(theta=theta, X=X, y=y, model='poisson', weights=weights)


#################################################################
# Robust Regression Estimating Equations


def ee_robust_regression(theta, X, y, model, k, weights=None):
    r"""Default stacked estimating equation for robust linear regression. Specifically, robust linear regression is
    robust to outlying observations of the outcome variable (``y``). The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n \psi_k(Y_i - X_i^T \theta) X_i = 0

    where k indicates the upper and lower bounds. Here, theta is a 1-by-b array, where b is the distinct covariates
    included as part of X. For example, if X is a 3-by-n matrix, then theta will be a 1-by-3 array. The code is general
    to allow for an arbitrary number of X's (as long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Here, theta corresponds to the coefficients in a robust linear regression model

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely OLS). Therefore, optimization of OLS via a separate functionality can be done then those
    estimated parameters are fed forward as the initial values (which should result in a more stable optimization).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    k : int, float
        Value to set the symmetric maximum upper and lower bounds on the difference between the observations and
        predicted values
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_robust_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_robust_linear_regression

    Some generic data to estimate a robust linear regression model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, scale=3, size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_robust_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'], k=3)


    Calling the M-estimation procedure (note that ``init`` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Preparation of input shapes and object types
    X, y, beta = _prep_inputs_(X=X, y=y, theta=theta, penalty=None)

    # Allowing for a weighted linear model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Generating predictions and applying Huber function for robust
    preds = np.clip(y - np.dot(X, beta), -k, k)

    # Output b-by-n matrix
    return w*(preds                  # ... linear regression requires no transformations
              * X).T                 # ... multiply by coefficient and transpose for correct orientation


def ee_robust_linear_regression(theta, X, y, k, weights=None):
    r"""Default stacked estimating equation for robust linear regression. Specifically, robust linear regression is
    robust to outlying observations of the outcome variable (``y``). The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n \psi_k(Y_i - X_i^T \theta) X_i = 0

    where k indicates the upper and lower bounds. Here, theta is a 1-by-b array, where b is the distinct covariates
    included as part of X. For example, if X is a 3-by-n matrix, then theta will be a 1-by-3 array. The code is general
    to allow for an arbitrary number of X's (as long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as ``psi``.

    Here, theta corresponds to the coefficients in a robust linear regression model

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely OLS). Therefore, optimization of OLS via a separate functionality can be done then those
    estimated parameters are fed forward as the initial values (which should result in a more stable optimization).

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    k : int, float
        Value to set the symmetric maximum upper and lower bounds on the difference between the observations and
        predicted values
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_robust_linear_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_robust_linear_regression

    Some generic data to estimate a robust linear regresion model

    >>> n = 100
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, scale=3, size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_robust_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'], k=3)


    Calling the M-estimation procedure (note that ``init`` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Preparation of input shapes and object types
    X, y, beta = _prep_inputs_(X=X, y=y, theta=theta, penalty=None)

    # Allowing for a weighted linear model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Generating predictions and applying Huber function for robust
    preds = np.clip(y - np.dot(X, beta), -k, k)

    # Output b-by-n matrix
    return w*(preds                  # ... linear regression requires no transformations
              * X).T                 # ... multiply by coefficient and transpose for correct orientation


#################################################################
# Penalized Regression Estimating Equations
# TODO LASSO, Elastic-Net, Bridge


def ee_ridge_regression(theta, y, X, model, penalty, weights=None):
    r"""Default stacked estimating equation for ridge linear regression. Ridge regression applies an L2-regularization
    through a squared magnitude penalty. The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i - \lambda \theta = 0

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    The 'strength' of the penalty term is indicated by :math:`\lambda`, which is the ``penalty`` argument scaled (or
    divided by) the number of observations.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by ``[0, ] * X.shape[1]``.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : ndarray, list, vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    penalty : int, float, ndarray, list, vector
        Penalty term to apply to all coefficients (if only a integer or float is provided) or the corresponding
        coefficient (if a list or vector of integers or floats is provided). Note that the penalty term should either
        consists of a single value or b values (to match the length of ``theta``).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. No missing weights should be included. Default is None, which assigns a
        weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_ridge_linear_regression`` should be done similar to the
    following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_ridge_regression

    Some generic data to estimate a linear regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         x, y = data[['C', 'X', 'Z']], data['Y']
    >>>         return ee_ridge_regression(theta=theta, X=x, y=y, model='linear', penalty=5.5)

    Calling the M-estimation procedure (note that ``init`` has 3 values now, since ``X.shape[1] = 3``).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    References
    ----------
    Fu WJ. (1998). Penalized regressions: the bridge versus the lasso. Journal of Computational and Graphical
    Statistics, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. Biometrics, 59(1), 126-132.
    """
    # Preparation of input shapes and object types
    X, y, beta, penalty = _prep_inputs_(X=X, y=y, theta=theta, penalty=penalty)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)      # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta))             # Generating predicted values

    # Allowing for a weighted penalized regression model
    w = _generate_weights_(weights=weights, n_obs=X.shape[0])

    # Creating penalty term for ridge regression (bridge with gamma=2 is the special case of ridge)
    penalty_terms = _bridge_penalty_(theta=theta, n_obs=X.shape[0], penalty=penalty, gamma=2)

    # Output b-by-n matrix
    return w*(((y - pred_y) * X).T - penalty_terms[:, None])    # Score function with penalty term subtracted off


#################################################################
# Utility functions for regression equations


def _prep_inputs_(X, y, theta, penalty=None):
    """Internal use function to simplify variable transformations for regression. This function is used on the inputs
    to ensure they are the proper shapes

    Parameters
    ----------
    X : ndarray
    y : ndarray
    theta : ndarray
    penalty : ndarray, None, optiona

    Returns
    -------

    """
    X = np.asarray(X)                       # Convert to NumPy array
    y = np.asarray(y)[:, None]              # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra

    if penalty is None:                     # Return the transformed objects
        return X, y, beta
    else:                                   # Convert penalty term then return all
        penalty = np.asarray(penalty)       # Convert to NumPy array
        return X, y, beta, penalty


def _model_transform_(model):
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


def _bridge_penalty_(theta, gamma, penalty, n_obs):
    r"""Internal use function to calculate the corresponding penalty term. The penalty term formula is based on the
    bridge penalty, where LASSO is :math:`\gamma = 1` and ridge is :math:`\gamma = 2`. The penalty term is defined for
    :math:`\gamma > 0` but :math:`\gamma < 1` requires special optimization. While the penalty term does not prevent
    inputs less than 1, it will generate a :code:`UserWarning`.

    Note
    ----
    All penalties are scaled by the number of observations.

    The penalty term for the score function (first derivative) is:

    .. math::

        \lambda \gamma | \theta |^{\gamma - 1} sgn(\theta)

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
    """
    # Checking the penalty term is non-negative
    if penalty.size != 1:
        if penalty.shape[0] != len(theta):
            raise ValueError("The penalty term must be either a single number or the same length as theta.")

    # Checking a valid hyperparameter is being provided
    # if gamma <= 0:
    #     raise ValueError("L_{gamma} is only available for `gamma` > 0")
    if gamma < 1:
        # warnings.warn("L_{gamma} for `gamma` < 1 is difficult to solve. Therefore, penalization behavior should not "
        #               "be trusted.", UserWarning)
        raise ValueError("L_{gamma} for `gamma` < 1 is not currently able to be supported with estimating equations.")

    # Calculating the penalties
    penalty_scaled = penalty / (gamma * n_obs)
    penalty_terms = penalty_scaled * gamma * (np.abs(theta)**(gamma-1)) * np.sign(theta)
    return penalty_terms
