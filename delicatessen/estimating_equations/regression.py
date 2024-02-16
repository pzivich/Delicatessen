#####################################################################################################################
# Estimating functions for regression models
#####################################################################################################################

import warnings
import numpy as np
from scipy.stats import norm, cauchy

from delicatessen.utilities import (logit, inverse_logit, identity,
                                    robust_loss_functions,
                                    additive_design_matrix,
                                    digamma, polygamma, standard_normal_cdf, standard_normal_pdf)
from delicatessen.estimating_equations.processing import generate_weights

#################################################################
# Basic Regression Estimating Equations


def ee_regression(theta, X, y, model, weights=None, offset=None):
    r"""Estimating equation for regression. Options include: linear, logistic, and Poisson regression. The general
    estimating equation is

    .. math::

        \sum_{i=1}^n \left\{ Y_i - g(X_i^T \theta) \right\} X_i = 0

    where :math:`g` indicates a transformation function. For linear regression, :math:`g` is the identity function.
    Logistic regression uses the inverse-logit function, :math:`\text{expit}(u) = 1 / (1 + \exp(u))`. Finally, Poisson
    regression is :math:`\exp(u)`.

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then :math:`\theta` will be a 1-by-3 array. The code is general to allow for an arbitrary
    number of X's (as long as there is enough support in the data).

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
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

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
    >>> data['Y3'] = np.random.poisson(lam=np.exp(0.5 + 2*data['X'] - 1*data['Z']), size=n)
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
    X, y, beta, offset = _prep_inputs_(X=X, y=y, theta=theta, penalty=None, offset=offset)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)    # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta) + offset)  # Generating predicted values via speedy matrix calculation

    # Allowing for a weighted linear model
    w = generate_weights(weights=weights, n_obs=X.shape[0])

    # Output b-by-n matrix
    return w*((y - pred_y) * X).T           # Return weighted regression score function


def ee_glm(theta, X, y, distribution, link, hyperparameter=None, weights=None, offset=None):
    r"""Estimating equation for regression with a generalized linear model. Unlike ``ee_regression``, this functionality
    supports generic distribution and link specifications. The general estimating equation for the outcome :math:`Y_i`
    with the design matrix :math:`X_i`

    .. math::

        \sum_{i=1}^n W_i \left\{ Y_i - g^{-1}(X_i^T \theta) \times \frac{D(\theta)}{v(\theta)} \right\} X_i = 0

    where :math:`g` is the link function, :math:`g^{-1}` is the inverse link function, :math:`D(\theta)` is the
    derivative of the inverse link function by :math:`\theta`, and :math:`v(\theta)` is the variance function for the
    specified distribution.

    Note
    ----
    Some distributions (i.e., negative-binomial, gamma) involve additional parameters. These are estimated using
    additional parameter-specific estimating equations.


    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then :math:`\theta` will be a 1-by-3 array. The code is general to allow for an arbitrary
    number of X's (as long as there is enough support in the data).

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
    distribution : str
        Distribution for the generalized linear model. Options are:
        ``'normal'`` (alias: ``gaussian``),
        ``'binomial'`` (aliases: ``bernoulli``, ``bin``),
        ``'poisson'``,
        ``'gamma'``,
        ``'inverse_normal'`` (alias: ``inverse_gaussian``),
        ``'negative_binomial'`` (alias: ``nb``),
        and ``'tweedie'``.
    link : str
        Link function for the generalized linear model. Options are:
        ``identity``,
        ``log``,
        ``logistic`` (alias: ``logit``),
        ``probit``,
        ``cauchit`` (alias: ``cauchy``),
        ``loglog``,
        ``cloglog``,
        ``inverse``,
        and ``square_root`` (alias: ``sqrt``).
    hyperparameter : None, int, float
        Hyperparameter specification. Default is None. This option is only used by the tweedie distribution. It is
        ignored by all other distributions.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is None, which assigns a weight of 1 to all observations.
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

    Note
    ----
    Link and distribution combinations are not checked for their validity. Some pairings may not converge or may
    produce nonsensical results. Please check the combination you are using is valid.

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_regression`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_glm

    Some generic data to estimate the regression models

    >>> d = pd.DataFrame()
    >>> d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
    >>> d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> d['Y1'] = [1, 2, 4, 5, 2, 3, 1, 1, 3, 4, 2, 3, 7, 8, 2, 2, 1, 4, 2, 1]
    >>> d['Y2'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
    >>> d['C'] = 1
    >>> X = d[['C', 'X', 'Z']]  # design matrix used hereafter

    To start, we will demonstrate a GLM with a normal distribution and identity link. This GLM is equivalent to linear
    regression. Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_glm(theta, X=X, y=d['Y1'],
    >>>                   distribution='normal', link='identity')

    Calling the M-estimator (note that ``init`` requires 3 values, since ``X.shape[1]`` is 3).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Next, we will show a GLM with a binomial distribution and log link. This GLM can be used for binary data and
    estimates log risk ratios. So, one may prefer this model for interpretability over logistic regression (GLM with
    binomial distribution and logit link).

    >>> def psi(theta):
    >>>     return ee_glm(theta, X=X, y=d['Y2'],
    >>>                   distribution='binomial', link='log')

    Calling the M-estimator (note that ``init`` requires 3 values, since ``X.shape[1]`` is 3).

    >>> estr = MEstimator(stacked_equations=psi, init=[-0.9, 0., 0.,])
    >>> estr.estimate()

    Notice that the root-finding solution may go off to weird places if bad starting values are given for the
    log-binomial GLM. This is because the log-binomial GLM is not bounded. Providing starting values close to the truth
    (or changing link functions) can help alleviate these issues. Other variations for binomial distribution link
    functions that are bounded include: ``logit``, ``cauchy``, ``probit``, or ``loglog``.

    The negative-binomial and gamma distributions for GLM have an additional parameter that is estimated. Therefore,
    both of these distribution specifications require ``X.shape[1] + 1`` input starting values. Here, we illustrate
    a gamma distribution and log link GLM

    >>> def psi(theta):
    >>>     return ee_glm(theta, X=X, y=d['Y1'],
    >>>                   distribution='gamma', link='log')

    Calling the M-estimator (note that ``init`` requires 4 values, since ``X.shape[1]`` is 3 and the gamma distribution
    has an additional parameter).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0.])
    >>> estr.estimate()

    Note that ``delicatessen`` appropriately incorporates the estimation of the additional parameter for the
    negative-binomial and gamma distributions. This is unlike some statistical software that estimates this parameter
    but does *not* incorporate the uncertainty in estimation of that parameter. This may explain differences you
    encounter across software (and the ``delicatessen`` implementation is preferred, as it is a more honest expression
    of the uncertainty).

    Finally, the tweedie distribution for GLM is a generalization of the Poisson and gamma distributions. Unlike the
    negative-binomial and gamma distributions, there is a fixed (i.e., not estimated) hyperparameter bounded to be >0.
    When the tweedie distribution hyperparameter is set to 1, it is equivalent to the Poisson distribution. When the
    tweedie distribution hyperparameter is set to 2, it is equivalent to the gamma distribution. When the tweedie
    distribution hyperparameter is set to 3, it is equivalent to the inverse-normal distribution. However, the tweedie
    distribution hyperparameter can be specified for any values. Here, we illustrate the tweedie distribution that is
    between a Poisson and gamma distribution.

    >>> def psi(theta):
    >>>     return ee_glm(theta, X=X, y=d['Y1'],
    >>>                   distribution='tweedie', link='log',
    >>>                   hyperparameter=1.5)

    Calling the M-estimator (note that ``init`` requires 3 values, since ``X.shape[1]`` is 3).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.])
    >>> estr.estimate()

    Notice that the tweedie distribution does not estimate an additional parameter, unlike the gamma distribution GLM
    described previously.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Hilbe JM. (2011). *Negative Binomial Regression*. Cambridge University Press.

    Nakashima E. (1997). Some methods for estimation in a Negative-Binomial model.
    *Annals of the Institute of Statistical Mathematics*, 49, 101-115.
    """
    distribution = distribution.lower()
    if distribution in ['gamma', 'negative_binomial', 'nb']:
        beta, alpha = theta[:-1], np.exp(theta[-1])
    else:
        beta = theta
        alpha = None

    # Preparation of input shapes and object types
    X, y, beta, offset = _prep_inputs_(X=X, y=y, theta=beta, penalty=None, offset=offset)

    # Transforming data for score equations
    betaX = np.dot(X, beta) + offset                                   # Compute (X * B)
    pred_y, deriv = _inverse_link_(betax=betaX, link=link)             # Compute g^{-1}(X * B), d/dB g^{-1}(X * B)
    variance = _distribution_variance_(dist=distribution, mu=pred_y,   # Compute v(g^{-1}(X * B))
                                       hyperparameter=hyperparameter,  # ... hyperparameter for tweedie distribution
                                       alpha=alpha)                    # ... hyperparameter for negative-binomial

    # Allowing for a weighted generalized linear model
    w = generate_weights(weights=weights, n_obs=X.shape[0])            # Compute the corresponding weight vector

    # Generic score functions for GLM
    ee_beta = w*((y - pred_y) * deriv / variance * X).T

    # Additional processing of regression models with additional parameters
    if distribution == 'gamma':                                                          # Gamma model
        ee_alpha = w*((1 - y / pred_y) + np.log(alpha * y / pred_y) - digamma(alpha)).T  # ... nuisance for gamma
        return np.vstack([ee_beta, ee_alpha])                                            # ... return stacked EE
    elif distribution in ['negative_binomial', 'nb']:                                    # Negative Binomial model
        p1 = - alpha ** -2 * polygamma(0, y + 1 / alpha)                                 # ... breaking equation into
        p2 = alpha ** -2 * polygamma(0, 1 / alpha)                                       # ... simpler pieces
        p3 = y / (alpha ** 2 * pred_y + alpha)
        p4 = - (alpha*pred_y / (alpha*pred_y + 1) + np.log(1 / (alpha*pred_y + 1))) / alpha**2
        ee_alpha = (p1 + p2 + p3 + p4).T
        return np.vstack([ee_beta, ee_alpha])                                            # ... return stacked EE
    else:                                                                                # All other models
        return ee_beta                                                                   # ... only beta EE


def ee_mlogit(theta, X, y, weights=None, offset=None):
    r"""Estimating equation for multinomial logistic regression. This estimating equation functionality supports
    unranked categorical outcome data, unlike ``ee_regression`` and ``ee_glm``.

    Note
    ----
    Unlike the other regression estimating equations, ``ee_mlogit`` expects a matrix of indicators for each possible
    value of ``y``, with the first column being used as the referent category. In other words, the outcome variable is
    a matrix of dummy variables that includes the reference.


    The estimating equation for column :math:`r` of the indicator variable :math:`Y_{r}`
    of a :math:`Y` with :math:`k` unique categories is

    .. math::

        \sum_{i=1}^n \left\{ Y_{r,i} - \frac{\exp(X_i^T \theta_r)}{1 + \sum_{j=2}^{k} \exp(X_i^T \theta_j)}  \right\}
        X_i = 0

    where :math:`\theta_r` are the coefficients correspond to the log odds ratio comparing :math:`Y_r` to all other
    categories of :math:`Y`. Here, :math:`\theta` is a 1-by-(b :math`\times` (k-1)) array, where b is the distinct
    covariates included as part of X. So, the stack of estimating equations consists of :math:`(k-1)` estimating
    equations of the dimension :math:`X_i`. For example, if X is a 3-by-n matrix and :math:`Y` has three unique
    categories, then :math:`\theta` will be a 1-by-6 array.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughout, these
    user-defined functions are defined as ``psi``.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta in this case consists of b :math:`\times` (k-1) values. Therefore, initial values should consist of the
        same number as the number of columns present in the design matrix for each category of the outcome matrix
        besides the reference.
    X : ndarray, list, vector
        2-dimensional design matrix of n observed covariates for b variables.
    y : ndarray, list, vector
        2-dimensional indicator matrix of n observed outcomes.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is None, which assigns a weight of 1 to all observations.
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

    Returns
    -------
    array :
        Returns a (b*(k-1))-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_regression`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mlogit

    Some generic data to estimate a multinomial logistic regression model

    >>> d = pd.DataFrame()
    >>> d['W'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> d['Y'] = [1, 1, 1, 1, 2, 2, 3, 3, 3, 1, 2, 2, 3, 3]
    >>> d['C'] = 1

    First, notice that ``Y`` needs to be pre-processed for use with ``ee_mlogit``. To prepare the data, we need to
    convert ``d['Y']`` into a matrix of indicator variables. We can do this manually by

    >>> d['Y1'] = np.where(d['Y'] == 1, 1, 0)
    >>> d['Y2'] = np.where(d['Y'] == 2, 1, 0)
    >>> d['Y3'] = np.where(d['Y'] == 3, 1, 0)

    This can also be accomplished with ``pd.get_dummies(d['Y'], drop_first=False)``.

    For the reference category, we want to have ``Y=1`` as the reference. Therefore, ``Y1`` will be the first column in
    ``y``. The pair of matrices are

    >>> y = d[['Y1', 'Y2', 'Y3']]
    >>> X = d[['C', 'W']]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mlogit(theta, X=X, y=y)

    Calling the M-estimator (note that ``init`` requires 4 values, since ``X.shape[1]`` is 2 and ``y.shape[1]`` is 3).

    >>> estr = MEstimator(stacked_equations=psi, init=[0., 0., 0., 0.])
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Here, the first two values of ``theta`` correspond to ``Y2`` and the last two values of ``theta`` correspond to
    ``Y3``.

    A weighted multinomial logistic regression can be implemented by specifying the ``weights`` argument. An offset can
    be added by specifying the ``offset`` argument.

    References
    ----------
    Kwak C & Clayton-Matthews A. (2002). Multinomial logistic regression. *Nursing Research*, 51(6), 404-410.
    """
    # Preparation of input shapes and object types
    X, y, theta, offset = _prep_inputs_(X=X, y=y, theta=theta, penalty=None, offset=offset, reshape_y=False)
    w = generate_weights(weights=weights, n_obs=X.shape[0])          # Compute the corresponding weight vector

    # Setting up parameters for later use
    n_y_vals = y.shape[1]                             # Number of categories in Y
    n_x_vals = X.shape[1]                             # Number of columns / predictors in X
    n_params = theta.shape[0]                         # Number of parameters provided
    denom = 1                                         # Default value for denominator
    exp_pred_y = []                                   # Storage for the expected values of Y
    efuncs = []                                       # Storage for the stacked estimating functions
    start_index = 0                                   # Starting index for looping over beta for all Y categories

    # Checking that shapes agree to prevent user headaches
    if (n_y_vals-1) * n_x_vals != theta.shape[0]:
        raise ValueError("There is a mismatch in the number of provided parameters, " + str(n_params)
                         + ", and the number of parameters for " + str(n_y_vals) + " columns of Y and "
                         + "a design matrix with " + str(n_x_vals) + " columns. There should be "
                         + str((n_y_vals-1) * n_x_vals) + " parameters.")

    # Computing the overall denominator for the multinomial logistic model
    for i in range(1, n_y_vals):                      # Looping over all columns of Y
        end_index = start_index + n_x_vals            # ... get the current end_index
        beta_i = theta[start_index: end_index]        # ... grab the corresponding beta's for Y column
        pred_y = np.exp(np.dot(X, beta_i) + offset)   # ... generate predicted value of Y column
        exp_pred_y.append(pred_y)                     # ... store the particular predicted values for Y
        denom = denom + pred_y                        # ... update the denominator with summation
        start_index = end_index                       # ... update start_index to current end_index

    # Computing the stacked estimating equations for each column of Y
    yhat_ref = y[:, 0][:, None] - 1/denom                          # Compute the residual for the reference Y
    for i in range(1, n_y_vals):                                   # Looping over all columns of Y
        y_reshape = y[:, i][:, None]                               # ... extract and reshape current Y indicator
        yhat_i = yhat_ref + (y_reshape - exp_pred_y[i-1]/denom)    # ... get residuals for current Y versus reference Y
        residual = w*(yhat_i*X).T                                  # ... expanding residuals by design matrix
        efuncs.append(residual)                                    # ... store the current residuals

    # Output b-by-n matrix
    return np.vstack(efuncs)


#################################################################
# Robust Regression Estimating Equations


def ee_robust_regression(theta, X, y, model, k, loss='huber', weights=None, upper=None, lower=None, offset=None):
    r"""Estimating equations for (unscaled) robust regression. Robust linear regression is robust to outlying
    observations of the outcome variable. Currently, only linear regression is supported by
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


    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then :math:`\theta` will be a 1-by-3 array. The code is general to allow for an arbitrary
    number of X's (as long as there is enough support in the data).

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
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

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

    Weighted models can be estimated by specifying the optional ``weights`` argument.

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
    X, y, beta, offset = _prep_inputs_(X=X, y=y, theta=theta, penalty=None, offset=offset)

    # Allowing for a weighted linear model
    w = generate_weights(weights=weights, n_obs=X.shape[0])

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model,                # Looking up corresponding transformation
                                  assert_linear_model=True)   # ... and make sure it is a linear model
    pred_y = transform(np.dot(X, beta) + offset)              # Generating predicted values

    # Generating predictions and applying Huber function for robust
    residual = robust_loss_functions(residual=y - pred_y,     # Calculating robust residuals
                                     k=k,                     # ... hyperparameter for loss function
                                     loss=loss,               # ... chosen loss function
                                     a=lower,                 # ... upper limit (Hampel only)
                                     b=upper)                 # ... lower limit (Hampel only)

    # Output b-by-n matrix
    return w*(residual * X).T    # Score function


#################################################################
# Penalized Regression Estimating Equations


def ee_ridge_regression(theta, X, y, model, penalty, weights=None, center=0., offset=None):
    r"""Estimating equations for ridge regression. Ridge regression applies an L2-regularization through a squared
    magnitude penalty. The estimating equation for Ridge linear regression is

    .. math::

        \sum_{i=1}^n \left\{(Y_i - X_i^T \theta) X_i - \lambda \theta \right\} = 0

    where :math:`\lambda` is the penalty term.

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then :math:`\theta` will be a 1-by-3 array. The code is general to allow for an arbitrary
    number of X's (as long as there is enough support in the data).

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
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

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
    Fu WJ. (1998). Penalized regressions: the Bridge versus the LASSO. *Journal of Computational and Graphical
    Statistics*, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. *Biometrics*, 59(1), 126-132.
    """
    # Calling internal bridge penalized regression for implementation
    return ee_bridge_regression(theta=theta,
                                X=X, y=y,
                                model=model,
                                weights=weights,
                                penalty=penalty, gamma=2, center=center,
                                offset=offset)


def ee_lasso_regression(theta, X, y, model, penalty, epsilon=3.e-3, weights=None, center=0., offset=None):
    r"""Estimating equation for an approximate LASSO (least absolute shrinkage and selection operator) regressor. LASSO
    regression applies an L1-regularization through a magnitude penalty.

    Note
    ----
    As the derivative of the estimating equation for LASSO is not defined, the bread (and sandwich) cannot be used to
    estimate the variance in all settings.


    The estimating equation for the approximate LASSO linear regression is

    .. math::

        \sum_{i=1}^n \left\{(Y_i - X_i^T \theta) X_i - \lambda (1 + \epsilon) | \theta |^{\epsilon} sign(\theta)
        \right\} = 0

    where :math:`\lambda` is the penalty term.

    Here, an approximation based on the bridge penalty for the LASSO is used. For the bridge penalty, LASSO is the
    special case where :math:`\epsilon = 0`. By making :math:`\epsilon > 0`, we can approximate the LASSO. The true
    LASSO may not be possible to implement due to the existence of multiple solutions

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then :math:`\theta` will be a 1-by-3 array. The code is general to allow for an arbitrary
    number of X's (as long as there is enough support in the data).

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
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

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
    Fu WJ. (1998). Penalized regressions: the Bridge versus the LASSO. *Journal of Computational and Graphical
    Statistics*, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. *Biometrics*, 59(1), 126-132.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be greater than zero for the approximate LASSO")

    # Calling internal bridge penalized regression for implementation
    ee_bridge_regression(theta=theta,
                         X=X, y=y,
                         model=model,
                         weights=weights,
                         penalty=penalty, gamma=1+epsilon, center=center,
                         offset=offset)


def ee_elasticnet_regression(theta, X, y, model, penalty, ratio, epsilon=3.e-3, weights=None, center=0., offset=None):
    r"""Estimating equations for Elastic-Net regression. Elastic-Net applies both L1- and L2-regularization at a
    pre-specified ratio. Notice that the L1 penalty is based on an approximation. See ``ee_lasso_regression`` for
    further details on the approximation for the L1 penalty.

    Note
    ----
    As the derivative of the estimating equation for Elastic-Net is not defined, the bread (and sandwich) cannot be
    used to estimate the variance in all settings.


    The estimating equation for Elastic-Net linear regression with the approximate L1 penalty is

    .. math::

        \sum_{i=1}^n \left\{ (Y_i - X_i^T \theta) X_i - \lambda r (1 + \epsilon)
        | \theta |^{\epsilon} sign(\theta) - \lambda (1-r) \theta \right\} = 0

    where :math:`\lambda` is the penalty term and :math:`r` is the ratio for the L1 vs L2 penalty.

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then :math:`\theta` will be a 1-by-3 array. The code is general to allow for an arbitrary
    number of X's (as long as there is enough support in the data).

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
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

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
    Fu WJ. (1998). Penalized regressions: the Bridge versus the LASSO. *Journal of Computational and Graphical
    Statistics*, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. *Biometrics*, 59(1), 126-132.
    """
    # Preparation of input shapes and object types
    X, y, beta, penalty, center, offset = _prep_inputs_(X=X, y=y, theta=theta,
                                                        penalty=penalty, center=center,
                                                        offset=offset)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)    # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta) + offset)  # Generating predicted values

    # Allowing for a weighted penalized regression model
    w = generate_weights(weights=weights, n_obs=X.shape[0])

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


def ee_bridge_regression(theta, X, y, model, penalty, gamma, weights=None, center=0., offset=None):
    r"""Estimating equation for bridge penalized regression. The bridge penalty is a generalization of penalized
    regression, that includes L1 and L2-regularization as special cases.

    Note
    ----
    While the bridge penalty is defined for :math:`\gamma > 0`, the provided estimating equation only supports
    :math:`\gamma \ge 1`. Additionally, the derivative of the estimating equation is not defined when
    :math:`\gamma<2`. Therefore, the bread (and sandwich) cannot be used to estimate the variance in those settings.


    The estimating equation for bridge penalized linear regression is

    .. math::

        \sum_{i=1}^n \left\{ (Y_i - X_i^T \theta) X_i - \lambda \gamma | \theta |^{\gamma - 1} sign(\theta) \right\} = 0

    where :math:`\lambda` is the penalty term and :math:`\gamma` is a tuning parameter.

    Here, :math:`\theta` is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if
    X is a 3-by-n matrix, then :math:`\theta` will be a 1-by-3 array. The code is general to allow for an arbitrary
    number of X's (as long as there is enough support in the data).

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
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

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
    Fu WJ. (1998). Penalized regressions: the Bridge versus the LASSO. *Journal of Computational and Graphical
    Statistics*, 7(3), 397-416.

    Fu WJ. (2003). Penalized estimating equations. *Biometrics*, 59(1), 126-132.
    """
    # Preparation of input shapes and object types
    X, y, beta, penalty, center, offset = _prep_inputs_(X=X, y=y, theta=theta,
                                                        penalty=penalty, center=center,
                                                        offset=offset)

    # Determining transformation function to use for the regression model
    transform = _model_transform_(model=model)    # Looking up corresponding transformation
    pred_y = transform(np.dot(X, beta) + offset)  # Generating predicted values

    # Allowing for a weighted penalized regression model
    w = generate_weights(weights=weights, n_obs=X.shape[0])

    # Creating penalty term for ridge regression (bridge with gamma=2 is the special case of ridge)
    penalty_terms = _bridge_penalty_(theta=theta, n_obs=X.shape[0], penalty=penalty, gamma=gamma, center=center)

    # Output b-by-n matrix
    return w * (((y - pred_y) * X).T - penalty_terms[:, None])  # Score function with penalty term subtracted off


#################################################################
# Flexible Regression Estimating Equations


def ee_additive_regression(theta, X, y, specifications, model, weights=None, offset=None):
    r"""Estimating equation for Generalized Additive Models (GAMs). GAMs are an extension of generalized linear models
    that allow for more flexible specifications of relationships of continuous variables. This flexibility is
    accomplished via splines. To further control the flexibility, the spline terms are penalized.

    Note
    ----
    The implemented GAM uses L2-penalization. This penalization only applies to the generated spline terms
    (i.e., penalization decreases the 'wiggliness' of the estimated relationships).


    The estimating equation for a generalized additive linear regression model is

    .. math::

        \sum_{i=1}^n \left\{ (Y_i - f(X_i)^T \theta) f(X_i) - \lambda \theta \right\} = 0

    where :math:`\lambda` is the penalty term.

    While this looks similar to Ridge regression, there are two important differences: the function :math:`f()` and
    how :math:`\lambda` is defined. First, the function :math:`f()` denotes a general vector function. For spline terms,
    this function defines the basis functions for the splines (set via the ``specifications`` parameter). For non-spline
    terms, this is the identity function (i.e., no changes are made to the input). This setup allows for terms
    to be selectively modeled using splines (e.g., categorical features are not modeled using splines). Next, the
    penalty term, :math:`\lambda`, is only non-zero for :math:`\theta` that correspond to parameters for splines (i.e.,
    only the spline terms are penalized). This is distinction from default Ridge regression, which penalizes all terms
    in the model.

    Note
    ----
    Originally, GAMs were implemented via splines with a knot at each unique values of :math:`X`. More recently, GAMs
    use a more moderate amount of knots to improve computationally efficiency. Both versions can be implemented by
    ``ee_additive_regression`` through setting the knot locations.


    Here, :math:`\theta` is a 1-by-(b+k) array, where b is the distinct covariates included as part of X and the k
    distinct spline basis functions. For example, if X is a 2-by-n matrix with a 10-knot natural spline for the second
    column in X, then :math:`\theta` will be a 1-by-(2+9) array. The code is general to allow for an arbitrary
    number of X variables and spline knots.

    Parameters
    ----------
    theta : ndarray, list, vector
        Parameter values. Number of values should match the number of columns in the additive design matrix.
    X : ndarray, list, vector
        2-dimensional vector of n observed values for b variables.
    y : ndarray, list, vector
        1-dimensional vector of n observed values.
    specifications : list, dict, None
        A list of dictionaries that define the hyperparameters for the spline (e.g., number of knots, strength of
        penalty). For terms that should not have splines, ``None`` should be specified instead (see examples below).
        Each dictionary supports the following parameters:
        "knots", "natural", "power", "penalty"
        * knots (list): controls the position of the knots, with knots are placed at given locations. There is no
            default, so must be specified by the user.
        * natural (bool): controls whether to generate natural (restricted) or unrestricted splines.
            Default is ``True``, which corresponds to natural splines.
        * power (float): controls the power to raise the spline terms to. Default is 3, which corresponds to cubic
            splines.
        * penalty (float): penalty term (:math:`\lambda`) applied to each corresponding spline basis term. Default is 0,
            which applies no penalty to the spline basis terms.
    model : str
        Type of regression model to estimate. Options are ``'linear'`` (linear regression), ``'logistic'`` (logistic
        regression), and ``'poisson'`` (Poisson regression).
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ``None``, which assigns a weight of 1 to all observations.
    offset : ndarray, list, vector, None, optional
        A 1-dimensional offset to be included in the model. Default is None, which applies no offset term.

    Returns
    -------
    array :
        Returns a (b+k)-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_additive_regression`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_additive_regression
    >>> from delicatessen.utilities import additive_design_matrix, regression_predictions

    Some generic data to estimate a generalized additive model

    >>> n = 2000
    >>> data = pd.DataFrame()
    >>> x = np.random.uniform(-5, 5, size=n)
    >>> data['X'] = x
    >>> data['Y1'] = np.exp(x+0.5)) + np.abs(x) + np.random.normal(scale=1.5, size=n)
    >>> data['Y2'] = np.random.binomial(n=1, p=logistic.cdf(np.exp(np.sin(x+0.5)) + np.abs(x)), size=n)
    >>> data['Y3'] = np.random.poisson(lam=np.exp(np.exp(np.sin(x+0.5)) + np.abs(x)), size=n)
    >>> data['C'] = 1

    Note that ``C`` here is set to all 1's. This will be the intercept in the regression. Further, notice that the
    relationship between ``X`` and the various ``Y``'s is not linear.

    The design matrix for linear regression would be ``X = np.asarray(d[['C', 'X']])``. As the intercept is a constant,
    we only want spline terms to be applied to ``'X'`` column. To define the spline specifications, we create the
    following list

    >>> specs = [None, {"knots": [-4, -3, -2, -1, 0, 1, 2, 3, 4], "penalty": 10}]

    This tells ``ee_additive_regression`` to not generate a spline term for the first column in the input design matrix
    and to generate a default spline with knots at the specified locations and penalty of 10 for the second column in
    the input design matrix. Interally, the design matrix processing is done by the ``additive_design_matrix`` utility
    function. We can see what the output of that function looks like via

    >>> Xa_design = additive_design_matrix(X=np.asarray(data[['C', 'X']]), specifications=specs)

    That output matrix is the corresponding design matrix. Use of the ``additive_design_matrix`` utility will be
    demonstrated later for generating predictions from the estimated parameters.

    Now psi, or the stacked estimating equations.

    >>> def psi(theta):
    >>>     x, y = data[['C', 'X']], data['Y']
    >>>     return ee_additive_regression(theta=theta, X=x, y=y, model='linear', specifications=specs)

    Calling the M-estimator. Note that the input initial values depends on the number of splines being generated. To
    easily determine the number of initial values, we can use the shape of the previously generated design matrix
    (``Xa_design``)

    >>> n_params = Xa_design.shape[1]
    >>> estr = MEstimator(stacked_equations=psi, init=[0., ]*n_params)
    >>> estr.estimate()

    Inspecting the parameter estimates, variance, and confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    While all these estimates can be easily inspected, interpretting them is not easy. Instead, we can generate
    predictions from the GAM and plot the estimated regression line. To do this, we will first create a new design
    matrix where ``'X'`` is evenly spaced over a range of values

    >>> p = pd.DataFrame()
    >>> p['X'] = np.linspace(-5, 5, 200)
    >>> p['C'] = 1
    >>> Xa_pred = additive_design_matrix(X=np.asarray(p[['C', 'X']]), specifications=specs)

    To generate the predicted values of Y (and the corresponding confidence intervals), we do the following

    >>> yhat = regression_predictions(Xa_pred, theta=estr.theta, covariance=estr.variance)

    For further details, see the ``regression_predictions`` utility function documentation.

    Other optional specifications are available for the spline terms. Here, we will specify an unrestricted quadratic
    spline with a penalty of 5.5 for the second column of the design matrix.

    >>> specs = [None, {"knots": [-4, -2, 0, 2, 4], "natural": False, "power": 2, "penalty": 5.5}]

    See the documentation of ``additive_design_matrix`` for additional examples of how to specify the additive design
    matrix and corresponding splines.

    Lastly, knots could be placed at each unique observation via

    >>> specs = [None, {"knots": np.unique(data['X']), "penalty": 500}]

    Note that the penalty is increased here (as the number of knots has dramatically increased).

    A GAM for a binary outcome (i.e., logistic regression) can be implemented as follows

    >>> specs = [None, {"knots": [-4, -3, -2, -1, 0, 1, 2, 3, 4], "penalty": 10}]
    >>> Xa_design = additive_design_matrix(X=np.asarray(data[['C', 'X']]), specifications=specs)
    >>> n_params = Xa_design.shape[1]

    >>> def psi(theta):
    >>>     x, y = data[['C', 'X']], data['Y2']
    >>>     return ee_additive_regression(theta=theta, X=x, y=y, model='logistic', specifications=specs)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.]*n_params)
    >>> estr.estimate(solver='lm', maxiter=5000)

    A GAM for count outcomes (i.e., Poisson regression) can be implemented as follows

    >>> specs = [None, {"knots": [-4, -3, -2, -1, 0, 1, 2, 3, 4], "penalty": 10}]
    >>> Xa_design = additive_design_matrix(X=np.asarray(data[['C', 'X']]), specifications=specs)
    >>> n_params = Xa_design.shape[1]

    >>> def psi(theta):
    >>>     x, y = data[['C', 'X']], data['Y3']
    >>>     return ee_additive_regression(theta=theta, X=x, y=y, model='poisson', specifications=specs)

    >>> estr = MEstimator(stacked_equations=psi, init=[0.]*n_params)
    >>> estr.estimate(solver='lm', maxiter=5000)

    Weighted models can be estimated by specifying the optional ``weights`` argument.

    References
    ----------
    Fu WJ. (2003). Penalized estimating equations. *Biometrics*, 59(1), 126-132.

    Hastie TJ. (2017). Generalized additive models. *In Statistical models in S* (pp. 249-307). Routledge.

    Marx BD, & Eilers PH. (1998). Direct generalized additive modeling with penalized likelihood.
    *Computational Statistics & Data Analysis*, 28(2), 193-209.

    Wild CJ, & Yee TW. (1996). Additive extensions to generalized estimating equation methods.
    *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(4), 711-725.
    """
    # Compute the design matrix for the additive model
    Xa, penalty = additive_design_matrix(X=X,                               # Create the additive design matrix
                                         specifications=specifications,     # ... with the provided specifications
                                         return_penalty=True)               # ... and return corresponding penalties

    # Apply spline-penalized regression
    return ee_bridge_regression(theta=theta, y=y, X=Xa,                     # Call bridge reg with additive design
                                model=model, penalty=penalty,               # ... matrix and processed penalties
                                gamma=2, weights=weights,                   # ... and set gamma=2 (Ridge reg)
                                center=0.,                                  # ... with splines ALWAYS penalized to zero
                                offset=offset)                              # ... and provided offset


#################################################################
# Utility functions for regression equations

def _prep_inputs_(X, y, theta, penalty=None, center=None, offset=None, reshape_y=True):
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
    reshape_y : bool, optional
        Whether to reshape the y array. All regression functions, besides multinomial logit, expect the reshaped
        array. Therefore, default is True, with only ``ee_mlogit`` modifying this parameter.

    Returns
    -------
    transformed parameters
    """
    X = np.asarray(X)                       # Convert to NumPy array
    if reshape_y:                           # Convert to NumPy array
        y = np.asarray(y)[:, None]          # ... ensure correct shape for matrix algebra
    else:                                   # ... otherwise
        y = np.asarray(y)                   # ... keep original shape
    beta = np.asarray(theta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra

    # Logic to determine the offset variable if requested
    if offset is None:                           # When offset is None
        offset = 0                               # ... modify by adding a zero (i.e., no mod)
    else:                                        # Otherwise
        offset = np.asarray(offset)[:, None]     # ... ensure that a NumPy array is passed forward

    # What to return if penalty is or is not given
    if penalty is None:                     # Return the transformed objects
        return X, y, beta, offset
    else:                                   # Convert penalty term then return all
        penalty = np.asarray(penalty)       # Convert to NumPy array
        center = np.asarray(center)         # Convert to NumPy array
        return X, y, beta, penalty, center, offset


def _model_transform_(model, assert_linear_model=False):
    """Internal use function to simplify the checking procedure for the model form to use. Takes the input string and
    returns the corresponding function for the variable transformation.

    Parameters
    ----------
    model : str
        Model identifier to calculate the transformation for
    assert_linear_model : bool, optional
        Flag to assert whether only a linear model is supported.

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


def _inverse_link_(betax, link):
    # Distributions not implemented: power, inverse power
    if link == 'identity':
        py = identity(betax)                    # Inverse link
        dpy = 1                                 # Derivative of inverse link
    elif link == 'log':
        py = np.exp(betax)                      # Inverse link
        dpy = py                                # Derivative of inverse link
    elif link in ['logistic', 'logit']:
        py = inverse_logit(betax)               # Inverse link
        dpy = py * (1 - py)                     # Derivative of inverse link
    elif link == 'inverse':
        py = 1 / betax                          # Inverse link
        dpy = -1 / (betax**2)                   # Derivative of inverse link
    elif link == 'loglog':
        py = np.exp(-1*np.exp(-betax))          # Inverse link
        dpy = -np.exp(-betax - np.exp(-betax))  # Derivative of inverse link
    elif link == 'cloglog':
        py = 1 - np.exp(-1*np.exp(betax))       # Inverse link
        dpy = np.exp(betax - np.exp(betax))     # Derivative of inverse link
    elif link == 'probit':
        # py = norm.cdf(betax)                  # Inverse link
        # dpy = norm.pdf(betax)                 # Derivative of inverse link
        py = standard_normal_cdf(x=betax)       # Inverse link
        dpy = standard_normal_pdf(x=betax)      # Derivative of inverse link
    elif link in ['cauchit', 'cauchy']:
        # py = cauchy.cdf(betax)                # Inverse link
        # dpy = cauchy.pdf(betax)               # Derivative of inverse link
        py = (1/np.pi)*np.arctan(betax) + 0.5   # Inverse link
        dpy = 1 / (np.pi*(1 + betax**2))        # Derivative of inverse link (by-hand)
    elif link in ['square_root', 'sqrt']:
        py = betax**2                           # Inverse link
        dpy = 2 * betax                         # Derivative of inverse link
    else:
        raise ValueError("invalid link")
    return py, dpy


def _distribution_variance_(dist, mu, hyperparameter=None, alpha=None):
    if dist in ['normal', 'gaussian']:
        v = 1
    elif dist == 'poisson':
        v = mu
    elif dist in ['binomial', 'bin', 'bernoulli']:
        v = mu - mu**2
    elif dist == 'gamma':
        v = mu**2
    elif dist in ['negative_binomial', 'nb']:
        v = mu + alpha*(mu**2)
    elif dist in ['inverse_normal', 'inverse_gaussian']:
        v = mu**3
    elif dist == 'tweedie':
        if 0 > hyperparameter:
            raise ValueError("The Tweedie distribution requires the "
                             "hyperparameter to be non-negative, i.e., >0.")
        v = mu**hyperparameter
    else:
        raise ValueError("invalid distribution")
    return v


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
