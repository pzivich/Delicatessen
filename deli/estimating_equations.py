import numpy as np
from deli.utilities import logit, inverse_logit

#################################################################
# Basic Estimating Equations


def ee_mean(theta, y):
    """Default stacked estimating equation for the mean. The estimating equation for the mean is

    .. math::

        \sum_i^n \psi(Y_i, \theta_1) = \sum_i^n Y_i - \theta_1 = 0

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : vector
        Theta in the case of the mean consists of a single value. Therefore, an initial value like the form of
        [0, ] is recommended.
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the mean).

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_mean` should be done similar to the following

    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_mean

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean(theta=theta, y=y_dat)

    Calling the M-estimation procedure

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Output 1-by-n array
    return np.asarray(y) - theta     # Estimating equation for the mean


def ee_mean_robust(theta, y, k):
    """Default stacked estimating equation for robust mean (robust location) estimator. The estimating equation for
    the robust mean is

    .. math::

        \sum_i^n \psi(Y_i, \theta_1) = \sum_i^n Y^*_i - \theta_1 = 0

    where Y* is bounded between k and -k.

    Note
    ----
    Since psi is non-differentiable at k or -k, it must be assumed that the mean is sufficiently far from k. Otherwise,
    difficulties might arise in the variance calculation.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : vector
        Theta in the case of the robust mean consists of a single value. Therefore, an initial value like the form of
        [0, ] is recommended.
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the robust mean).
    k : int, float
        Value to set the maximum upper and lower bounds on the observed values.

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_mean_robust` should be done similar to the following

    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_mean_robust

    Some generic data to estimate the mean for

    >>> y_dat = [-10, 1, 2, 4, 1, 2, 3, 1, 5, 2, 33]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean_robust(theta=theta, y=y_dat, k=9)

    Calling the M-estimation procedure

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    var = np.asarray(y)                   # Convert y to NumPy array
    var = np.where(var > k, k, var)       # Apply the upper bound
    var = np.where(var < -k, -k, var)     # Apply the lower bound

    # Output 1-by-n array
    return var - theta                    # Estimating equation for robust mean


def ee_mean_variance(theta, y):
    """Default stacked estimating equation for mean and variance. The estimating equations for the mean and
     variance are

    .. math::

        \sum_i^n \psi_1(Y_i, \theta_1) = \sum_i^n Y_i - \theta_1 = 0

        \sum_i^n \psi_2(Y_i, \theta_1) = \sum_i^n (Y_i - \theta_1)^2 - \theta_2 = 0

    Unlike `ee_mean`, theta consists of 2 elements. The output covariance matrix will also provide estimates for each
    of the theta values.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Parameters
    ----------
    theta : vector
        Theta in this case consists of two values. Therefore, initial values like the form of [0, 0] is recommended.
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior when attempting to calculate the mean).

    Returns
    -------
    array :
        Returns a 2-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_mean_variance` should be done similar to the following

    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_mean_variance

    Some generic data to estimate the mean for

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_mean_variance(theta=theta, y=y_dat)

    Calling the M-estimation procedure (note that `init` has 2 values now).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    In this example, `mestimation.theta[1]` and `mestimation.asymptotic_variance[0][0]` are expected to be equal.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    # Output 2-by-n matrix
    return (y - theta[0],                  # Estimating equation for mean
            (y - theta[0])**2 - theta[1])  # Estimating equation for variance


#################################################################
# Regression Estimating Equations


def ee_linear_regression(theta, X, y):
    """Default stacked estimating equation for linear regression. The estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i = 0

    Here, theta is a 1-by-b array, where b is the distinct covariates included as part of X. For example, if X is a
    3-by-n matrix, then theta will be a 1-by-3 array. The code is general to allow for an arbitrary number of X's (as
    long as there is enough support in the data).

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Here, theta corresponds to the coefficients in a linear regression model

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely OLS). Therefore, optimization of OLS via a separate functionality can be done then those
    estimated parameters are fed forward as the initial values (which should result in a more stable optimization).

    Parameters
    ----------
    theta : vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. No missing data should be included (missing data may cause unexpected
        behavior).

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_linear_regression` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_linear_regression

    Some generic data to estimate a linear regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    >>> data['C'] = 1

    Note that `C` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_linear_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])


    Calling the M-estimation procedure (note that `init` has 3 values now, since `X.shape[1]` is equal to 3).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    X = np.asarray(X)                       # Convert to NumPy array
    y = np.asarray(y)[:, None]              # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]       # Convert to NumPy array and ensure correct shape for matrix algebra

    # Output b-by-n matrix
    return ((y -                            # Speedy matrix algebra for regression
             np.dot(X, beta))               # ... linear regression requires no transfomrations
            * X).T                          # ... multiply by coefficient and transpose for correct orientation


def ee_logistic_regression(theta, X, y):
    """Default stacked estimating equation for logistic regression. The estimating equation is

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
    user-defined functions are defined as `psi`.

    Here, theta corresponds to the coefficients in a logistic regression model, and therefore are the log-odds.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely logistic regression). Therefore, optimization of logistic regression via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).

    Parameters
    ----------
    theta : vector
        Theta in this case consists of b values. Therefore, initial values should consist of the same number as the
        number of columns present. This can easily be accomplished generally by `[0, ] * X.shape[1]`.
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).

    Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_logistic_regression` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_logistic_regression

    Some generic data to estimate a logistic regresion model

    >>> n = 500
    >>> data = pd.DataFrame()
    >>> data['X'] = np.random.normal(size=n)
    >>> data['Z'] = np.random.normal(size=n)
    >>> data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
    >>> data['C'] = 1

    Note that `C` here is set to all 1's. This will be the intercept in the regression.

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_logistic_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'])


    Calling the M-estimation procedure (note that `init` has 3 values now, since `X.shape[1]` is equal to 3).

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0., 0.,])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    X = np.asarray(X)                      # Convert to NumPy array
    y = np.asarray(y)[:, None]             # Convert to NumPy array and ensure correct shape for matrix algebra
    beta = np.asarray(theta)[:, None]      # Convert to NumPy array and ensure correct shape for matrix algebra

    # Output b-by-n matrix
    return ((y -                                # Speedy matrix algebra for regression
             inverse_logit(np.dot(X, beta)))    # ... inverse-logit transformation of predictions
            * X).T                              # ... multiply by coefficient and transpose for correct orientation


#################################################################
# Survival Analysis Estimating Equations


def ee_kaplan_meier(theta, time, delta, adjust=1e-5):
    """Default stacked estimating equation for non-parametric estimator of the risk function. While the form of the
    estimator is not the Kaplan-Meier, this estimator is exactly equivalent to the Kaplan-Meier (as formally shown in
    Satten & Datta 2001). The estimating equations are the following equation stacked for each of the times

    .. math::

        \sum_i^n \psi_t(T_i, \delta_i, \theta) = \sum_i^n \frac{\delta_i I(T_i <= t)}{\pi_i(T_i)} = 0

    with the following probability weight

    .. math::

        \pi_i(t) = \widehat{\Pr}(C>t)

    Here, pi is a marginal inverse probability of censoring weight. See Satten & Datta 2001 for more explicit details
    on how this approach works.

    The above estimating equation is defined for a particular event time. In the general case, psi will consist of s
    estimating equations, where s is the number of unique *event* times observed. Therefore, the dimension of theta
    depends on the number of unique event times that are observed in the data. This is distinct from the other
    estimating equations, and can make it more difficult to correctly specify `init`. See recommendations on
    specifying `init` below.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Here, theta corresponds to the estimated risk function in the sorted unique event times observed in the data.

    Parameters
    ----------
    theta : vector
        Theta in this case consists of s values, where s denotes the number of unique event times. Therefore, initial
        values in M-estimation should consists of the same number as the number of unique event times. This can easily
        be accomplished generally by `[0.5, ] * np.unique(time[delta == 1]).shape[0]`
    time : vector
        1-dimensional vector of n observed values for time. No missing data should be included (missing data
        may cause unexpected behavior).
    delta : vector
        1-dimensional vector of n observed event indicators. Here, 1 denotes the event of interest occured and 0 denotes
        censoring. No missing data should be included (missing data may cause unexpected behavior).
    adjust : float, int, optional
        Due to the underlying assumption in all of survival analysis that if event and censor are tied, then the event
        always occurs first, potential ties in time between events and censoring need to be broken. This is done
        internally by shifting all censored observations by a small amount of time. The default shift is `1e-5`, which
        should be finer than the observed time scale, but this can be made smaller by specifying this parameter.

    Returns
    -------
    array :
        Returns a s-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_kaplan_meier` should be done similar to the following

    >>> import pandas as pd
    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_kaplan_meier

    Some generic survival data

    >>> d = pd.DataFrame()
    >>> d['t'] = [1, 2, 3, 2, 1, 4, 2, 5, 5, 5]
    >>> d['d'] = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>         return ee_kaplan_meier(theta=theta, time=d['t'], delta=d['d'])

    Calling the M-estimation procedure. Here, we present an easy way to define the initial values used for the
    optimization of theta. `np.unique(d.loc[d['d'] == 1, 't'])` finds all the unique event times, then we extract the
    total number of unique events via `.shape[0]`. This procedure will scale for arbitrary numbers of unique event
    times.

    >>> initial_vals = [0.5, ] * np.unique(d.loc[d['d'] == 1, 't']).shape[0]
    >>> mestimation = MEstimator(stacked_equations=psi, init=initial_vals)
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    References
    ----------
    Satten GA, & Datta S. (2001). The Kaplan–Meier estimator as an inverse-probability-of-censoring weighted average.
    The American Statistician, 55(3), 207-210.
    """
    # Ensuring correct typing
    time = np.asarray(time)                      # Convert to NumPy array
    delta = np.asarray(delta)                    # Convert to NumPy array

    # Removing any ties between events and censoring times
    time = np.where(delta == 1,                  # For all events
                    time,                        # ... keep the observed time
                    time + adjust)               # ... for censored observations add the time `adjust`

    # All unique event times
    event_times = time[delta == 1]                        # Finding all event times
    unique_event_times = np.unique(event_times)           # Finding all unique event times
    unique_times = np.append([0, ], np.unique(time))      # Finding all unique times

    # Estimate the marginal probability of remaining uncensored (pi) via the Kaplan-Meier
    pi_ti = []                                            # Empty storage
    for tau in unique_times:                              # For each unique event
        censored = np.sum((delta == 0) * (time == tau))   # ...censored at t
        nk = np.sum(time >= tau)                          # ...observations making it to or past t
        pi_ti.append(1 - censored / nk)                   # ... observations remaining at T
    pi = np.cumprod(pi_ti)                                # Cumulative product
    pi_i = []                                             # Estimating the individual pi values
    for t in time:                                        # For each individual / observation
        pi_i.append(pi[t <= unique_times][0])             # ... extract pi at the closest time
    pi_i = np.where(delta == 0, 1, pi_i)                  # Replacing pi with 1 for censored observations
    # This last line doesn't impact later calculations, but prevents division-by-zero errors

    # Stacked estimating equations for each t (in unique_event_times) and i (in rows of data)
    i_est_eqs = []                                                        # Creating storage for estimating equations
    for t_index in range(unique_event_times.shape[0]):                    # For each unique event time (in order)
        preds = (delta * (time <= unique_event_times[t_index])) / pi_i    # ... calculate vector of predicted values
        i_est_eqs.append(preds - theta[t_index])                          # ... subtract off theta[t]

    # Output s-by-n matrix
    return np.array(i_est_eqs)                            # Converting list of lists to NumPy array


def ee_cox_ph_model(theta, X, time, delta):
    """Default stacked estimating equation for the semi-parametric Cox proportional hazards model. The estimating
    equation is

    .. math::

        \sum_i^n \Delta_i \left( X_i - \frac{\sum_j^n X_j exp(\beta X_j) I(T_j >= T_i)}{\sum_j^n exp(\beta X_j) I(T_j >= T_i)} \right) = 0

    This is the score function for the Cox proportional hazards model. A more in-depth study of this estimating equation
    is provided on pages 113-125 of Tsiatis 2006.

    Note
    ----
    The form above requires that no two events happen at the exact same time. While there are existing methods to
    handle ties, the estimating equations here expect the user to manually break / prevent any ties.

    Here, theta corresponds to the estimated log-transformed hazard ratios for the covariates.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely Cox models). Therefore, optimization of Cox model via a separate functionality can be done
    then those estimated parameters are fed forward as the initial values (which should result in a more stable
    optimization).

    Parameters
    ----------
    theta : array, list
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    X : array, list, Series
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior). Additionally, the Cox model has not intercept, so NO intercept term should be
        provided.
    time : array, vector, Series
        1-dimensional vector of n observed values for time. No missing data should be included (missing data
        may cause unexpected behavior).
    delta : vector
        1-dimensional vector of n observed event indicators. Here, 1 denotes the event of interest occured and 0 denotes
        censoring. No missing data should be included (missing data may cause unexpected behavior).

        Returns
    -------
    array :
        Returns a b-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_cox_ph_model` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_cox_ph_model

    Some generic survival data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['w'] = np.random.normal(size=n)
    >>> d['x'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> d['T'] = np.exp(4.60837 + np.log(3)*d['x'] - np.log(2)*d['w']) * np.random.weibull(a=0.5, size=n)
    >>> d['C'] = np.exp(2.72811) * np.random.exponential(size=n)
    >>> d['C'] = np.where(d['C'] > 365, 365, d['C'])
    >>> d['t'] = np.min(d[['T', 'C']], axis=1)
    >>> d['d'] = np.where(d['t'] == d['T'], 1, 0)

    Defining psi, or the stacked estimating equations

    >>> def psi(theta):
    >>>     return ee_cox_ph_model(theta, X=d[['x', 'w']], time=d['t'], delta=d['d'])

    Calling the M-estimation procedure. Since `X` is 2-by-n here, the initial values should be of length 2.

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, 0])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    References
    ----------
    Tsiatis AA. (2006). Semiparametric theory and missing data. Springer, New York, NY.
    """
    # Ensuring correct typing
    X = np.asarray(X)                            # Convert to NumPy array
    time = np.asarray(time)                      # Convert to NumPy array
    delta = np.asarray(delta)                    # Convert to NumPy array

    # Calculating estimating equations
    vals = []                                          # Storage for output
    e_coef = np.exp(np.dot(theta, X.T))                # Exponential coefficients
    for i in range(X.shape[0]):                        # For each observation
        den = e_coef * np.asarray(time >= time[i])     # ... calculating denominator with coefficients and indicator
        num = X.T * den                                # ... calculating numerator with X and the denominator
        v = delta[i] * (X[i] - (np.sum(num, axis=1)    # ... calculating overall psi value for individual i
                                / np.sum(den)))
        # Logic to avoid some issues
        if X.shape[1] == 1:                            # if 1-dimensional theta
            vals.append(v[0])                          # ... append only the 'first' value (avoids .shape issues later)
        else:                                          # otherwise for >1-dimensional theta
            vals.append(v)                             # ... append all the values

    # Output b-by-n matrix
    return np.asarray(vals).T                          # Converting list of lists to array and transpose


#################################################################
# Causal Inference (ATE) Estimating Equations


def ee_gformula(theta, X, y, treat_index, force_continuous=False):
    """Default stacked estimating equation for the parametric g-formula in the time-fixed setting. The parameter(s) of
    interest is the risk difference, with potential interest in the underlying risk or mean functions. For continuous
    Y, the linear regression estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - X_i^T \theta) X_i = 0

    and for logistic regression, the estimating equation is

    .. math::

        \sum_i^n \psi(Y_i, X_i, \theta) = \sum_i^n (Y_i - expit(X_i^T \theta)) X_i = 0

    By default, `ee_gformula` detects whether `y` is all binary (zero or one), and applies logistic regression if that
    happens. See the parameters for more details.

    For the implementation of the g-formula, stacked estimating equations are also used for the risk / mean had
    everyone been given treatment=1, the risk / mean had everyone been given treatment=0, and the risk / mean
    difference between those two risks. Respectively, those estimating equations look like

    .. math::

        \sum_i^n \psi_1(Y_i, X_i, \theta_1) = \sum_i^n g(\hat{Y}_i) - \theta_1 = 0

    .. math::

        \sum_i^n \psi_0(Y_i, X_i, \theta_0) = \sum_i^n g(\hat{Y}_i) - \theta_0 = 0

    .. math::

        \sum_i^n \psi_0(Y_i, X_i, \theta_0) = \sum_i^n \theta_1 - \theta_0 = 0

    Here, the function g() is a generic function. If linear regression was used, g() is the identity function. If
    logistic regression was used, g() is the expit or inverse-logit function.

    Due to these 3 extra values, the length of the theta vector is b+3, where b is the number of parameters in the
    regression model.

    Note
    ----
    All provided estimating equations are meant to be wrapped inside a user-specified function. Throughtout, these
    user-defined functions are defined as `psi`.

    Here, theta corresponds to a variety of different quantities. The *first* value in theta vector is the risk / mean
    difference (or average treatment effect), the *second* is the risk / mean had everyone been given treatment=0, the
    *third* is the risk / mean had everyone been given treatment=1. The remainder of the parameters correspond to the
    regression model coefficients, in the order input.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely regression models). Therefore, optimization of the regression model via a separate
    functionality can be done then those estimated parameters are fed forward as the initial values (which should
    result in a more stable optimization).

    Parameters
    ----------
    theta : array, list
        Array of parameters to estimate. For the Cox model, corresponds to the log hazard ratios
    X : vector
        2-dimensional vector of n observed values for b variables. No missing data should be included (missing data
        may cause unexpected behavior).
    y : vector
        1-dimensional vector of n observed values. The Y values should all be 0 or 1. No missing data should be
        included (missing data may cause unexpected behavior).
    treat_index : int
        Column index for the treatment vector.
    force_continuous : bool, optional
        Option to force the use of linear regression despite detection of a binary variable.

    Returns
    -------
    array :
        Returns a (3+b)-by-n NumPy array evaluated for the input theta and y

    Examples
    --------
    Construction of a estimating equation(s) with `ee_cox_ph_model` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from deli import MEstimator
    >>> from deli.estimating_equations import ee_gformula

    Some generic survival data

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['W'] = np.random.binomial(1, p=0.5, size=n)
    >>> d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    >>> d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    >>> d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    >>> d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    >>> d['C'] = 1

    Defining psi, or the stacked estimating equations. Note that 'A' is the treatment of interest, so `treat_index` is
    set to 1 (compared to input `X`).

    >>> def psi(theta):
    >>>     return ee_gformula(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

    Calling the M-estimation procedure. Since `X` is 3-by-n here and g-formula has 3 additional parameters, the initial
    values should be of length 3+3=6. In general, it will be best to start with [0., 0.5, 0.5, ...] as the initials for
    the risk parameters. This will start the initial at the exact middle value for each of those parameters. For the
    regression coefficients, those can be set as zero, or if there is difficulty in simultaneous optimization,
    coefficient estimates from outside `MEstimator` can be provided as inputs.

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0., 0.])
    >>> mestimation.estimate()

    Inspecting the parameter estimates and the variance

    >>> mestimation.theta
    >>> mestimation.variance

    More specifically, the average treatment effect and its variance are

    >>> mestimation.theta[0]
    >>> mestimation.variance[0, 0]

    The risk / mean had all been given treatment=1

    >>> mestimation.theta[1]
    >>> mestimation.variance[1, 1]

    The risk / mean had all been given treatment=0

    >>> mestimation.theta[2]
    >>> mestimation.variance[2, 2]

    References
    ----------
    Snowden JM, Rose S, & Mortimer KM. (2011). Implementation of G-computation on a simulated data set: demonstration
    of a causal inference technique. American Journal of Epidemiology, 173(7), 731-738.
    """
    # Ensuring correct typing
    X = np.asarray(X)                            # Convert to NumPy array
    y = np.asarray(y)                            # Convert to NumPy array
    beta = theta[3:]                       # Extracting out theta's for the regression model

    # Checking outcome variable type
    if np.isin(y, [0, 1]).all() and not force_continuous:
        continuous = False
    else:
        continuous = True

    if continuous:
        # Linear regression parameters
        preds_reg = ee_linear_regression(theta=beta,
                                         X=X,
                                         y=y)
        # Calculating Y(a=0)
        X[:, treat_index] = 0
        ya0 = np.dot(X, beta) - theta[2]
        # Calculating Y(a=1)
        X[:, treat_index] = 1
        ya1 = np.dot(X, beta) - theta[1]
    else:
        # Logistic regression parameters
        preds_reg = ee_logistic_regression(theta=beta,
                                           X=X,
                                           y=y)
        # Calculating Y(a=0)
        X[:, treat_index] = 0
        ya0 = inverse_logit(np.dot(X, beta)) - theta[2]
        # Calculating Y(a=1)
        X[:, treat_index] = 1
        ya1 = inverse_logit(np.dot(X, beta)) - theta[1]

    # Calculating Y(a=1) - Y(a=0)
    ate = X[:, 0] * (theta[1] - theta[2]) - theta[0]
    return np.vstack((ate,
                      ya1[None, :],
                      ya0[None, :],
                      preds_reg))
