#####################################################################################################################
# Estimating functions for measurement error
#####################################################################################################################

import numpy as np

from delicatessen.utilities import inverse_logit
from delicatessen.estimating_equations.regression import ee_regression
from delicatessen.estimating_equations.processing import generate_weights


def ee_rogan_gladen(theta, y, y_star, r, weights=None):
    r"""Estimating equation for the Rogan-Gladen correction for mismeasured *binary* outcomes. This estimator uses
    external data to estimate the sensitivity and specificity, and then uses those external estimates to correct the
    estimated proportion. The general form of the estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            \mu \times \left\{ \alpha + \beta - 1 \right\} - \left\{ \mu^* + \beta - 1 \right\} \\
            R_i (Y_i^* - \mu^*) \\
            (1-R_i) Y_i \left\{ Y^*_i - \beta \right\} \\
            (1-R_i) (1-Y_i) \left\{ (1 - Y^*_i) - \alpha \right\} \\
        \end{bmatrix}
        = 0

    where :math:`Y` is the true value of the outcome, :math:`Y^*` is the mismeasured value of the outcome, :math:`R` is
    the indicator for the main study data, :math:`\mu` is the corrected mean, :math:`\mu^*` is the mismeasured mean in
    the main study data, :math:`\beta` is the sensitivity, and :math:`\alpha` is the specificity. The first
    estimating equation is the corrected proportion, the second is the naive proportion, the third is for sensitivity,
    and the fourth for specificity.

    Here, ``theta`` is a 1-by-4 array.

    Note
    ----
    The Rogan-Gladen estimator may provide corrected proportions outside of :math:`[0,1]` when
    :math:`\alpha + \beta \le 1`.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 4 values.
    y : ndarray, list, vector
        1-dimensional vector of `n` observed values. These are the gold-standard :math:`Y` measurements in the external
        sample. All values should be either 0 or 1, and be non-missing among those with :math:`R=0`.
    y_star : ndarray, list, vector
        1-dimensional vector of `n` observed values. These are the mismeasured :math:`Y` values. All values should be
        either 0 or 1, and be non-missing among all observations.
    r : ndarray, list, vector
        1-dimensional vector of `n` indicators regarding whether an observation was part of the external validation
        data. Indicator should designate if observations are the main data.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a 4-by-`n` NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_rogan_gladen`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_rogan_gladen

    Replicating the published example from Cole et al. (2023).

    >>> d = pd.DataFrame()
    >>> d['Y_star'] = [0, 1] + [0, 1, 0, 1]
    >>> d['Y'] = [np.nan, np.nan] + [0, 0, 1, 1]
    >>> d['S'] = [1, 1] + [0, 0, 0, 0]
    >>> d['n'] = [270, 680] + [71, 18, 38, 203]
    >>> d = pd.DataFrame(np.repeat(d.values, d['n'], axis=0), columns=d.columns)

    Applying the Rogan-Gladen correction to this example

    >>> def psi(theta):
    >>>     return ee_rogan_gladen(theta=theta, y=d['Y'],
    >>>                            y_star=d['Y_star'], r=d['S'])

    Notice that ``y`` corresponds to the gold-standard outcomes (only available where R=0), ``y_star`` corresponds to
    the mismeasured covariate data (available for R=1 and R=0), and ``r`` corresponds to the indicator for the main
    data source. Now we can call the M-Estimator.

    >>> estr = MEstimator(psi, init=[0.5, 0.5, .75, .75])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    The corrected proportion is

    >>> estr.theta[0]

    Inverse probability weights can be used through the ``weights`` argument. See the applied examples for a
    demonstration.

    References
    ----------
    Cole SR, Edwards JK, Breskin A, Rosin S, Zivich PN, Shook-Sa BE, & Hudgens MG. (2023). Illustration of 2 Fusion
    Designs and Estimators. *American Journal of Epidemiology*, 192(3), 467-474.

    Rogan WJ & Gladen B. (1978). Estimating prevalence from the results of a screening test.
    *American Journal of Epidemiology*, 107(1), 71-76.

    Ross RK, Zivich PN, Stringer JSA, & Cole SR. (2024). M-estimation for common epidemiological measures: introduction
    and applied examples. *International Journal of Epidemiology*, 53(2), dyae030.
    """
    # Processing inputs
    y = np.asarray(y)                           # Convert to NumPy array
    y_star = np.asarray(y_star)                 # Convert to NumPy array
    r = np.asarray(r)                           # Convert to NumPy array
    y = np.where(r == 1, -999, y)               # Removing NaN (or any other indicators) for Y in main
    w = generate_weights(weights=weights,       # Formatting weights
                         n_obs=y.shape[0])      # ... and processing if None
    mu = theta[0]                               # Parameter for corrected proportion
    mu_star = theta[1]                          # Means (corrected and naive)
    sens = theta[2]                             # Parameters for the sensitivity model
    spec = theta[3]                             # Parameters for the specificity model

    # Estimating equation for naive (mismeasured) mean
    ee_naive_mean = r * w * (y_star - mu_star)  # Naive mean among obs

    # Estimating equation for sensitivity
    ee_sens = (y_star - sens) * (1-r) * y * w

    # Estimating equation for specificity
    ee_spec = (1 - y_star - spec) * (1-r) * (1-y) * w

    # Estimating equation for the Rogan-Gladen correction
    ee_corr_mean = mu*(sens + spec - 1) - (mu_star + spec - 1) * np.ones(y.shape[0])

    # Returning stacked estimating equations
    return np.vstack([ee_corr_mean,             # Corrected mean
                      ee_naive_mean,            # Naive mean
                      ee_sens,                  # Sensitivity model parameters
                      ee_spec])                 # Specificity model parameters


def ee_rogan_gladen_extended(theta, y, y_star, r, X, weights=None):
    r"""Estimating equation for the extended Rogan-Gladen correction for mismeasured *binary* outcomes. This estimator
    uses external data to estimate the sensitivity and specificity conditional on covariates, and then uses those
    external estimates to correct the estimated proportion. The general form of the estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            R_i \times \left\{ \frac{Y^* + m(X_i; \beta) - 1}{m(X_i; \alpha) + m(X_i; \beta) - 1}  - \mu \right\} \\
            (1-R_i) Y_i \left\{ Y^*_i - m(X_i; \beta) \right\} X_i^T \\
            (1-R_i) (1 - Y_i) \left\{ (1 - Y^*_i) - m(X_i; \beta) \right\} X_i^T \\
        \end{bmatrix}
        = 0

    where :math:`Y` is the true value of the outcome, :math:`Y^*` is the mismeasured value of the outcome. The first
    estimating equation is the corrected proportion, the second is for sensitivity, and the third for specificity.

    If :math:`X` is of dimension :math:`p`, then ``theta`` is a 1-by-(1+2`p`) array. Note that the design matrix is
    shared across the sensitivity and specificity models.

    Note
    ----
    The Rogan-Gladen estimator may provide corrected proportions outside of :math:`[0,1]` when
    :math:`\alpha + \beta \le 1`, or the addition of sensitivity and specificity is less than or equal to one.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 1+2`p` values.
    y : ndarray, list, vector
        1-dimensional vector of `n` observed values. These are the gold-standard :math:`Y` measurements in the external
        sample. All values should be either 0 or 1, and be non-missing among those with :math:`R=0`.
    y_star : ndarray, list, vector
        1-dimensional vector of `n` observed values. These are the mismeasured :math:`Y` values. All values should be
        either 0 or 1, and be non-missing among all observations.
    r : ndarray, list, vector
        1-dimensional vector of `n` indicators regarding whether an observation was part of the external validation
        data. Indicator should designate if observations are the main data.
    X : ndarray, list, vector
        2-dimensional vector of a design matrix for the sensitivity and specificity models.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a (1+2`p`)-by-`n` NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_rogan_gladen_extended`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_rogan_gladen_extended

    Replicating the example from Cole et al. (2023).

    >>> d = pd.DataFrame()
    >>> d['Y_star'] = [0, 1] + [0, 1, 0, 1]
    >>> d['Y'] = [np.nan, np.nan] + [0, 0, 1, 1]
    >>> d['S'] = [1, 1] + [0, 0, 0, 0]
    >>> d['n'] = [270, 680] + [71, 18, 38, 203]
    >>> d = pd.DataFrame(np.repeat(d.values, d['n'], axis=0), columns=d.columns)
    >>> d['C'] = 1

    Applying the Rogan-Gladen correction to this example

    >>> def psi(theta):
    >>>     return ee_rogan_gladen_extended(theta=theta, y=d['Y'],
    >>>                                     y_star=d['Y_star'],
    >>>                                     X=d[['C', ]], r=d['S'])

    Notice that ``y`` corresponds to the gold-standard outcomes (only available where R=0), ``y_star`` corresponds to
    the mismeasured covariate data (available for R=1 and R=0), and ``r`` corresponds to the indicator for the main
    data source. Now we can call the M-Estimator.

    >>> estr = MEstimator(psi, init=[0.5, 1., 1.])
    >>> estr.estimate(solver='lm')

    Inspecting the parameter estimates, variance, and 95% confidence intervals

    >>> estr.theta
    >>> estr.variance
    >>> estr.confidence_intervals()

    Note
    ----
    The sensitivity and specificity in ``ee_rogan_gladen_extended`` correspond to the logit transformations, unlike
    ``ee_rogan_gladen`` which returns the sensitivity and specificity directly.


    The corrected proportion is

    >>> estr.theta[0]

    Inverse probability weights can be used through the ``weights`` argument. See the applied examples for a
    demonstration.

    References
    ----------
    Cole SR, Edwards JK, Breskin A, Rosin S, Zivich PN, Shook-Sa BE, & Hudgens MG. (2023). Illustration of 2 Fusion
    Designs and Estimators. *American Journal of Epidemiology*, 192(3), 467-474.

    Rogan WJ & Gladen B. (1978). Estimating prevalence from the results of a screening test.
    *American Journal of Epidemiology*, 107(1), 71-76.

    Ross RK, Cole SR, Edwards JK, Zivich PN, Westreich D, Daniels JL, Price JT & Stringer JSA. (2024). Leveraging
    External Validation Data: The Challenges of Transporting Measurement Error Parameters. *Epidemiology*,
    35(2), 196-207.
    """
    # Processing inputs
    y = np.asarray(y)                           # Convert to NumPy array
    y_star = np.asarray(y_star)                 # Convert to NumPy array
    r = np.asarray(r)                           # Convert to NumPy array
    X = np.asarray(X)                           # Convert to NumPy array
    if weights is None:                         # Handle weights argument
        weights = 1                             # ... set all weight as 1
    else:                                       # Otherwise
        weights = np.asarray(weights)           # ... convert to NumPy array

    # Preparing data for estimating equation operations
    nXp = X.shape[1] + 1                        # Index start for the NumPy matrices
    y = np.where(r == 1, -999, y)               # Removing NaN (or any other indicators) for Y in main
    mu = theta[0]                               # Parameter of interest
    sens = theta[1:nXp]                         # Parameters for sensitivity model
    spec = theta[nXp:]                          # Parameters for specificity model

    # Nuisance models for sensitivity
    ee_sens = ee_regression(theta=sens, y=y_star, X=X,
                            model='logistic', weights=weights) * (1-r) * y
    sens_i = inverse_logit(np.dot(X, sens))     # Predicted sensitivity for each unit

    # Nuisance models for specificity
    ee_spec = ee_regression(theta=spec, y=1-y_star, X=X,
                            model='logistic', weights=weights) * (1-r) * (1-y)
    spec_i = inverse_logit(np.dot(X, spec))     # Predicted specificity for each unit

    # Estimating equation for the individual-level version of the Rogan-Gladen correction
    rg_equation = (y_star + spec_i - 1) / (sens_i + spec_i - 1)
    ee_corr_mean = r * (rg_equation - mu) * weights

    # Returning the stacked estimating equations
    return np.vstack([ee_corr_mean, ee_sens, ee_spec])


def ee_regression_calibration(theta, beta, a, a_star, r, X=None, weights=None):
    """Estimating equation for regression calibration with external data for a mismeasured *binary* action. Regression
    calibration is a simple to implement method to correct for measurement error.

    The general form of the estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            (\beta^* / \gamma_0) - \beta  \\
            (1-R_i) \left\{ A_i - \gamma_0 A_i^* + \gamma^T X_i \right\} X_i^T
        \end{bmatrix}
        = 0

    where :math:`A` is the gold-standard measurement of the action, :math:`A^*` is the mismeasured version of the binary
    action, :math:`X` is some additional covariates (including at least an intercept), and :math:`R` indicates whether
    someone was in the validation set (:math:`R=0` if in the validation set).

    The first estimating equation is for the corrected coefficient for :math:`A` on :math:`Y`. This is done by scaling
    the coefficient for :math:`A^*` on :math:`Y` (which comes from a model external to ``ee_regression_calibration``)
    by the predictiveness in terms of probability of :math:`A^*` for :math:`A`, :math:`\gamma_0`. The second estimating
    equation is used to estimate :math:`\gamma` using a linear probability model. Here, :math:`\gamma` are the
    parameters of the calibration model.

    Note
    ----
    For the second place in ``theta``, (i.e., ``theta[1]``), a starting value between between 0.5 and 1 is recommended.


    One caution for application of regression calibration is that it is only valid for non-differential measurement
    error. In cases of differential measurement error, methods like Multiple Imputation for Measurement Error (MIME)
    should be considered instead (Cole et al., 2006).

    If ``X=None`` then ``theta`` is a 1-by-3 array. Otherwise, ``theta`` is a 1-by-(2+`p`) array, where `p` is the is
    the dimension of :math:`X`.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 1+2`p` values.
    beta : float, int, ndarray
        Coefficient to correct from a model fit outside of ``ee_regression_calibration``. This coefficient should be
        for the main effect of ``a_star`` on the outcome. Notice that regression calibration only requires the
        coefficient to apply the correction (i.e., ``y`` is not needed for this estimating equation).
    a : ndarray, list, vector
        1-dimensional vector of `n` observed values. These are the gold-standard :math:`A` measurements in the external
        sample. All values should be either 0 or 1, and be non-missing among those with :math:`R=0`.
    a_star : ndarray, list, vector
        1-dimensional vector of `n` observed values. These are the mis-measured :math:`A` in the external and internal
        sample. All values should be either 0 or 1, and must be non-missing among those with :math:`R=0`.
    r : ndarray, list, vector
        1-dimensional vector of `n` indicators regarding whether an observation was part of the external validation
        data. Indicator should designate if observations are the main data.
    X : ndarray, list, vector, None, optional
        2-dimensional vector of a design matrix for calibration model. Notice that this design matrix should not include
        ``a``. Behind the scenes, ``a`` is added to this design matrix to make it easier to process the coefficients for
        the regression calibration step. Default is ``None``, which automatically generates an intercept, so the
        calibration model ``a ~ a_star + 1`` is fit by default.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of `n` weights. Default is ``None``, which assigns a weight of 1 to all observations. Note
        that weights are only used in the calibration model fitting.

    Returns
    -------
    array :
        Returns a 3-by-`n` or (2+`p`)-by-`n` NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_regression_calibration`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_regression_calibration

    TODO ... provide example ...

    References
    ----------
    Cole SR, Chu H, & Greenland S. (2006). Multiple-imputation for measurement-error correction.
    *International Journal of Epidemiology*, 35(4), 1074-1081.

    Cole SR, Jacobson LP, Tien PC, Kingsley L, Chmiel JS, & Anastos K. (2010). Using marginal structural
    measurement-error models to estimate the long-term effect of antiretroviral therapy on incident AIDS or death.
    *American Journal of Epidemiology*, 171(1), 113-122.
    """
    # Processing inputs
    a = np.asarray(a)                           # Convert to NumPy array
    r = np.asarray(r)                           # Convert to NumPy array
    if X is None:                               # Intercept-only model
        X = np.ones(a.shape)[:, None]           # ... intercept-only
    else:                                       # User-specified design matrix
        X = np.asarray(X)                       # ... convert to NumPy array
    if weights is None:                         # Handle weights argument
        weights = 1                             # ... set all weight as 1
    else:                                       # Otherwise
        weights = np.asarray(weights)           # ... convert to NumPy array

    # Preparing data for estimating equation operations
    beta_corrected = theta[0]                   # First parameter in vector will be corrected coefficient
    gamma = theta[1:]                           # All other parameters are for the regression calibration model

    # Calibration Model (via a linear probability model)
    ee_calib = ee_regression(theta=gamma,                              # Regression model for the calibration step
                             y=a, X=np.hstack([a_star[:, None], X]),   # ... Design matrices built as I expect them
                             model='linear',                           # ... linear regression
                             weights=weights)                          # ... with provided weights
    ee_calib = ee_calib * (1-r)                                        # Only have the external observations contribute

    # Corrected coefficient
    ee_corr_beta = np.ones(a.shape) * (beta / gamma[0] - beta_corrected)  # Transformation of coefficients for RC

    # Returning the stacked estimating equations
    return np.vstack([ee_corr_beta, ee_calib])
