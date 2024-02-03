#####################################################################################################################
# Estimating functions for measurement error
#####################################################################################################################

import numpy as np

from delicatessen.utilities import inverse_logit
from delicatessen.estimating_equations.regression import ee_regression
from delicatessen.estimating_equations.processing import generate_weights


def ee_rogan_gladen(theta, y, y_star, r, weights=None):
    """Estimating equation for the Rogan-Gladen correction for mismeasured *binary* outcomes. This estimator uses
    external data to estimate the sensitivity and specificity, and then uses those external estimates to correct the
    estimated proportion. The general form of the estimating equations are

    .. math::

        \sum_{i=1}^n
        \begin{bmatrix}
            \mu \times \left\{ \alpha + \beta - 1 \right\} - \left\{ \mu^* + \beta - 1 \right\} \\
            R_i (Y_i^* - \mu^*) \\
            (1-R_i) Y_i \left\{ Y^*_i - \beta \right\} \\
            (1-R_i) (1-Y_i) \left\{ Y^*_i - \alpha \right\} \\
        \end{bmatrix}
        = 0

    where :math:`Y` is the true value of the outcome, :math:`Y^*` is the mismeasured value of the outcome. The first
    estimating equation is the corrected proportion, the second is the naive proportion, the third is for sensitivity,
    and the fourth for specificity.

    Here, :math:`\theta` is a 1-by-4 array.

    Note
    ----
    The Rogan-Gladen estimator is only well-defined when :math:`\alpha + \beta > 1`, or the addition of sensitivity
    and specificity is greater than one.

    Parameters
    ----------
    theta : ndarray, list, vector
        Theta consists of 2+2b values. If ``X=None`` then 4 values should be provided.
    y : ndarray, list, vector
        1-dimensional vector of n observed values. These are the gold-standard :math:`Y` measurements in the external
        sample. All values should be either 0 or 1, and be non-missing among those with :math:`R=0`.
    y_star : ndarray, list, vector
        1-dimensional vector of n observed values. These are the mismeasured :math:`Y` values. All values should be
        either 0 or 1, and be non-missing among all observations.
    r : ndarray, list, vector
        1-dimensional vector of n indicators regarding whether an observation was part of the external validation data.
        Indicator should designate if observations are the main data.
    weights : ndarray, list, vector, None, optional
        1-dimensional vector of n weights. Default is ``None``, which assigns a weight of 1 to all observations.

    Returns
    -------
    array :
        Returns a (2+2b)-by-n NumPy array evaluated for the input ``theta``

    Examples
    --------
    Construction of a estimating equation(s) with ``ee_rogan_gladen`` should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_rogan_gladen

    Replicating the example from Cole et al. (2023).

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

    Inverse probability weights can be used through the ``weights`` argument.

    References
    ----------
    Cole SR, Edwards JK, Breskin A, Rosin S, Zivich PN, Shook-Sa BE, & Hudgens MG. (2023). Illustration of 2 Fusion
    Designs and Estimators. *American Journal of Epidemiology*, 192(3), 467-474.

    Rogan WJ & Gladen B. (1978). Estimating prevalence from the results of a screening test.
    *American Journal of Epidemiology*, 107(1), 71-76.
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
