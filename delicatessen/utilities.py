import numpy as np
from scipy.misc import derivative
from copy import copy


def partial_derivative(func, var, point, output, dx, order):
    """Calculate the partial derivatives for a given function

    Parameters
    ----------
    func : function
        Function with multiple variables to calculate the partial derivatives for
    var : int
        Index of variable to take the derivative of.
    point : ndarray
        Values to evaluate the partial derivative at
    output : int
        Index of the variable to extract the partial derivative of.
    dx : float
        Spacing to use to numerically approximate the partial derivatives of the bread matrix.
    order : int
        Number of points to use to evaluate the derivative. Must be an odd number

    Returns
    -------
    float
        Partial derivative at the variable at a particular points
    """
    args = copy(point[:])  # Copy is needed here to prevent over-writing (not sure why SciPy overwrites if not)

    def wraps(x):
        args[var] = x
        return func(args)[output]

    return derivative(wraps, point[var], dx=dx, order=order)


def logit(prob):
    """Logistic transformation of probabilities. Returns log-odds

    Parameters
    ----------
    prob : float, ndarray
        A single probability or an array of probabilities

    Returns
    -------
    logit-transformed probabilities
    """
    return np.log(prob / (1 - prob))


def inverse_logit(logodds):
    """Inverse logistic transformation. Returns probabilities

    Parameters
    ----------
    logodds : float, ndarray
        A single log-odd or an array of log-odds

    Returns
    -------
    inverse-logit transformed results (i.e. probabilities for log-odds)
    """
    return 1 / (1 + np.exp(-logodds))


def identity(value):
    """Identity transformation. Returns itself

    Note
    ----
    This function doesn't actually apply any transformation. It is used for arbitrary function calls that apply
    transformations, and this is called when no transformation is to be applied

    Parameters
    ----------
    value : float, ndarray
        A single value or an array of values

    Returns
    -------
    value
    """
    return value


def robust_loss_functions(residual, loss, k, a=None, b=None):
    r"""Loss functions for robust mean and robust regression estimating equations. This function is called internally
    for ``ee_mean_robust`` and ``ee_robust_regression``. This function can also be loaded, so user's can easily adapt
    their own regression models into robust regression models using the pre-defined loss functions.

    Note
    ----
    The loss functions here are technically the first-order derivatives of the loss functions


    The following score of the loss functions, :math:`f_k()`, are available.

    Andrew's Sine

    .. math::

        f_k(x) = I(k \pi <= x <= k \pi) \times \sin(x/k)

    Huber

    .. math::

        f_k(x) = x \times I(-k < x < k) + k \times (1 - I(-k < x < k)) \times \text{sign}(x)

    Tukey's biweight

    .. math::

        f_k(x) = x \times I(-k < x < k) + x \left( 1 - (x/k)^2 \right)^2

    Hampel (Hampel's add two additional parameters, :math:`a` and :math:`b`)

    .. math::

        f_k(x) =
        \begin{bmatrix}
            I(-a < x < a) \times x \\
            + I(a \ge |x| < b) \times a \times \text{sign}(x) \\
            + I(b \ge x < k) \times a \frac{k - x}{k - b} \\
            + I(-b \le x > -k) \times -a \frac{-k + x}{-k + b} \\
            + I(|x| \ge k) \times 0
        \end{bmatrix}

    Parameters
    ----------
    residual : ndarray, vector, list
        1-dimensional vector of n observed values. Input should consists of the residuals (the difference between the
        observed value and the predicted value). For the robust mean, this is :math:`Y_i - \mu`. For robust regression,
        this is :math:`Y_i - X_i^T \beta`
    loss : str
        Loss function to use. Options include: 'andrew', 'hampel', 'huber', 'minimax', 'tukey'
    k : int, float
        Tuning parameter for the corresponding loss function. Note: no default is provided, since each loss function
        has different recommendations.
    a : int, float, None, optional
        Lower parameter for the 'hampel' loss function
    b : int, float, None, optional
        Upper parameter for the 'hampel' loss function

    Returns
    -------
    array :
        Returns a 1-by-n NumPy array evaluated for the input theta and residual

    Examples
    --------
    Using the robust loss function

    >>> import numpy as np
    >>> from delicatessen.utilities import robust_loss_functions

    Some generic data to stand-in for the residuals

    >>> residuals = np.random.standard_cauchy(size=20)

    Huber's loss function

    >>> robust_loss_functions(residuals, loss='huber', k=1.345)

    Andrew's Sine

    >>> robust_loss_functions(residuals, loss='andrew', k=1.339)

    Tukey's biweight

    >>> robust_loss_functions(residuals, loss='tukey', k=4.685)

    Hampel's loss function

    >>> robust_loss_functions(residuals, loss='hampel', k=8, a=2, b=4)

    References
    ----------
    Andrews DF. (1974). A robust method for multiple linear regression. *Technometrics*, 16(4), 523-531.

    Beaton AE & Tukey JW (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic
    data. *Technometrics*, 16(2), 147-185.

    Hampel FR. (1971). A general qualitative definition of robustness. *The Annals of Mathematical Statistics*,
    42(6), 1887-1896.

    Huber PJ. (1964). Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73–101.

    Huber PJ, Ronchetti EM. (2009) Robust Statistics 2nd Edition. Wiley. pgs 98-100
    """
    # Checking type for later .lower() calls so informative error
    if not isinstance(loss, str):
        raise ValueError("The provided loss function should be a string.")

    # TODO add check that a < b < k for Hampel

    # Huber function
    elif loss.lower() == "huber":
        xr = np.clip(residual, a_min=-k, a_max=k)

    # Tukey's biweight
    elif loss.lower() == "tukey":
        xr = np.where(np.abs(residual) <= k, residual * (1-(residual/k)**2)**2, 0)

    # Andrew's Sine
    elif loss.lower() == "andrew":
        xr = np.where(np.abs(residual) <= k*np.pi,
                      np.sin(residual/k), np.nan)
        xr = np.where(residual > k*np.pi, 0, xr)
        xr = np.where(residual < -k*np.pi, 0, xr)

    # Hampel
    elif loss.lower() == "hampel":
        if a is None or b is None:
            raise ValueError("The 'hampel' loss function requires the optional `a` and `b` arguments")
        xr = np.where(np.abs(residual) < a, residual, np.nan)
        xr = np.where((a <= residual) & (residual < b), a, xr)
        xr = np.where((-a >= residual) & (residual > -b), -a, xr)
        xr = np.where((b <= residual) & (residual < k), (k - residual)/(k - b)*a, xr)
        xr = np.where((-b >= residual) & (residual > -k), (-k - residual)/(-k + b)*-a, xr)
        xr = np.where(np.abs(residual) >= k, 0, xr)

    # Catch-all value error for the provided loss function
    else:
        raise ValueError("The loss function "+str(loss)+" is not available.")

    # Returning the updated values
    return xr
