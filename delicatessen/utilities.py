import warnings

import numpy as np


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
        if not a < b < k:
            raise ValueError("The 'hampel' loss function requires that a < b < k")
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


def spline(variable, knots, power=3, restricted=True):
    r"""Generate generic polynomial spline terms for a given NumPy array and pre-specified knots. Default is restricted
    cubic splines but unrestricted splines to different polynomial terms can also be specified.

    Parameters
    ----------
    variable : ndarray, vector, list
        1-dimensional vector of observed values. Input should consists of the variable to generate spline terms for
    knots : ndarray, vector, list
        1-dimensional vector of pre-specified knot locations. All knots should be between the minimum and maximum
        values of the input variable
    power : int, float, optional
        Power or polynomial term to use for the splines. Default is 3, which corresponds to cubic splines
    restricted : bool, optional
        Whether to generate restricted or unrestricted splines. Default is True, which corresponds to restricted
        splines. Restricted splines return one less column than the number of knots, whereas unrestricted splines
        return the same number of columns as knots

    Returns
    -------
    ndarray :
        A 2-dimensional array of the spline terms in ascending order of the knots.

    Examples
    --------

    References
    ----------

    """
    # Processing input and output arrays
    knots = sorted(knots)                           # Sorting the order of the knots
    n_cols = len(knots)                             # Column number is based on the number of provided knots
    x = np.asarray(variable)                        # Converting input variable to another array
    spline_terms = np.empty((x.shape[0], n_cols))   # Creating the output spline array

    # Generating each spline with it's corresponding term
    for i in range(n_cols):
        spline_terms[:, i] = np.where(x > knots[i], (x - knots[i])**power, 0)
        spline_terms[:, i] = np.where(np.isnan(x), np.nan, spline_terms[:, i])

    # Logic for unrestricted and restricted splines
    if restricted is False:
        return spline_terms
    else:
        for i in range(n_cols - 1):
            spline_terms[:, i] = np.where(x > knots[i], spline_terms[:, i] - spline_terms[:, -1], 0)
            spline_terms[:, i] = np.where(np.isnan(x), np.nan, spline_terms[:, i])
        return spline_terms[:, :-1]


def additive_design_matrix(X, specifications, return_penalty=False):
    r"""Generate an additive design matrix for generalized additive models (GAM) given a set of spline specifications to
    apply.

    Note
    ----
    This function is interally called by ``ee_additive_regression``. This function can also be called to aid in easily
    generating predicted values.

    Parameters
    ----------
    X : ndarray, vector, list
        Input independent variable data.
    specifications : list, None,
        A list of dictionaries that define the hyperparameters for the spline (e.g., number of knots, strength of
        penalty). For terms that should not have splines, ``None`` should be specified instead (see examples below).
        Each dictionary supports the following parameters:
        "knots", "n_knots", "natural", "power", "penalty"
        * knots (list): controls the position of the knots. Must be specified if n_knots is not specified.
        * n_knots (int): controls the number of knots and places all knots at equidistant positions between the 2.5th
            and 97.5th percentiles. Must be specified if knots is not specified
        * natural (bool): controls whether to generate natural (restricted) or unrestricted splines.
            Default is ``True``, which corresponds to natural splines.
        * power (float): controls the power to raise the spline terms to. Default is 3, which corresponds to cubic
            splines.
        * penalty (float): penalty term (:math:`\lambda`) applied to each corresponding spline basis term. Default is 0,
            which applies no penalty to the spline basis terms.
    return_penalty : bool, optional
        Whether the list of the corresponding penalty terms should also be returned. This functionality is used
        internally to create the list of penalty terms to provide the Ridge regression model, where only the spline
        terms are penalized. Default is False.

    Returns
    -------
    array :
        Returns a (b+k)-by-n design matrix as a NumPy array, where b is the number of columns in the input array and k
        is determined by the specifications of the spline basis functions.

    Examples
    --------
    Construction of a design matrix for an additive model should be done similar to the following

    >>> import numpy as np
    >>> import pandas as pd
    >>> from delicatessen.utilities import additive_design_matrix

    Some generic data to estimate a generalized additive model

    >>> n = 200
    >>> d = pd.DataFrame()
    >>> d['X'] = np.random.normal(size=n)
    >>> d['Z'] = np.random.normal(size=n)
    >>> d['W'] = np.random.binomial(n=1, p=0.5, size=n)
    >>> d['C'] = 1

    To begin, consider the simple input design matrix of ``d[['C', 'X']]``. This initial design matrix consists of an
    intercept term and a continuous term. Here, we will specify a natural spline with 20 knots for the second term only

    >>> specs = [None, {"n_knots": 20, "penalty": 10}]
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X']], specifications=specs)

    Rather than specify the number of knots, we can also assign the exact position of the knots

    Note
    ----
    Either the number of knots or knot locations must be specified.


    >>> specs = [None, {"knots": [-2, -1, 0, 1, 2], "penalty": 10}]
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X']], specifications=specs)

    Note
    ----
    Internally, the input knots are always sorted in ascending order.


    Other optional specifications are also available. Here, we will specify an unrestricted quadratic spline with a
    penalty of 5.5 for the second column of the design matrix.

    >>> specs = [None, {"knots": [-4, -2, 0, 2, 4], "natural": False, "power": 2, "penalty": 5.5}]
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X']], specifications=specs)

    Now consider the input design matrix of ``d[['C', 'X', 'Z', 'W']]``. This initial design matrix consists of an
    intercept, two continuous, and a categorical term. Here, we will specify splines for both continuous terms

    >>> specs = [None,                         # Intercept term
    >>>          {"knots": 20, "penalty": 25}, # X (continuous)
    >>>          {"knots": 10, "penalty": 15}, # Z (continuous)
    >>>          None]                         # W (categorical)
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X', 'Z', 'W']], specifications=specs)

    Notice that the two continuous terms have different spline specifications.

    Finally, we could opt to only generate a spline basis for one of the continuous variables

    >>> specs = [None,                         # Intercept term
    >>>          {"knots": 20, "penalty": 25}, # X (continuous)
    >>>          None,                         # Z (continuous)
    >>>          None]                         # W (categorical)
    >>> Xa_design = additive_design_matrix(X=d[['C', 'X', 'Z', 'W']], specifications=specs)

    Specification of splines can be modified and paired in a variety of ways. These are determined by the object type
    in the specification list, and the input dictionary for the spline terms.
    """
    # TODO consider replacing n_knots with integer detector...

    def generate_spline(variable, specification):
        return spline(variable=variable,
                      knots=specification["knots"],
                      power=specification["power"],
                      restricted=specification["natural"])

    def generate_default_spline_parameters(xvar, specification):
        keys = specification.keys()
        expected_keys = ["knots", "n_knots", "natural", "power", "penalty", ]
        defaults = {"knots": None,
                    "n_knots": None,
                    "natural": True,
                    "power": 3,
                    "penalty": 0}

        if "knots" not in keys:
            if "n_knots" not in keys:
                raise ValueError("For each spline, either `knots` or `n_knots` must be specified.")
            else:
                n_knots = specification["n_knots"]
                if n_knots < 1:
                    raise ValueError("The number of knots, `n_knots` must be a non-negative integer")
                elif n_knots == 1:
                    specification["knots"] = [np.median(xvar), ]
                elif n_knots == 2:
                    specification["knots"] = np.percentile(xvar, q=[100/3, 200/3]).tolist()
                else:
                    percentiles = np.linspace(2.5, 97.5, n_knots)
                    specification["knots"] = np.percentile(xvar, q=percentiles).tolist()
        if "natural" not in keys:
            specification["natural"] = defaults["natural"]
        if "power" not in keys:
            specification["power"] = defaults["power"]
        if "penalty" not in keys:
            specification["penalty"] = defaults["penalty"]

        # Checking the keys in the input dictionary against the expected keys
        keys = specification.keys()
        extra_keys = [param for param in keys if param not in expected_keys]
        if len(extra_keys) != 0:
            warnings.warn("The following keys were found in the specification: " + str(extra_keys)
                          + ". These keys are not supported and are being ignored.",
                          UserWarning)

        # Returning processed spline parameters dictionary
        return specification

    # Extract meta-data
    X = np.asarray(X)
    n_cols = X.shape[1]      # Number of columns in the input data
    n_obs = X.shape[0]       # Number of observations in the input data
    if isinstance(specifications, dict):
        specifications = [specifications, ]*n_cols
    elif specifications is None:
        specifications = [specifications, ]*n_cols
    else:
        if len(specifications) != n_cols:
            raise ValueError("The number of input specifications (" + str(len(specifications)) +
                             ") and the number of columns (" + str(n_cols) + ") do not match")

    # Generate spline terms for each column
    Xt = []
    penalties = []
    for col_id in range(n_cols):
        xvar = X[:, col_id]
        xspec = specifications[col_id]

        Xt.append(xvar.reshape(n_obs, 1))
        penalties.append(0)
        if xspec is not None:
            spec_i = generate_default_spline_parameters(xvar=xvar,
                                                        specification=xspec)
            spline_matrix = generate_spline(variable=X[:, col_id],
                                            specification=spec_i)
            Xt.append(spline_matrix)
            penalties = penalties + [spec_i["penalty"], ]*spline_matrix.shape[1]

    # Return the transformed design matrix
    if return_penalty:
        return np.hstack(Xt), penalties
    else:
        return np.hstack(Xt)
