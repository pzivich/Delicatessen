#####################################################################################################################
# Implementation of the M-estimator
#####################################################################################################################

import warnings
import numpy as np
from scipy.optimize import newton, root, minimize
from scipy.stats import norm
from scipy.optimize import OptimizeWarning

from delicatessen.errors import check_alpha_level, check_variance_is_not_none
from delicatessen.sandwich import compute_bread, compute_meat, build_sandwich


class _GeneralEstimator:
    """Internal base class for M-estimator and GMM-estimator to be built from. Not intended for users.
    """
    def __init__(self, stacked_equations, init, subset=None):
        self.stacked_equations = stacked_equations     # User-input stacked estimating equations
        self.init = init                               # User-input initial starting values for solving estimating eqs
        if subset is None:                             # Handling subset of parameters
            self._subset_ = subset                     # ... when None, set as None
        else:                                          # Otherwise
            self._subset_ = sorted(subset)             # ... ensure it is sorted

        # Storage for results from the Estimation procedure
        self.n_obs = None                 # Number of unique observations (calculated later)
        self.theta = None                 # Optimized theta values (calculated later)
        self.bread = None                 # Bread from theta values (calculated later)
        self.meat = None                  # Meat from theta values (calculated later)
        self.variance = None              # Covariance matrix for theta values (calculated later)
        self.asymptotic_variance = None   # Asymptotic covariance matrix for theta values (calculated later)
        self.weight_matrix = None         # Weight matrix for the GMM estimator

    def confidence_intervals(self, alpha=0.05):
        r"""Calculate two-sided Wald-type :math:`(1 - \alpha) \times` 100% confidence intervals using the point
        and sandwich variance estimates. The formula for the confidence intervals is

        .. math::

            \hat{\theta} \pm z_{\alpha / 2} \times \widehat{SE}(\hat{\theta})

        Note
        ----
        The ``.estimate()`` function must be called before the confidence intervals can be calculated.

        Parameters
        ----------
        alpha : float, optional
            The :math:`0 < \alpha < 1` level for the corresponding confidence intervals. Default is 0.05, which
            corresponds to 95% confidence intervals.

        Returns
        -------
        array :
            b-by-2 array, where row 1 is the confidence intervals for :math:`\theta_1`, ..., and row b is the confidence
            intervals for :math:`\theta_b`
        """
        # Check that variance has been estimated
        check_variance_is_not_none(variance=self.variance)

        # Check valid alpha value is being provided
        check_alpha_level(alpha=alpha)

        # 'Looking up' via Z table
        z_alpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)   # Z_alpha value for CI

        # Calculating confidence intervals
        param_se = np.sqrt(np.diag(self.variance))          # Take the diagonal of the sandwich and then SQRT
        lower_ci = self.theta - z_alpha * param_se          # Calculate lower CI
        upper_ci = self.theta + z_alpha * param_se          # Calculate upper CI

        # Return 2D array of lower and upper confidence intervals
        return np.asarray([lower_ci, upper_ci]).T

    def z_scores(self, null=0):
        r"""Calculate the Z-score using the point estimates and the sandwich variance. The formula for the Z score is

        .. math::

            \frac{\hat{\theta} - \theta}{\widehat{SE}(\hat{\theta})}

        where :math:`\theta` is the null. The ``.estimate()`` function must be called before the Z-scores can be
        calculated. Note that the default value for the null is zero.

        Parameters
        ----------
        null: int, float, ndarray, optional
            Null or reference for the the corresponding P-values. Default is 0.

        Returns
        -------
        array :
            Array of Z-scores for :math:`\theta_1, ..., \theta_b`, respectively
        """
        # Check that variance has been estimated
        check_variance_is_not_none(variance=self.variance)

        # Calculating Z-scores
        se = np.sqrt(np.diag(self.variance))       # Extract the standard error estimates from the sandwich
        z_score = (self.theta - null) / se         # Compute the Z-score
        return z_score                             # Return the Z-score to the user

    def p_values(self, null=0):
        r"""Calculate two-sided Wald-type P-values using the Z-scores compute using the point and the sandwich variance
        estimates. Once the Z-scores are computed, the corresponding P-values are obtained by comparing to the standard
        normal distribution.

        The ``.estimate()`` function must be called before the P-values can be calculated.

        Parameters
        ----------
        null: int, float, ndarray, optional
            Null or reference for the the corresponding P-values. Default is 0.

        Returns
        -------
        array :
            Array of P-values for :math:`\theta_1, ..., \theta_b`, respectively
        """
        z_score = self.z_scores(null=null)         # Calculating the Z-scores
        p_value = norm.sf(np.abs(z_score)) * 2     # Compute the corresponding P-values
        return p_value                             # Return P-values to the user

    def s_values(self, null=0):
        r"""Calculate two-sided Wald-type S-values using the point estimates and the sandwich variance. The S-value,
        or Shannon Information value, is a transformation of the P-values that has been argued to be more easily
        interpretable as it can be related back to simple coin-flipping scenarios. The transformation from a P-value
        into a S-value is.

        .. math::

            S = - \log_2(P)

        where :math:`P` is the corresponding P-value. The ``.estimate()`` function must be called before the S-values
        can be calculated.

        The S-value can be contextualized in terms of coin-flips. Suppose the S-value is :math:`s`. Then :math:`s`
        corresponds to the number of heads in a row with a fair coin (equal chances heads or tails). As :math:`s`
        increases, one would be more 'surprised' by the result (e.g., it might not be surprising to have 2 heads in a
        row, but it would be surprising for 20 in a row).

        Parameters
        ----------
        null: int, float, ndarray, optional
            Null or reference for the the corresponding S-values. Default is 0.

        Returns
        -------
        array :
            Array of S-values for :math:`\theta_1, ..., \theta_b`, respectively

        References
        ----------
        Cole SR, Edwards JK, & Greenland S. (2021). Surprise! *American Journal of Epidemiology*, 190(2), 191-193.

        Greenland S. (2019). Valid P-values behave exactly as they should: Some misleading criticisms of P-values and
        their resolution with S-values. *The American Statistician*, 73(sup1), 106-114.
        """
        p_values = self.p_values(null=null)          # Calculate P-values
        s_values = -1 * np.log2(p_values)            # Transform into S-values
        return s_values                              # Return S-values to the user

    def influence_functions(self, allow_pinv=True):
        r"""Calculate the influence function values for individual observations. Note that the influence function for
        :math:`i` is computed using the following equality

        .. math::

            IF(O_i; \theta) = \frac{\psi(O_i; \theta)}{B_n(O, theta)}

        or that the influence functions is equal to the estimating function scaled by the inverse of the bread matrix.
        While this approach was historically used to derive the influence function, there are more general alternatives.
        `delicatessen` offers this implementation since

        Returns
        -------
        array :
            Returns a `n`-by-`b` NumPy array evaluated for the input ``theta``.

        References
        ----------
        Fisher A, & Kennedy EH. (2021). Visually communicating and teaching intuition for influence functions.
        *The American Statistician*, 75(2), 162-172.

        Hines O, Dukes O, Diaz-Ordaz K, & Vansteelandt S. (2022). Demystifying statistical learning based on efficient
        influence functions. *The American Statistician*, 76(3), 292-304.

        Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. *The American Statistician*, 56(1), 29-38.
        """
        # Check that variance has been estimated
        check_variance_is_not_none(variance=self.variance)

        # Calculating estimating function
        efunc_i = self.stacked_equations(theta=self.theta)   # Computing estimating function at theta-hat
        if efunc_i.ndim == 1:                                # Checking if one dimension
            efunc_i = np.asarray([efunc_i, ])                # ... so can make 2D for dot product

        # Calculating inverse of bread matrix
        if allow_pinv:                                       # Allowing the pseudo-inverse
            bread_invert = np.linalg.pinv(self.bread)        # ... then call pinv
        else:                                                # Only allowing the actual inverse
            bread_invert = np.linalg.inv(self.bread)         # ... then call inv

        # Calculating influence function
        ifunc_i = np.dot(bread_invert, efunc_i)
        return ifunc_i.T

    @staticmethod
    def _eval_ee_(stacked_equations, subset):
        """Function to evaluate the estimating equations over the :math:`n` units for a given set of parameters

        Note
        ----
        Added in v1.0 to replace the inner functionality of ``_mestimation_answer_`` for the new ``approx_fprime`` but
        still support ``subset`` (without having to flip the subset flag).

        Parameters
        ----------
        stacked_equations :
            Estimating equations to evaluate
        subset :
            Whether to consider a subset of parameters

        Returns
        -------
        numpy.array
        """
        # IF stacked_equation returns 1 value, only return that 1 value
        if len(stacked_equations.shape) == 1:          # Checking length
            vals = np.sum(stacked_equations)           # ... avoid SciPy error by returning value rather than tuple
        # ELSE need to return a tuple for the root-finding procedure
        else:                                          # ... also considering how subset argument is handled
            # NOTE: switching to np.sum(..., axis=1) didn't speed things up versus a for-loop
            vals = ()                                  # ... create empty tuple
            rows = stacked_equations.shape[0]          # ... determine how many rows / parameters are present
            if subset is None:                         # ... if no subset, then simple loop where
                for i in range(rows):                  # ... go through each individual theta in the stack
                    row = stacked_equations[i, :]      # ... extract corresponding row
                    vals += (np.sum(row), )            # ... then add the theta sum to the tuple of thetas
            else:                                      # ... if subset, then conditional loop (to speed up)
                for i in range(rows):                  # ... go through each individual theta in the stack
                    if i in subset:                    # ... if parameter is in subset then
                        row = stacked_equations[i, :]  # ... extract corresponding row
                        vals += (np.sum(row), )        # ... then add the theta sum to the tuple of thetas
            vals = np.asarray(vals)                    # ... converting to a NumPy array for ease

        # Return the calculated values of theta
        return vals

    @staticmethod
    def _error_checker_(vals_at_init):
        # Error checking before running procedure
        if np.sum(vals_at_init) is None:
            raise ValueError("When evaluating the estimating equation, `None` was returned. Please check that the "
                             "stacked_equations returns an array evaluated at theta.")
        if np.isnan(np.sum(vals_at_init)):         # Check to see if any np.nan's occur with the initial values
            # Identifying the bad columns
            nans_in_column = np.sum(np.isnan(vals_at_init), axis=0)        # Counting up all NAN's per estimating eq
            columns_w_nans = np.argwhere(nans_in_column >= 1).flatten()    # Returning indices that have any NAN's
            raise ValueError("When evaluated at the initial values, the `stacked_equations` return at least one "
                             "np.nan at the following estimating equation indices: " +
                             str(list(columns_w_nans)) + ". "
                             "As delicatessen does not natively handle missing data, please ensure the "
                             "provided estimating equations resolve any np.nan values accordingly. For details on "
                             "how to handle np.nan's see the documentation at: "
                             "https://deli.readthedocs.io/en/latest/Custom-EE.html#Handling-np.nan")


class MEstimator(_GeneralEstimator):
    r"""M-Estimator for stacked estimating equations.

    Estimating equations are a general approach to point and variance estimation that consists of defining an estimator
    as the solution to a vector of equations that are equal to zero. The corresponding estimators, often called
    M-estimators or Z-estimators, satisify the following equation

    .. math::

        \sum_{i=1}^n \psi(O_i, \hat{\theta}) = 0

    where :math:`\psi` is the :math:`v`-dimensional vector of estimating function(s), :math:`\hat{\theta}` is the
    :math:`v`-dimensional parameter vector, and :math:`O_i` is the observed data (where units are independent but not
    necessarily identically distributed).

    M-Estimators consists of two broad step: point estimation and variance estimation. Point estimation is carried out
    by determining the values of :math:`\theta` where the sum of the estimating equations are zero. This is done via
    standard root-finding algorithms.

    For variance estimation, sandwich variance estimator is used. The asymptotic sandwich variance estimator consists of

    .. math::

        V_n(O, \hat{\theta}) = B_n(O, \hat{\theta})^{-1} F_n(O, \hat{\theta}) \left\{B_n(O, \hat{\theta}^{-1})\right\}^T

    where :math:`B` is the 'bread' and :math:`F` is the 'filling' matrix. These matrices are defined as

    .. math::

        B_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} - \frac{\partial}{\partial \theta} \psi(O_i, \hat{\theta})

    .. math::

        F_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} \psi(O_i, \hat{\theta}) \psi(O_i, \hat{\theta})^T

    respectively. The partial derivatives for the bread are calculated using either numerical approximation (e.g.,
    forward difference method) or forward-mode automatic differentiation. Inverting the bread is done via NumPy's
    ``linalg.pinv``. For the filling, the dot product is taken at :math:`\hat{\theta}`.

    After completion of these steps, point and variance estimates are stored. These can be extracted from
    ``MEstimator``. Further, confidence intervals, Z-scores, P-values, or S-values can all be generated.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    init : list, set, array
        Initial values for the root-finding algorithm. A total of `v` values should be provided.
    subset : list, set, array, None, optional
        Optional argument to conduct the root-finding procedure on a subset of parameters in the estimating equations.
        The input list is used to location index the parameter array via ``np.take()``. The subset list will
        only affect the root-finding procedure (i.e., the sandwich variance estimator ignores the subset argument).
        Default is ``None``, which runs the root-finding procedure for all parameters in the estimating equations.

    Note
    ----
    Because the root-finding procedure is NOT ran for parameters outside of the subset, those coefficients *must* be
    solved outside of ``MEstimator``. In general, I do *NOT* recommend using the ``subset`` argument unless a series of
    complex estimating equations need to be solved. In general, this argument does not massively improve speed until
    the estimating equations consist of hundreds of parameters.


    Attributes
    ----------
    theta : ndarray
        Parameters for estimating equations. Becomes the estimated parameters after `.estimate()` is called
    variance : ndarray
        Covariance matrix for the parameters, in the same order as the parameter vector
    asymptotic_variance : ndarray
        Asymptotic covariance matrix for the parameters. Not scaled by :math:`n`
    bread : ndarray
        Bread matrix for the parameter vector
    meat : ndarray
        Meat matrix for the parameter vector

    Examples
    --------
    Loading necessary functions and building a generic data set for estimation of the mean

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import MEstimator
    >>> from delicatessen.estimating_equations import ee_mean_variance

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    M-estimation with built-in estimating equation for the mean and variance. First, ``psi``, or the stacked estimating
    equations, is defined

    >>> def psi(theta):
    >>>     return ee_mean_variance(theta=theta, y=y_dat)

    Calling the M-estimation procedure

    >>> estr = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate()

    Inspecting the output results

    >>> estr.theta                                  # Point estimates
    >>> estr.variance                               # Covariance
    >>> estr.asymptotic_variance                    # Asymptotic covariance
    >>> np.sqrt(np.diag(estr.asymptotic_variance))  # Standard deviation
    >>> estr.variance                               # Covariance
    >>> np.sqrt(np.diag(estr.variance))             # Standard error
    >>> estr.confidence_intervals()                 # Confidence intervals
    >>> estr.z_scores()                             # Z-scores
    >>> estr.p_values()                             # P-values
    >>> estr.s_values()                             # S-values

    Alternatively, a custom estimating equation can be specified. This is done by constructing a valid estimating
    equation for the ``MEstimator``. The ``MEstimator`` expects the ``psi`` function to return a `v`-by-`n` array,
    where `v` is the number of parameters (length of ``theta``) and n is the total number of observations. Below is an
    example of the mean and variance estimating equation from before, but implemented by-hand

    >>> def psi(theta):
    >>>     y = np.array(y_dat)
    >>>     mean = y - theta[0]
    >>>     variance = (y - theta[0]) ** 2 - theta[1]
    >>>     return mean, variance

    The M-estimation procedure is called using the same approach as before

    >>> estr = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate()

    Note that ``len(init)`` should be equal to `v`. So in this case, two initial values are provided.

    ``MEstimator`` can also be run with a user-provided root-finding algorithm. To specify a custom root-finder, a
    function must be created by the user that consists of two keyword arguments (``stacked_equations``, ``init``) and
    must return only the optimized values. The following is an example with SciPy's Levenberg-Marquardt algorithm
    implemented in ``root``.

    >>> def custom_solver(stacked_equations, init):
    >>>     options = {"maxiter": 1000}
    >>>     opt = root(stacked_equations,
    >>>                x0=np.asarray(init),
    >>>                method='lm', tol=1e-9, options=options)
    >>>     return opt.x

    The provided custom root-finder can then be implemented like the following (continuing with the estimating equation
    from the previous example):

    >>> estr = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate(solver=custom_solver)

    For more examples on how to apply ``MEstimator``, see https://deli.readthedocs.io/en/latest/

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Ross RK, Zivich PN, Stringer JSA, & Cole SR. (2024). M-estimation for common epidemiological measures: introduction
    and applied examples. *International Journal of Epidemiology*, 53(2), dyae030.

    Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. *The American Statistician*, 56(1), 29-38.
    """
    def estimate(self, solver='lm', maxiter=5000, tolerance=1e-9, deriv_method='approx', dx=1e-9, allow_pinv=True):
        """Run the point and variance estimation procedures for given estimating equation and starting values. This
        function carries out the point and variance estimation of ``theta``. After this procedure, the point estimates
        (in ``theta``) and the covariance matrix (in ``variance``) can be extracted from the ``MEstimator`` object.

        Parameters
        ----------
        solver : str, function, callable, optional
            Method to use for the root-finding procedure. Default is the Levenberg-Marquardt algorithm
            (``scipy.optimize.root(method='lm')``, specified by ``solver='lm'``). Other built-in option is the secant
            method (``scipy.optimize.newton``, specified by ``solver='newton'``), and a modification of the Powell
            hybrid method (``scipy.optimize.root(method='hybr')``, specified by ``solver='hybr'``). Finally, any generic
            root-finding algorithm can be used via a user-provided callable object. The function must consist of two
            keyword arguments: ``stacked_equations``, and ``init``. Additionally, the function should return only the
            optimized values. Please review the provided example in the documentation for how to implement a custom
            root-finding algorithm.
        maxiter : int, optional
            Maximum iterations to consider for the root finding procedure. Default is 5000 iterations. For complex
            estimating equations, this value may need to be increased. This argument is not used when a custom
            root-finding method (e.g., ``solver``) is provided.
        tolerance : float, optional
            Maximum tolerance for errors in the root finding in ``scipy.optimize``. Default is 1e-9. This argument is
            not used when a custom root-finding method (e.g., ``solver``) is provided.
        deriv_method : str, optional
            Method to compute the derivative of the estimating equations for the bread matrix. Default is ``'approx'``.
            Options include numerical approximation via the forward difference method via SciPy (``'approx'``), forward
            difference as implemented in delicatessen (`'fapprox'`), backward difference as implemented in delicatessen
            (`'bapprox'`), central difference implemented as in delicatessen (`'capprox'`), or forward-mode automatic
            differentiation as implemented in delicatessen(``'exact'``).
        dx : float, optional
            Spacing to use to numerically approximate the partial derivatives of the bread matrix. Default is 1e-9.
            Here, a small value for ``dx`` should be used, since some large values can result in poor approximations.
            This argument is only used with numerical approximation methods.
        allow_pinv : bool, optional
            Whether to allow for the pseudo-inverse (via ``numpy.linalg.pinv``) if the bread matrix is determined to be
            non-invertible. If you want to disallow the pseudo-inverse (i.e., use ``numpy.linalg.inv``), set this
            argument to ``False``. Default is ``True``, which  is more robust to the possible bread matrices.

        Returns
        -------
        None
        """
        # Evaluate stacked estimating equations at init
        vals_at_init = self.stacked_equations(theta=self.init)    # Calculating the initial values
        vals_at_init = np.asarray(vals_at_init                    # Convert output to an array (in case it isn't)
                                  ).T                             # ... transpose so N is always the 1st element

        # Error checking for inputs
        self._error_checker_(vals_at_init=vals_at_init)

        # Checking dimensions of output for M-estimator
        if vals_at_init.ndim == 1 and np.asarray(self.init).shape[0] == 1:     # Checks to ensure dimensions align
            # the starting if-state is to work-around inits=[0, ] (otherwise breaks the first else-if)
            pass
        elif vals_at_init.ndim == 1 and np.asarray(self.init).shape[0] != 1:
            raise ValueError("The number of initial values and the number of rows returned by `stacked_equations` "
                             "should be equal but there are " + str(np.asarray(self.init).shape[0]) + " initial values "
                             "and the `stacked_equations` function returns " + str(1) + " row.")
        elif np.asarray(self.init).shape[0] != vals_at_init.shape[1]:
            raise ValueError("The number of initial values and the number of rows returned by `stacked_equations` "
                             "should be equal but there are " + str(np.asarray(self.init).shape[0]) + " initial values "
                             "and the `stacked_equations` function returns " + str(vals_at_init.shape[1])
                             + " row(s).")
        elif vals_at_init.ndim > 2:
            raise ValueError("A 2-dimensional array is expected, but the `stacked_equations` returns a "
                             + str(vals_at_init.ndim) + "-dimensional array.")
        else:
            pass

        # Trick to get the number of observations from the estimating equations (transposed above)
        self.n_obs = vals_at_init.shape[0]

        # STEP 1: solving the M-estimator stacked equations
        # To allow for optimization of only a subset of parameters in the estimating equation (in theory meant to
        #   simplify the process of complex stacked estimating equations where pre-washing can be done effectively),
        #   we do some internal processing. Essentially, we 'freeze' the parameters outside of self._subset_ as their
        #   inits, and let the root-finding procedure update the self._subset_ parameters. We do this by subsetting out
        #   the init values then passing them along to root(). Behind the scenes, self._mestimation_answer_() expands
        #   the parameters (to include everything), calculates the estimating equation at those values, and then
        #   extracts the corresponding subset.
        #   This process only takes place within Step 1. There is an inherent danger with this process in that if
        #   non-subset parameters are not pre-washed, then the returned parameters will not be correct. I am
        #   considering adding a warning for self_subset_, but I currently just trust the user...

        # Processing initial values based on whether subset option was specified
        if self._subset_ is None:                        # If NOT subset,
            inits = self.init                            # ... give all initial values
        else:                                            # If subset,
            inits = np.take(self.init, self._subset_)    # ... then extract initial values for the subset

        # Calculating parameters values via the root-finder (for only the subset of values!)
        slv_theta = self._solve_coefficients_(stacked_equations=self._mestimation_answer_,  # Give the EE's
                                              init=inits,                                   # Give the subset vals
                                              method=solver,                                # Specify the solver
                                              maxiter=maxiter,                              # Set max iterations
                                              tolerance=tolerance                           # Set allowable tolerance
                                              )

        # Processing parameters after the root-finding procedure
        if self._subset_ is None:                        # If NOT subset,
            self.theta = slv_theta                       # ... then use the full output/solved theta
        else:                                            # If subset,
            self.theta = np.asarray(self.init)           # ... copy the initial values
            for s, n in zip(self._subset_, slv_theta):   # ... then look over the subset and input theta
                self.theta[s] = n                        # ... and update the subset to the output/solved theta

        # STEP 2: calculating the sandwich variance
        # After solving for the parameters, we now can compute the empirical sandwich variance estimator. This is
        #   done by compute the bread and meat matrices and then combining them. This is now done by a separate
        #   functionalities within the `sandwich.py` file as of v2.2.
        # STEP 2.1: baking the Bread
        self.bread = compute_bread(stacked_equations=self.stacked_equations,
                                   theta=self.theta,
                                   deriv_method=deriv_method,
                                   dx=dx) / self.n_obs

        # STEP 2.2: slicing the meat
        self.meat = compute_meat(stacked_equations=self.stacked_equations,
                                 theta=self.theta) / self.n_obs

        # STEP 2.3: assembling the sandwich (variance)
        self.asymptotic_variance = build_sandwich(bread=self.bread,
                                                  meat=self.meat,
                                                  allow_pinv=allow_pinv)
        if self.asymptotic_variance is None:
            self.variance = self.asymptotic_variance
        else:
            self.variance = self.asymptotic_variance / self.n_obs

    def _mestimation_answer_(self, theta):
        """Internal function to evaluate the sum of the estimating equations. The summation is internally evaluated
        since access to the estimating functions is needed for the sandwich variance computations. This function is
        used by the root-finding procedure (since we need the subset applied).

        Parameters
        ----------
        theta : array
            b-by-n matrix to sum over the values of n.

        Returns
        -------
        array :
            b-by-1 array, which is the sum over n for each b.
        """
        # Option for the subset argument
        if self._subset_ is None:                      # If NOT subset then,
            full_theta = theta                         # ... then use the full input theta
        else:                                          # If subset then,
            full_theta = np.asarray(self.init)         # ... copy the initial values to ndarray
            np.put(full_theta,                         # ... update in place the previous array
                   ind=self._subset_,                  # ... go to the subset indices
                   v=theta)                            # ... then input current iteration values

        stacked_equations = np.asarray(self.stacked_equations(full_theta))  # Returning stacked equation
        return self._eval_ee_(stacked_equations=stacked_equations,   # Passing to evaluating function
                              subset=self._subset_)                  # ... with specified subset

    @staticmethod
    def _solve_coefficients_(stacked_equations, init, method, maxiter, tolerance):
        """Calls the root-finding procedure for the values of theta, such that the estimating equations are equal to
        zero. Default uses the Levenberg-Marquardt algorithm from SciPy.

        Parameters
        ----------
        stacked_equations : function
            Function that contains the estimating equations
        init : array
            Initial values for the optimizer
        method : str, function, callable
            Method to use for the root finding procedure. Can be either a string or a callable object
        maxiter : int
            Maximum iterations to consider for the root finding procedure
        tolerance : float
            Maximum tolerance for errors in the root finding. This is SciPy's `tol` argument.

        Returns
        -------
        psi: array
            Solved or optimal values for theta given the estimating equations and data
        """
        # Newton solver has a special catch (is not nested in root())
        if method == "newton":
            psi = newton(stacked_equations,    # ... stacked equations to solve (should be written as sums)
                         x0=np.asarray(init),  # ... initial values for solver
                         # full_output=True,   # ... returns full output (rather than just the root)
                         maxiter=maxiter,      # ... setting maximum number of iterations
                         tol=tolerance,        # ... setting some tolerance values (should be able to update)
                         disp=True)            # ... option to raise RuntimeError if doesn't converge
        # Otherwise, goes to root with some selected optimizers
        elif method in ['lm', ]:
            options = {"maxiter": maxiter}
            opt = root(stacked_equations,      # ... stacked equations to solve (should be written as sums)
                       x0=np.asarray(init),    # ... initial values for solver
                       method=method,          # ... allow for valid root-finding methods
                       tol=tolerance,          # ... setting some tolerance values
                       options=options)        # ... options for the selected solver
            psi = opt.x                        # Error handling if fails to converge
            if opt.success == 0:
                print("Root-finding failed to converge...")
                raise RuntimeError(opt.message)
        elif method in ['hybr', ]:
            options = {"maxfev": maxiter}
            opt = root(stacked_equations,      # ... stacked equations to solve (should be written as sums)
                       x0=np.asarray(init),    # ... initial values for solver
                       method=method,          # ... allow for valid root-finding methods
                       tol=tolerance,          # ... setting some tolerance values
                       options=options)        # ... allow for options in hybrid
            psi = opt.x                        # Error handling if fails to converge
            if opt.success == 0:
                print("Root-finding failed to converge...")
                raise RuntimeError(opt.message)
        elif callable(method):
            try:
                psi = method(stacked_equations=stacked_equations,
                             init=np.asarray(init))
            except TypeError:
                raise TypeError("The user-specified root-finding `solver` must be a function (or callable object) with "
                                "the following keyword arguments: `stacked_equations`, `init`.")
            if psi is None:
                raise ValueError("The user-specified root-finding `solver` must return the solution to the "
                                 "optimization")
        else:
            # ... otherwise throw ValueError if no other root-finding steps are triggered.
            raise ValueError("The solver '" + str(method) + "' is not available. Please see the "
                             "documentation for valid options for the optimizer.")

        # Return optimized theta array
        return psi


class GMMEstimator(_GeneralEstimator):
    r"""Generalized Method of Moments (GMM) Estimator for stacked estimating equations.

    Estimating equations are a general approach to point and variance estimation that consists of defining an estimator
    as the solution to a vector of equations that are equal to zero. The corresponding GMM estimator satisifies the
    following equation

    .. math::

        \text{argmin}_{\theta} \left[ \sum_{i=1}^n \psi(O_i, \hat{\theta}) \right]^T
        \text{Q}
        \left[ \sum_{i=1}^n \psi(O_i, \hat{\theta}) \right]^T

    where :math:`\psi` is the :math:`\ge v`-dimensional vector of estimating function(s), :math:`\hat{\theta}` is the
    :math:`v`-dimensional parameter vector, :math:`\text{Q}` is a weight matrix, and :math:`O_i` is the observed data
    (where units are independent but not necessarily identically distributed).

    GMM estimators consists of two broad step: point estimation and variance estimation. Point estimation is carried
    out by finding where the minimum of the negative product of the estimating equations is. Initially :math:`\text{Q}`
    is set to be the identity matrix. This process is done via standard minimization algorithms.

    Note
    ----
    For over-identified settings, ``GMMEstimator`` uses an two-step iterative procedure, where :math:`\theta` is
    estimated, then the meat matrix is computed. The meat matrix replaces the identity matrix as :math:`\text{Q}`. Then
    the optimization process is repeated.


    For variance estimation, sandwich variance estimator is used. The asymptotic sandwich variance estimator consists of

    .. math::

        V_n(O, \hat{\theta}) = B_n(O, \hat{\theta})^{-1} F_n(O, \hat{\theta}) \left\{B_n(O, \hat{\theta}^{-1})\right\}^T

    where :math:`B` is the 'bread' and :math:`F` is the 'filling' matrix. These matrices are defined as

    .. math::

        B_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} - \frac{\partial}{\partial \theta} \psi(O_i, \hat{\theta})

    .. math::

        F_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} \psi(O_i, \hat{\theta}) \psi(O_i, \hat{\theta})^T

    respectively. The partial derivatives for the bread are calculated using either numerical approximation (e.g.,
    forward difference method) or forward-mode automatic differentiation. Inverting the bread is done via NumPy's
    ``linalg.pinv``. For the filling, the dot product is taken at :math:`\hat{\theta}`.

    After completion of these steps, point and variance estimates are stored. These can be extracted from
    ``GMMEstimator``. Further, confidence intervals, Z-scores, P-values, or S-values can all be generated.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a `v`-by-`n` NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    init : list, set, array
        Initial values for the minimization algorithm. A total of `v` values should be provided.
    subset : list, set, array, None, optional
        Optional argument to conduct the minimization procedure on a subset of parameters in the estimating equations.
        The input list is used to location index the parameter array via ``np.take()``. The subset list will
        only affect the minimization procedure (i.e., the sandwich variance estimator ignores the subset argument).
        Default is ``None``, which runs the minimization procedure for all parameters in the estimating equations.
    overid_maxiter : int, optional
        Maximum number of iterations for the two-step GMM-estimation procedure with over-identified parameters. Note
        that this procedure is only used for estimating equations that include an over-identified parameter, otherwise
        this argument has no impact. Default is ``10``.
    overid_tolerance : float, optional
        Error tolerance for the two-step GMM-estimation procedure with over-identified parameters. Note that this
        procedure is only used for estimating equations that include an over-identified parameter, otherwise this
        argument has no impact. If the tolerance threshold is met, the two-step procedure is terminated. The threshold
        is defined as the maximum of the absolute value of the difference between the parameters of the current
        iteration and the previous iteration. Default is ``1e-9``.

    Note
    ----
    Because the minimization procedure is NOT ran for parameters outside of the subset, those coefficients *must* be
    solved outside of ``GMMEstimator``. In general, I do *NOT* recommend using the ``subset`` argument unless a series
    of complex estimating equations need to be solved. In general, this argument does not massively improve speed until
    the estimating equations consist of hundreds of parameters.

    Attributes
    ----------
    theta : ndarray
        Parameters for estimating equations. Becomes the estimated parameters after `.estimate()` is called
    variance : ndarray
        Covariance matrix for the parameters, in the same order as the parameter vector
    asymptotic_variance : ndarray
        Asymptotic covariance matrix for the parameters. Not scaled by :math:`n`
    bread : ndarray
        Bread matrix for the parameter vector
    meat : ndarray
        Meat matrix for the parameter vector
    weight_matrix : ndarray
        Weight matrix, :math:`\text{Q}` used. For just-identified problems, the weight matrix is simply the identity
        matrix, otherwise the weight matrix is updated to the meat matrix from the prior iteration.

    Examples
    --------
    Loading necessary functions and building a generic data set for estimation of the mean

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import logistic
    >>> from delicatessen import GMMEstimator
    >>> from delicatessen.estimating_equations import ee_mean_variance

    >>> y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]

    GMM-estimation with built-in estimating equation for the mean and variance. First, ``psi``, or the stacked
    estimating equations, is defined

    >>> def psi(theta):
    >>>     return ee_mean_variance(theta=theta, y=y_dat)

    Calling the GMM-estimation procedure

    >>> estr = GMMEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate()

    Inspecting the output results

    >>> estr.theta                                  # Point estimates
    >>> estr.variance                               # Covariance
    >>> estr.asymptotic_variance                    # Asymptotic covariance
    >>> np.sqrt(np.diag(estr.asymptotic_variance))  # Standard deviation
    >>> estr.variance                               # Covariance
    >>> np.sqrt(np.diag(estr.variance))             # Standard error
    >>> estr.confidence_intervals()                 # Confidence intervals
    >>> estr.z_scores()                             # Z-scores
    >>> estr.p_values()                             # P-values
    >>> estr.s_values()                             # S-values

    For more examples on how to apply ``GMMEstimator``, see https://deli.readthedocs.io/en/latest/

    References
    ----------
    Duncan GM. (1987). A simplified approach to M-estimation with application to two-stage estimators.
    *Journal of Econometrics*, 34(3), 373-389.

    Hansen LP. (1982). Large sample properties of generalized method of moments estimators.
    *Econometrica: Journal of the Econometric Society*, 1029-1054.

    Jesus J, & Chandler RE. (2011). Estimating functions and the generalized method of moments. *Interface Focus*,
    1(6), 871-885.

    Wirjanto T. (1997). Estimating Functions and Over-Identified Models. *Lecture Notes - Monograph Series*, 239-256.
    """
    def __init__(self, stacked_equations, init, subset=None, overid_maxiter=10, overid_tolerance=1e-9):
        # Calling BaseClass for the initial setup
        super().__init__(stacked_equations=stacked_equations, init=init, subset=subset)

        # Processing GMM-specific arguments
        self._over_id_iterations_ = overid_maxiter
        self._over_id_tolerance_ = overid_tolerance

    def estimate(self, solver='bfgs', maxiter=5000, tolerance=1e-9, deriv_method='approx', dx=1e-9, allow_pinv=True):
        """Run the point and variance estimation procedures for given estimating equation and starting values. This
        function carries out the point and variance estimation of ``theta``. After this procedure, the point estimates
        (in ``theta``) and the covariance matrix (in ``variance``) can be extracted from the ``GMMEstimator`` object.

        Parameters
        ----------
        solver : str, function, callable, optional
            Method to use for the minimization procedure. Default is the BFGS algorithm
            (``scipy.optimize.minimize(method='bfgs')``, specified by ``solver='bfgs'``). Other built-in options are
            the L-BFGS-B algorithm (``'l-bfgs-b'``),
            the conjugate gradient algorithm (``'cg'``),
            Nelder-Mead (``'nelder-mead'``),
            and the modified Powell algorithm (``'powell'``).
            Finally, any generic minimization algorithm can be used via a user-provided callable object. The function
            must consist of two keyword arguments: ``stacked_equations``, and ``init``. Additionally, the function
            should return only the optimized values. Please review the provided example in the documentation for how to
            implement a custom minimization algorithm.
        maxiter : int, optional
            Maximum iterations to consider for the minimization procedure. Default is 5000 iterations. For complex
            estimating equations, this value may need to be increased. This argument is not used when a custom
            minimization method (e.g., ``solver``) is provided.
        tolerance : float, optional
            Maximum tolerance for errors in the minimization in ``scipy.optimize``. Default is 1e-9. This argument is
            not used when a custom minimization method (e.g., ``solver``) is provided.
        deriv_method : str, optional
            Method to compute the derivative of the estimating equations for the bread matrix. Default is ``'approx'``.
            Options include numerical approximation via the forward difference method via SciPy (``'approx'``), forward
            difference as implemented in delicatessen (`'fapprox'`), backward difference as implemented in delicatessen
            (`'bapprox'`), central difference implemented as in delicatessen (`'capprox'`), or forward-mode automatic
            differentiation as implemented in delicatessen(``'exact'``).
        dx : float, optional
            Spacing to use to numerically approximate the partial derivatives of the bread matrix. Default is 1e-9.
            Here, a small value for ``dx`` should be used, since some large values can result in poor approximations.
            This argument is only used with numerical approximation methods.
        allow_pinv : bool, optional
            Whether to allow for the pseudo-inverse (via ``numpy.linalg.pinv``) if the bread matrix is determined to be
            non-invertible. If you want to disallow the pseudo-inverse (i.e., use ``numpy.linalg.inv``), set this
            argument to ``False``. Default is ``True``, which  is more robust to the possible bread matrices.

        Returns
        -------
        None
        """
        # Evaluate stacked estimating equations at init
        vals_at_init = self.stacked_equations(theta=self.init)    # Calculating the initial values
        vals_at_init = np.asarray(vals_at_init                    # Convert output to an array (in case it isn't)
                                  ).T                             # ... transpose so N is always the 1st element

        # Error checking for inputs
        self._error_checker_(vals_at_init=vals_at_init)

        # Processing dependent on number of estimating functions
        if vals_at_init.ndim == 1:
            n_params = 1
            over_identified = False
            if np.asarray(self.init).shape[0] > 1:
                raise ValueError(
                    "The number of initial values should be less than or equal to the number of rows "
                    "returned by `stacked_equations` but there are " + str(np.asarray(self.init).shape[0])
                    + " initial values and the `stacked_equations` function returns 1 row.")
        # If estimating functions are of 2 dimensions, need to check for over-identification
        elif vals_at_init.ndim == 2:
            n_params = vals_at_init.shape[1]                        # Getting the number of parameters
            if np.asarray(self.init).shape[0] == n_params:          # Finding if the problem is just-identified
                over_identified = False                             # ... flag for over-ID iteration procedure
            elif np.asarray(self.init).shape[0] < n_params:         # Finding if the problem is over-identified
                over_identified = True                              # ... flag for over-ID iteration procedure
            else:                                                   # Under-identified parameters are not supported
                raise ValueError(
                    "The number of initial values should be less than or equal to the number of rows "
                    "returned by `stacked_equations` but there are " + str(np.asarray(self.init).shape[0])
                    + " initial values and the `stacked_equations` function returns "
                    + str(vals_at_init.shape[1]) + " row(s).")

        # Catching if user provided invalid estimating function shape
        elif vals_at_init.ndim > 2:
            raise ValueError("A 2-dimensional array is expected, but the `stacked_equations` returns a "
                             + str(vals_at_init.ndim) + "-dimensional array.")

        # Nothing should arrive here via logic of previous operations
        else:
            raise ValueError("This ValueError should not be reached by the structure of the problem. If you receive "
                             "this error, please reach out https://github.com/pzivich/Delicatessen")

        # Trick to get the number of observations from the estimating equations (transposed above)
        self.n_obs = vals_at_init.shape[0]

        # Obtaining initial weight matrix
        self.weight_matrix = np.identity(n=n_params)

        # Updating theta to init values for minimization call (since needed in _gmmestimation_answer_)
        self.theta = self.init

        # STEP 1: solving the GMM-estimator stacked equations
        # To allow for optimization of only a subset of parameters in the estimating equation (in theory meant to
        #   simplify the process of complex stacked estimating equations where pre-washing can be done effectively),
        #   we do some internal processing. Essentially, we 'freeze' the parameters outside of self._subset_ as their
        #   inits, and let the minimization procedure update the self._subset_ parameters. We do this by subsetting out
        #   the init values then passing them along to solver. Behind the scenes, self._solve_coefficients_() expands
        #   the parameters (to include everything), calculates the estimating equation at those values, and then
        #   extracts the corresponding subset.
        #   This process only takes place within Step 1. There is an inherent danger with this process in that if
        #   non-subset parameters are not pre-washed, then the returned parameters will not be correct. I am
        #   considering adding a warning for self_subset_, but I currently just trust the user...
        # STEP 1.1: Initial solving procedure for point estimation
        self._solve_for_theta_(solver=solver, maxiter=maxiter, tolerance=tolerance, init=self.theta)

        # STEP 1.2: Iterative procedure if we are in the over-identified setting
        if over_identified:
            # Iterative process argument processing
            current_iteration = 0
            if self._over_id_iterations_ < 1:
                terminate = True
            else:
                terminate = False

            # Applying the iterative procedure
            while not terminate:
                # Setup for current iteration
                current_iteration = current_iteration + 1
                prev_theta = np.asarray(self.theta)
                self.weight_matrix = self._get_q_matrix_(allow_pinv=allow_pinv)
                self._solve_for_theta_(solver=solver, maxiter=maxiter,
                                       tolerance=tolerance, init=self.theta)

                # Checking ending conditions
                error = np.abs(np.asarray(self.theta) - prev_theta)
                if np.max(error) <= self._over_id_tolerance_:
                    terminate = True
                if current_iteration >= self._over_id_iterations_:
                    terminate = True
                    warnings.warn("Maximum number of iterations, `overid_maxiters=" + str(self._over_id_iterations_)
                                  + "`, has been exceeded for the iterative GMM updating. Terminating the "
                                  + "over-identification procedure at the last estimated values.",
                                  UserWarning)

        # STEP 2: calculating the sandwich variance
        # After solving for the parameters, we now can compute the empirical sandwich variance estimator. This is
        #   done by compute the bread and meat matrices and then combining them. This is now done by a separate
        #   functionalities within the `sandwich.py` file as of v2.2.
        # STEP 2.1: baking the Bread
        self.bread = compute_bread(stacked_equations=self.stacked_equations,
                                   theta=self.theta,
                                   deriv_method=deriv_method,
                                   dx=dx) / self.n_obs

        # STEP 2.2: slicing the meat
        self.meat = compute_meat(stacked_equations=self.stacked_equations,
                                 theta=self.theta) / self.n_obs

        # STEP 2.3: assembling the sandwich (variance)
        self.asymptotic_variance = build_sandwich(bread=self.bread,
                                                  meat=self.meat,
                                                  allow_pinv=allow_pinv)
        if self.asymptotic_variance is None:
            self.variance = self.asymptotic_variance
        else:
            self.variance = self.asymptotic_variance / self.n_obs

    def _gmmestimation_answer_(self, theta):
        """Internal function to evaluate the function to be minimized over. The summation is internally evaluated
        since access to the estimating functions is needed for the sandwich variance computations. This function is
        used by the minimization procedure (since we need the subset applied).

        Parameters
        ----------
        theta : array
            b-by-n matrix to sum over the values of n.

        Returns
        -------
        array :
            b-by-1 array, which is the sum over n for each b.
        """
        # Option for the subset argument
        if self._subset_ is None:                      # If NOT subset then,
            full_theta = theta                         # ... then use the full input theta
        else:                                          # If subset then,
            full_theta = np.asarray(self.theta)        # ... copy the initial values to ndarray
            np.put(full_theta,                         # ... update in place the previous array
                   ind=self._subset_,                  # ... go to the subset indices
                   v=theta)                            # ... then input current iteration values

        # Evaluating estimating functions
        stacked_equations = np.asarray(self.stacked_equations(full_theta))  # Returning stacked equation
        est_eqtns = self._eval_ee_(stacked_equations=stacked_equations,     # Passing to evaluating function
                                   subset=None)                             # ... with specified subset
        est_eqtns = est_eqtns / self.n_obs                                  # Scaling EE by n (helpful for minimizer)

        # Evaluating GMM estimator at given theta
        return np.dot(np.dot(est_eqtns, self.weight_matrix), est_eqtns)

    def _solve_for_theta_(self, solver, maxiter, tolerance, init):
        """Internal function to evaluate and solve the estimating equations for theta. This is expressed as an internal
        function to simplify the iterative procedure for over-identified problems

        Parameters
        ----------
        solver : str, function, callable, optional
            Method to use for the minimization procedure.
        maxiter : int, optional
            Maximum iterations to consider for the minimization procedure.
        tolerance : float, optional
            Maximum tolerance for errors.

        Returns
        -------
        None
        """
        # TODO I added init as an argument rather than calling self.init. I want to think about this more
        if self._subset_ is None:                        # If NOT subset,
            inits = init                                 # ... give all initial values
        else:                                            # If subset,
            inits = np.take(init, self._subset_)         # ... then extract initial values for the subset

        # Calculating parameters values via the minimizer (for only the subset of values!)
        slv_theta = self._solve_coefficients_(stacked_equations=self._gmmestimation_answer_,
                                              init=inits,
                                              method=solver,
                                              maxiter=maxiter,
                                              tolerance=tolerance)

        # Processing parameters after the minimization procedure
        if self._subset_ is None:                        # If NOT subset,
            self.theta = slv_theta                       # ... then use the full output/solved theta
        else:                                            # If subset,
            self.theta = np.asarray(self.theta)          # ... copy the initial values
            for s, n in zip(self._subset_, slv_theta):   # ... then look over the subset and input theta
                self.theta[s] = n                        # ... and update the subset to the output/solved theta

    def _get_q_matrix_(self, allow_pinv):
        """Internal function to compute the weight matrix Q for over-identified problems

        Returns
        -------

        """
        # Compute the outer product (meat matrix)
        meat_matrix = compute_meat(stacked_equations=self.stacked_equations, theta=self.theta) / self.n_obs

        # Invert meat matrix
        if allow_pinv:
            q_matrix = np.linalg.pinv(meat_matrix)
        else:
            q_matrix = np.linalg.inv(meat_matrix)
        return q_matrix

    @staticmethod
    def _solve_coefficients_(stacked_equations, init, method, maxiter, tolerance):
        """Calls the minimization procedure for the values of theta, such that the estimating equations are equal to
        zero. Default uses the BFGS algorithm from SciPy.

        Parameters
        ----------
        stacked_equations : function
            Function that contains the estimating equations
        init : array
            Initial values for the optimizer
        method : str, function, callable
            Method to use for the minimization procedure. Can be either a string or a callable object
        maxiter : int
            Maximum iterations to consider for the minimization procedure
        tolerance : float
            Maximum tolerance for errors in the minimization. This is SciPy's `tol` argument.

        Returns
        -------
        psi: array
            Solved or optimal values for theta given the estimating equations and data
        """
        # Default SciPy minimizers provided that are valid to use here
        if method in ['cg', 'bfgs', 'l-bfgs-b', 'powell', 'nelder-mead']:
            opt = minimize(stacked_equations,            # ... stacked equations to solve (should be written as sums)
                           x0=np.asarray(init),          # ... initial values for solver
                           method=method,                # ... chosen minimization method
                           options={"maxiter": maxiter,  # ... setting maximum number of iterations
                                    "disp": False},      # ... don't display results (SciPy prints stuff on success)
                           tol=tolerance)                # ... setting some tolerance values (should be able to update)
            if opt.status != 0:                                # Check whether optimization was successful
                warnings.warn(opt.message, OptimizeWarning)    # ... if not, throw the SciPy standard warning to user
            psi = opt.x                                        # Extract the estimated parameter vector

        # Callable custom minimizer provided by the user
        # TODO still need to see if pytorch works here
        elif callable(method):
            try:
                psi = method(stacked_equations=stacked_equations,
                             init=np.asarray(init))
            except TypeError:
                raise TypeError("The user-specified minimizer `solver` must be a function (or callable object) with "
                                "the following keyword arguments: `stacked_equations`, `init`.")
            if psi is None:
                raise ValueError("The user-specified minimizer `solver` must return the solution to the "
                                 "optimization")

        # Value error for unsupported solvers
        else:
            # ... otherwise throw ValueError if no other minimizer steps are triggered.
            raise ValueError("The solver '" + str(method) + "' is not available. Please see the "
                             "documentation for valid options for the optimizer.")

        # Returning the theta estimates
        return psi
