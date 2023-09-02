import numpy as np
from scipy.optimize import newton, root
from scipy.misc import derivative
from scipy.optimize import approx_fprime
from scipy.stats import norm

from delicatessen.derivative import auto_differentiation


class MEstimator:
    r"""M-Estimator for stacked estimating equations.

    M-Estimation, or loosely referred to as estimating equations, is a general approach to point and variance
    estimation that consists of defining an estimator as the solution to an estimating equation. M-estimators
    satisify the following

    .. math::

        \sum_{i=1}^n \psi(O_i, \hat{\theta}) = 0

    where :math:`\psi` is the :math:`v`-dimensional vector of estimating equation(s), :math:`\hat{\theta}` is the
    :math:`v`-dimensional parameter vector, and :math:`O_i` is the observed data (where units are independent but not
    necessarily identically distributed).

    Note
    ----
    M-Estimation is advantageous in both theoretical and applied research. M-estimation simplifies proofs of
    consistency and asymptotic normality of estimators under a large-sample approximation framework. In application,
    M-estimators simplify estimation of the variance of parameters and automate the delta-method.


    M-Estimation consists of two broad step: point estimation and variance estimation. Point estimation is carried out
    by determining the values of :math:`\theta` where the sum of the estimating equations are zero. For variance
    estimation, the asymptotic sandwich variance estimator is used, which consists of

    .. math::

        B_n(O, \hat{\theta})^{-1} F_n(O, \hat{\theta}) \left\{B_n(O, \hat{\theta}^{-1})\right\}^T

    where :math:`B` is the 'bread' and :math:`F` is the 'filling'

    .. math::

        B_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} - \psi'(O_i, \hat{\theta})

    .. math::

        F_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} \psi(O_i, \hat{\theta}) \psi(O_i, \hat{\theta})^T

    The partial derivatives for the bread are calculated using either numerical approximation (i.e., central difference
    method) or forward-mode automatic differentiation. Inverting the bread is done via NumPy's ``linalg.pinv``. For
    the filling, the dot product is taken at :math:`\hat{\theta}`.

    Note
    ----
    A hard part, that must be done by the user, is to specify the estimating equations. Be sure to check the provided
    examples for the expected format. Pre-built estimating equations for common problems are also made available.


    After completion of these steps, point and variance estimates are stored. These can be extracted from
    ``MEstimator``.

    Note
    ----
    For complex regression problems, the root-finding algorithms are not as robust relative to maximization approaches.
    A solution for difficult problems is to 'pre-wash' the initial values.

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a b-by-n NumPy array of the estimating equations. See provided examples in the
        documentation for how to construct a set of estimating equations.
    init : list, set, array
        Initial values for the root-finding algorithm.
    subset : list, set, array, None, optional
        Optional argument to conduct the root-finding procedure on a subset of parameters in the estimating equations.
        The input list is used to location index the parameter array via ``np.take()``. The subset list will
        only affect the root-finding procedure (i.e., the sandwich variance estimator ignores the subset argument).
        Default is ``None``, which runs the root-finding procedure for all parameters in the estimating equations.

    Note
    ----
    Because the root-finding procedure is NOT ran for parameters outside of the subset, those coefficients must be
    solved outside of ``MEstimator``. In general, I do NOT recommend using the ``subset`` argument unless a series of
    complex estimating equations need to be solved.

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
    >>> np.sqrt(np.diag(estr.variance))             # Standard error
    >>> estr.confidence_intervals()                 # Confidence intervals

    Alternatively, a custom estimating equation can be specified. This is done by constructing a valid estimating
    equation for the ``MEstimator``. The ``MEstimator`` expects the ``psi`` function to return a b-by-n array, where b
    is the number of parameters (length of ``theta``) and n is the total number of observations. Below is an example
    of the mean and variance estimating equation from before

    >>> def psi(theta):
    >>>     y = np.array(y_dat)
    >>>     mean = y - theta[0]
    >>>     variance = (y - theta[0]) ** 2 - theta[1]
    >>>     return mean, variance

    The M-estimation procedure is called using the same approach as before

    >>> estr = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> estr.estimate()

    Note that ``len(init)`` should be equal to b. So in this case, two initial values are provided.

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

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. The American Statistician, 56(1), 29-38.
    """
    def __init__(self, stacked_equations, init=None, subset=None):
        self.stacked_equations = stacked_equations     # User-input stacked estimating equations
        self.init = init                               # User-input initial starting values for solving estimating eqs
        if subset is None:                             # Handling subset of parameters
            self._subset_ = subset                     # ... when None, set as None
        else:                                          # Otherwise
            self._subset_ = sorted(subset)             # ... ensure it is sorted

        # Storage for results from the M-Estimation procedure
        self.n_obs = None                 # Number of unique observations (calculated later)
        self.theta = None                 # Optimized theta values (calculated later)
        self.bread = None                 # Bread from theta values (calculated later)
        self.meat = None                  # Meat from theta values (calculated later)
        self.variance = None              # Covariance matrix for theta values (calculated later)
        self.asymptotic_variance = None   # Asymptotic covariance matrix for theta values (calculated later)

    def estimate(self, solver='newton', maxiter=1000, tolerance=1e-9, deriv_method='approx', dx=1e-9, allow_pinv=True):
        """Function to carry out the point and variance estimation of theta. After this procedure, the point estimates
        (in ``theta``) and the covariance matrix (in ``variance``) can be extracted.

        Parameters
        ----------
        solver : str, function, callable, optional
            Method to use for the root-finding procedure. Default is the secant method (``scipy.optimize.newton``).
            Other built-in option is the Levenberg-Marquardt algorithm (``scipy.optimize.root(method='lm')``), and
            a modification of the Powell hybrid method (``scipy.optimize.root(method='hybr')``). Finally, any generic
            root-finding algorithm can be used via a user-provided callable object. The function must
            consist of two keyword arguments: ``stacked_equations``, and ``init``. Additionally, the function should
            return only the optimized values. Please review the provided example in the documentation for how to
            implement a custom root-finding algorithm.
        maxiter : int, optional
            Maximum iterations to consider for the root finding procedure. Default is 1000 iterations. For complex
            estimating equations (without preceding optimization), this value may need to be increased. This argument
            is not used for user-specified solvers.
        tolerance : float, optional
            Maximum tolerance for errors in the root finding. This argument is passed ``scipy.optimize`` via the
            ``tol`` parameter. This argument is not used for user-specified solvers. Default is 1e-9.
        deriv_method : str, optional
            Method to compute the derivative of the estimating equations for the bread matrix. Options include numerical
            approximation via the central difference method (``'approx'``) and forward-mode automatic differentiation
            (``'exact'``). Default is ``'approx'``.
        dx : float, optional
            Spacing to use to numerically approximate the partial derivatives of the bread matrix. Here, a small value
            for ``dx`` should be used, since some large values can result in poor approximations. This argument is only
            used when ``deriv_method='approx'``. Default is 1e-9.
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
                             "https://deli.readthedocs.io/en/latest/Custom%20Equations.html#handling-np-nan")

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
        #   This process only takes place within Step 1 (the sandwich variance did not require any corresponding
        #   updates). There is an inherent danger with this process in that if non-subset parameters are not pre-washed,
        #   then the returned parameters will not be correct. I am considering adding a warning for self_subset_, but I
        #   may just have to trust the user...

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

        # STEP 2: calculating Variance
        # STEP 2.1: baking the Bread
        self.bread = self._bread_matrix_(theta=self.theta,                           # Provide theta-hat
                                         method=deriv_method,                        # Method to use
                                         dx=dx) / self.n_obs                         # Derivative approximation value

        # STEP 2.2: slicing the meat
        evald_theta = np.asarray(self.stacked_equations(theta=self.theta))           # Evaluating EE at theta-hat
        self.meat = np.dot(evald_theta, evald_theta.T) / self.n_obs                  # Meat is dot product of arrays

        # STEP 2.3: assembling the sandwich (variance)
        if allow_pinv:                                                               # Support 1D theta-hat
            bread_invert = np.linalg.pinv(self.bread)                                # ... find pseudo-inverse
        else:                                                                        # Support 1D theta-hat
            bread_invert = np.linalg.inv(self.bread)                                 # ... find inverse
        # Two sets of matrix multiplication to get the sandwich variance
        sandwich = np.dot(np.dot(bread_invert, self.meat), bread_invert.T)

        # STEP 3: updating storage for results
        self.asymptotic_variance = sandwich       # Asymptotic variance requires division by n (done above)
        self.variance = sandwich / self.n_obs     # Variance estimate requires division by n^2 (second done here)

    def confidence_intervals(self, alpha=0.05):
        r"""Calculate Wald-type :math:`(1 - \alpha) \times` 100% confidence intervals using the point estimates and
        the sandwich variance. The formula for the confidence intervals are

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
        # Check that estimate() has been called
        if self.theta is None:
            raise ValueError("theta has not been estimated yet. "
                             "estimate() must be called before confidence_intervals()")
        # Check valid alpha value is being provided
        if not 0 < alpha < 1:
            raise ValueError("`alpha` must be 0 < a < 1")

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
        calculated.

        Parameters
        ----------
        null: int, float, ndarray, optional
            Null or reference for the the corresponding P-values. Default is 0.

        Returns
        -------
        array :
            Array of Z-scores for :math:`\theta_1, ..., \theta_b`, respectively
        """
        # Check that self.estimate() has been called
        if self.theta is None:
            raise ValueError("theta has not been estimated yet. "
                             "estimate() must be called before confidence_intervals()")

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
        interpretable as it can be related back to simple coin-flipping scenarios.

        Suppose the S-value is :math:`s`. Then :math:`s` corresponds to the number of heads in a row with a fair coin
        (equal chances heads or tails). As :math:`s` increases, one would be more 'surprised' by the result (e.g., it
        might not be surprising to have two heads in a row, but it would be surprising for 10 in a row).

        The transformation from a P-value into a S-value is.

        .. math::

            S = - \log_2(P)

        where :math:`P` is the corresponding P-value. The ``.estimate()`` function must be called before the S-values
        can be calculated.

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
        return self._mestimator_sum_(stacked_equations=stacked_equations,   # Passing to evaluating function
                                     subset=self._subset_)                  # ... with specified subset

    def _mestimation_answer_no_subset_(self, theta):
        """Internal function to evaluate the sum of the estimating equations. The summation is internally evaluated
        since access to the estimating functions is needed for the sandwich variance computations. This function is
        used by the bread matrix computation procedure (since subset is ignored for the bread).

        Parameters
        ----------
        theta : array
            b-by-n matrix to sum over the values of n.

        Returns
        -------
        array :
            b-by-1 array, which is the sum over n for each b.
        """
        stacked_equations = np.asarray(self.stacked_equations(theta))       # Returning stacked equation
        return self._mestimator_sum_(stacked_equations=stacked_equations,   # Passing to evaluating function
                                     subset=None)                           # ... with always /no/ subset

    @staticmethod
    def _mestimator_sum_(stacked_equations, subset):
        """Function to evaluate the sum of the M-estimator over the :math:`n` units.

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
    def _solve_coefficients_(stacked_equations, init, method, maxiter, tolerance):
        """Quasi-Newton solver for the values of theta, such that the estimating equations are equal to zero. Default
        uses the secant method from SciPy's `newton` optimizer.

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
            raise ValueError("The solver '" +  # ... otherwise throw ValueError
                             str(method) +
                             "' is not available. Please see the "
                             "documentation for valid"
                             "options for the optimizer.")

        # Return optimized theta array
        return psi

    def _bread_matrix_(self, theta, method, dx):
        """Evaluate the bread matrix by taking all partial derivatives of the thetas in the estimating equation.

        Parameters
        ----------
        theta : ndarray, float
            Solved values of theta to evaluate at
        dx : float
            Spacing to use to numerically approximate the partial derivatives of the bread matrix.

        Returns
        -------
        numpy.array
        """
        val_range = len(theta)                                       # Check how many values of theta there is
        est_eq = self._mestimation_answer_no_subset_                 # Estimating equations to compute derivative of

        # Compute the derivative
        if method.lower() == "approx":                               # Numerical approximation method
            if val_range == 1:                                       # When only a single theta is present
                d = derivative(est_eq,                               # ... approximate the derivative
                               theta, dx=dx)                         # ... at the solved theta (input)
                bread_matrix = np.array([[d, ], ])                   # ... return as 1-by-1 array object for inversion
            else:                                                    # Otherwise approximate the partial derivatives
                bread_matrix = approx_fprime(xk=theta,               # ... use built-in jacobian functionality of SciPy
                                             f=est_eq,               # ... with not-subset estimating equations
                                             epsilon=dx)             # ... order option removed in v1.0

        elif method.lower() == "exact":                              # Automatic Differentiation
            bread_matrix = auto_differentiation(xk=theta,            # Compute the exact derivative at theta
                                                f=est_eq)            # ... for the given estimating equations

        else:                                                        # Error for unsupported option
            raise ValueError("The input for deriv_method was "
                             + str(method)
                             + ", but only 'approx' and 'exact' are available.")

        # Return bread (multiplied by negative 1 as in Stefanski & Boos)
        return -1 * bread_matrix
