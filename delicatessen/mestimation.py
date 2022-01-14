import numpy as np
from scipy.optimize import newton, root
from scipy.misc import derivative

from delicatessen.utilities import partial_derivative


class MEstimator:
    r"""Generalized M-Estimator for stacked estimating equations.

    M-estimation, or loosely referred to as estimating equations, is a general approach to point and variance
    estimation that consists of defining an estimator as the solution to an estimating equation (but does not require
    the derivative of a log-likelihood function). M-estimators satisify the following constraint

    .. math::

        \sum_{i=1}^n \psi(Y_i, \hat{\theta}) = 0

    Note
    ----
    One advantage of M-Estimation is that many estimators can be written as M-Estimators. This simplifies theoretical
    analysis and application under a large-sample approximation framework.


    M-Estimation consists of two broad step: point estimation and variance estimation. Point estimation is carried out
    by determining at which values of theta the given estimating equations are equal to zero. This is done via SciPy's
    `newton` algorithm by default.

    For variance estimation, the sandwich asymptotic variance estimator is used, which consists of

    .. math::

        B_n(Y, \hat{\theta})^{-1} \times F_n(Y, \hat{\theta}) \times B_n(Y, (\hat{\theta})^{-1})^T

    where B is the bread and F is the filling

    .. math::

        B_n(Y, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} - \psi'(Y_i, \hat{\theta})

    .. math::

        F_n(Y, \hat{\theta}) = n^{-1} \sum_{i=1}^{n} \psi(Y_i, \hat{\theta}) \times \psi(Y_i, \hat{\theta})^T

    The partial derivatives for the bread are calculated using an adaptation of SciPy's ``derivative`` functionality
    for partial derivatives. Inverting the bread is done via NumPy's ``linalg.pinv``. For the filling, the dot product
    is taken for the evaluated theta's.

    Note
    ----
    A hard part (that must be done by the user) is to specify the stacked estimating equations. Be sure to check
    the provided examples for the format. Pre-built estimating equations for common problems are available to ease
    burden.

    After completion of these steps, point and variance estimates for theta stored. These can be directly pulled from
    the class object and further manipulated. For example, calculation of 95% confidence intervals.

    Note
    ----
    For complex regression problems, the optimizer behind the scenes is not particularly robust (unlike functions
    specializing in solely logistic regression). Therefore, pre-washed values can be fed forward as the initial values
    (which should result in a more stable optimization).

    Parameters
    ----------
    stacked_equations : function, callable
        Function that returns a b-by-n NumPy array of the estimating equations. See documentation for how to construct
        a set of estimating equations.
    init : list, set, array
        Initial values to optimize for the function.

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

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> mestimation.estimate()

    Inspecting the parameter estimates, the variance, and the asymptotic variance

    >>> mestimation.theta
    >>> mestimation.variance
    >>> mestimation.asymptotic_variance

    Alternatively, a custom estimating equation can be specified. This is done by constructing a valid estimating
    equation for the ``MEstimator``. The ``MEstimator`` expects the `psi` function to return a b-by-n array, where b is
    the number of parameters (length of theta) and n is the total number of observations. Below is an example of the
    mean and variance estimating equation from before

    >>> def psi(theta):
    >>>     y = np.array(y_dat)
    >>>     piece_1 = y - theta[0]
    >>>     piece_2 = (y - theta[0]) ** 2 - theta[1]
    >>>     return piece_1, piece_2

    The M-estimation procedure is called using the same approach as before

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> mestimation.estimate()

    Note that ``len(init)`` should be equal to b. So in this case, two initial values are provided.

    # TODO provide custom root-finding example here.
    Finally, the M-Estimator can also be run with a user-provided root-finding algorithm. To specify a custom
    root-finder, a function must be created by the user that consists of two keyword arguments (``stacked_equations``,
    ``init``) and must return only the optimized values. The following is an example with SciPy's Levenberg-Marquardt
    algorithm in ``root``.

    >>> def custom_solver(stacked_equations, init):
    >>>     options = {"maxiter": 1000}
    >>>     opt = root(stacked_equations,
    >>>                x0=np.asarray(init), method='lm', tol=1e-9,
    >>>                options=options)
    >>>     return opt.x

    The provided custom root-finder can then be implemented like the following (continuing with the estimating equation
    from the previous example):

    >>> mestimation = MEstimator(stacked_equations=psi, init=[0, 0, ])
    >>> mestimation.estimate(solver=custom_solver)

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.

    Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. The American Statistician, 56(1), 29-38.
    """
    def __init__(self, stacked_equations, init=None):
        self.stacked_equations = stacked_equations     # User-input stacked estimating equations
        self.init = init                               # User-input initial starting values for solving estimating eqs

        # Storage for results from the M-Estimation procedure
        self.n_obs = None                 # Number of unique observations (calculated later)
        self.theta = None                 # Optimized theta values (calculated later)
        self.bread = None                 # Bread from theta values (calculated later)
        self.meat = None                  # Meat from theta values (calculated later)
        self.variance = None              # Covariance matrix for theta values (calculated later)
        self.asymptotic_variance = None   # Asymptotic covariance matrix for theta values (calculated later)

    def estimate(self, solver='newton', maxiter=1000, tolerance=1e-9, dx=1e-9, allow_pinv=True):
        """Function to carry out the point and variance estimation of theta. After this procedure, the point estimates
        (in ``theta``) and the covariance matrix (in ``variance``) can be extracted.

        Parameters
        ----------
        solver : str, function, callable, optional
            Method to use for the root finding procedure. Default is the secant method (``scipy.optimize.newton``).
            Another built-in option is the Levenberg-Marquardt algorithm (``scipy.optimize.root(method='lm')``).
            Finally, any generic root-finding algorithm can be used via a user-provided callable object (function).
            The function should consist of two keyword arguments: ``stacked_equations``, and ``init``. Additionally,
            the function should return only the optimized values. Please review the example in the documentation or on
            ReadTheDocs for how to provide a custom root-finding algorithm.
        maxiter : int, optional
            Maximum iterations to consider for the root finding procedure. Default is 1000 iterations. For complex
            estimating equations (without preceding optimization), this value may need to be increased. This argument
            is not used for user-specified solvers
        tolerance : float, optional
            Maximum tolerance for errors in the root finding. This argument is passed ``scipy.optimize`` via the
            ``tol`` parameter. Default is 1e-9, which I have seen good performance with. I do not recommend going below
            this tolerance level (at this time). This argument is not used for user-specified solvers
        dx : float, optional
            Spacing to use to numerically approximate the partial derivatives of the bread matrix. Default is 1e-9,
            which should work well for most applications. It is generally not recommended to increase dx, since some
            large values can poorly approximate derivatives.
        allow_pinv : bool, optional
            The default is ``True`` which uses ``numpy.linalg.pinv`` to find the inverse (or pseudo-inverse if matrix is
            non-invertible) for the bread. This default option is more robust to the possible matrices. If you want
            to use ``numpy.linalg.inv`` instead (which does not support pseudo-inverse), set this parameter to False.

        Returns
        -------
        None
        """
        # Trick to get the number of observations from the estimating equations
        self.n_obs = np.asarray(self.stacked_equations(theta=self.init)  # ... convert output to an array
                                ).T.shape[0]                             # ... transpose so N is always the 1st element

        # Step 1: solving the M-estimator stacked equations
        self.theta = self._solve_coefficients_(stacked_equations=self._mestimation_answer_,  # Give the EE's
                                               init=self.init,                               # Give the initial vals
                                               method=solver,                                # Specify the solver
                                               maxiter=maxiter,                              # Set max iterations
                                               tolerance=tolerance                           # Set allowable tolerance
                                               )

        # Step 2: calculating Variance
        # Step 2.1: baking the Bread
        self.bread = self._bread_matrix_(theta=self.theta,                           # Use inner function for bread
                                         stacked_equations=self.stacked_equations,
                                         dx=dx) / self.n_obs                         # ... and divide by n

        # Step 2.2: slicing the meat
        evald_theta = np.asarray(self.stacked_equations(theta=self.theta))  # Evaluating EE at the optim values of theta
        self.meat = np.dot(evald_theta, evald_theta.T) / self.n_obs         # Meat is a simple dot product of two arrays

        # Step 2.3: assembling the sandwich (variance)
        if self.bread.ndim == 0:                        # NumPy's linalg throws an error if bread is a single value
            bread_invert = 1 / self.bread               # ... so directly take inverse of the single value
        else:                                           # otherwise bread must be a matrix
            if allow_pinv:
                bread_invert = np.linalg.pinv(self.bread)   # ... so find the inverse (or pseudo-inverse)
            else:
                bread_invert = np.linalg.inv(self.bread)    # ... so find the inverse (NOT pseudo-inverse)
        # Two sets of matrix multiplication to get the sandwich variance
        sandwich = np.dot(np.dot(bread_invert, self.meat), bread_invert.T)

        # Step 3: updating storage for results
        self.asymptotic_variance = sandwich       # Asymptotic variance estimate requires division by n (done above)
        self.variance = sandwich / self.n_obs     # Variance estimate requires division by n^2

    def _mestimation_answer_(self, theta):
        """Internal function to evaluate the sum of the estimating equations. The summation must be internally evaluated
        since the bread requires calculation of the sum of the derivatives.

        Parameters
        ----------
        theta : array
            b-by-n matrix to sum over the values of n.

        Returns
        -------
        array :
            b-by-1 array, which is the sum over n for each b.
        """
        stacked_equations = np.asarray(self.stacked_equations(theta))  # Returning stacked equation

        if len(stacked_equations.shape) == 1:        # If stacked_equation returns 1 value, only return that 1 value
            vals = np.sum(stacked_equations)         # ... avoids SciPy error by returning value rather than tuple
        else:                                        # Else return a tuple for optimization
            vals = ()                                # ... create empty tuple
            for i in self.stacked_equations(theta):  # ... go through each individual theta
                vals += (np.sum(i),)                 # ... add the theta sum to the tuple of thetas

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

        return psi                             # Return optimized theta array

    @staticmethod
    def _bread_individual_(theta, variable_index, output_index, stacked_equations, dx):
        """Calculate the partial derivative for a cell of the bread matrix. Transforms the partial derivative by taking
        the negative sum.

        Parameters
        ----------
        theta
            Solved values of theta to evaluate at
        variable_index
            Index of the input theta (e.g., variable_index=1 means that derivative of theta[1] is approximated).
        output_index
            Index of the output theta (e.g., out_index=1 means that theta[1] is output).
        stacked_equations
            Function containing the estimating equations
        dx : float
            Spacing to use to numerically approximate the partial derivatives of the bread matrix.

        Returns
        -------
        float
        """
        # Calculate the partial derivatives for a single i,j of the bread matrix
        d = partial_derivative(stacked_equations,    # ... stacked estimating equations
                               var=variable_index,   # ... index for the theta of interest
                               point=theta,          # ... point to evaluate the derivative at
                               output=output_index,  # ... index location to output
                               dx=dx)                # ... spacing for derivative approximation
        return -1 * np.sum(d)                        # Calculate the bread for i,j

    def _bread_matrix_(self, theta, stacked_equations, dx):
        """Evaluate the bread matrix by taking all partial derivatives of the thetas in the estimating equation.

        Parameters
        ----------
        theta : ndarray, float
            Solved values of theta to evaluate at
        stacked_equations : function
            Function containing the estimating equations
        dx : float
            Spacing to use to numerically approximate the partial derivatives of the bread matrix.

        Returns
        -------
        numpy.array
        """
        # Check how many values of theta there is
        val_range = len(theta)

        # Evaluate the bread matrix
        if val_range == 1:                                       # When only a single theta is present
            d = derivative(stacked_equations, theta, dx=dx)      # ... approximate the derivative
            return -1 * np.sum(d)                                # ... then return negative sum
        else:                                                    # Otherwise approximate the partial derivatives
            bread_matrix = np.empty((val_range, val_range))      # ... create empty matrix
            for i in range(val_range):                           # ... for each i in len(theta)
                for j in range(val_range):                       # ... for each j in len(theta)
                    b = self._bread_individual_(theta=theta,     # ... calculate the i,j cell value by -sum part-deriv
                                                variable_index=i,
                                                output_index=j,
                                                stacked_equations=stacked_equations,
                                                dx=dx)
                    bread_matrix[j, i] = b                       # ... update bread matrix value with new
            return bread_matrix                                  # Return completed bread matrix
