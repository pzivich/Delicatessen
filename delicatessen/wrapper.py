import numpy as np

from delicatessen.estimation import MEstimator, GMMEstimator


class WaxPaperMEstimator:
    """Configuration template for higher-level interfaces based on MEstimator. This class is meant to provide a
    template for designing tools that use `delicatessen.MEstimator` for estimation. The motivation is to provide a
    unified framework for how developers and users can change or adapt model estimation steps across different packages
    that rely on `delicatessen` for the actual estimation but want to provide a higher-level interfaces. This can also
    be used to design wrapper tools for certain estimators a research group might regularly apply.

    # TODO add a table detailing all the options

    Returns
    -------
    None
    """
    # Default parameter settings (can be over-written)
    solver = 'lm'                         # Default solving method for root-finding
    maxiter = 5000                        # Default maximum iterations for root-finding
    tolerance = 1e-9                      # Default tolerance for root-finding
    deriv_method = 'approx'               # Default differentiation method for sandwich
    dx = 1e-9                             # Default approximation distance for sandwich
    allow_pinv = True                     # Default to allow for the pseudo-inverse of the bread
    alpha = 0.05                          # Save designated alpha for statistical inference
    null = 0.                             # Save designated null value for statistical inference

    # Storage (should not be over-written)
    estimator = None                      # Storage of the M-estimator for developer or user access

    def _fit_estimator_(self, psi, initial_values, subset=None, finite_correction=None):
        """Fit the corresponding M-estimator for the provided estimating functions and initial starting values.

        Parameters
        ----------
        psi
        initial_values

        Returns
        -------
        None
        """
        # Set up the M-estimator based on inputs
        estr = MEstimator(stacked_equations=psi,                 # Instantiate the MEstimator class with EFs
                          init=initial_values,                   # ... give the initial values for root-finding
                          subset=subset,                         # ... whether to optimize over only a subset
                          finite_correction=finite_correction)   # ... whether to apply finite correction to variance

        # Root-finding Procedure
        estr.estimate(solver=self.solver,                        # Root-finding algorithm to call
                      maxiter=self.maxiter,                      # ... maximum number of iterations to use
                      tolerance=self.tolerance,                  # ... error tolerance for the root-finding procedure
                      deriv_method=self.deriv_method,            # ... method to compute derivatives
                      dx=self.dx,                                # ... approximation error for derivatives
                      allow_pinv=self.allow_pinv)                # ... whether to allow for the pseudo-inverse

        # Updating stored estimator
        self.estimator = estr

    def _process_estimator_(self):
        """Process some common outputs from the estimator into an array. Labels for that array are also provided. These
        can be organized for easy display.

        Returns
        -------
        [...]
        """
        # Error handling
        if self.estimator is None:
            raise ValueError("_fit_estimator_() must be called before _process_estimator_()")

        # Extracting the key results (that show up in print_results)
        point = self.estimator.theta
        stderr = np.diag(self.estimator.variance)**0.5
        zscore = self.estimator.z_scores(null=self.null)
        ci = self.estimator.confidence_intervals(alpha=self.alpha)
        lower_ci = ci[:, 0]
        upper_ci = ci[:, 1]
        pval = self.estimator.p_values(null=self.null)
        sval = self.estimator.s_values(null=self.null)

        results = np.asarray([point, stderr, zscore, lower_ci, upper_ci, pval, sval]).T
        labels = ["Theta", "StdErr", "Z-score", "LCL", "UCL", "P-value", "S-value"]

        return results, labels
