####################################################################################################################
# Tests for GMM-estimator features
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic
from scipy.optimize import minimize

from delicatessen import GMMEstimator
from delicatessen.utilities import inverse_logit
from delicatessen.estimating_equations import ee_regression, ee_2sls

np.random.seed(236461)


class TestGMMEstimation:

    @pytest.fixture
    def data_c(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5, 3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5]
        d['I'] = 1
        return d

    def test_error_none_returned(self):
        """Checks for an error when the estimating equations return None
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, np.nan])

        def psi(theta):
            y - theta  # Nothing returned here

        estr = GMMEstimator(psi, init=[0, ])
        with pytest.raises(ValueError, match="returns an array evaluated"):
            estr.estimate()

    def test_error_nan(self):
        """Checks for an error when estimating equations return a NaN at the init values
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, np.nan])

        def psi(theta):
            return y - theta

        estr = GMMEstimator(psi, init=[0, ])
        with pytest.raises(ValueError, match="at least one np.nan"):
            estr.estimate()

    def test_error_minimizer1(self):
        """Checks for an error when an invalid minimizer is provided
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        estr = GMMEstimator(psi, init=[0, ])
        with pytest.raises(ValueError, match="The solver 'not-avail'"):
            estr.estimate(solver='not-avail')

    def test_error_minimizer2(self):
        """Check that user-specified solver has correct arguments
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        def custom_solver(stacked_equations):
            opt = minimize(stacked_equations, x0=np.asarray([0, ]),
                           method='bfgs')
            return opt.x

        estr = GMMEstimator(psi, init=[0, ])
        with pytest.raises(TypeError, match="The user-specified minimizer `solver` must be a function"):
            estr.estimate(solver=custom_solver)

    def test_error_rootfinder3(self):
        """Check that user-specified solver returns something besides None
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        def custom_solver(stacked_equations, init):
            opt = minimize(stacked_equations, x0=np.asarray(init),
                           method='bfgs')

        mestimator = GMMEstimator(psi, init=[0, ])
        with pytest.raises(ValueError, match="must return the solution to the"):
            mestimator.estimate(solver=custom_solver)

    def test_error_bad_inits1(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta[0]

        estr = GMMEstimator(psi, init=[0, 0])
        with pytest.raises(ValueError, match="should be less than or equal to"):
            estr.estimate()

    def test_error_bad_inits2(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return np.asarray((y - theta[0],
                               y ** 2 - theta[1]))

        estr = GMMEstimator(psi, init=[0, 0, 0])
        with pytest.raises(ValueError, match="should be less than or equal to"):
            estr.estimate()

    def test_error_bad_inits3(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return np.asarray((y - theta[0],
                               y ** 2 - theta[1]))

        estr = GMMEstimator(psi, init=[0, ])
        with pytest.raises(IndexError):
            estr.estimate()

    def test_error_dimensions(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return np.zeros((2, 3, 4))

        estr = GMMEstimator(psi, init=[0, 0, 0])
        with pytest.raises(ValueError, match="A 2-dimensional array is expected"):
            estr.estimate()

    def test_mean_variance_1eq(self):
        """Tests the mean / variance with a single estimating equation.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        estr = GMMEstimator(psi, init=[0, ])
        estr.estimate()

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.asymptotic_variance,
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_mean_variance_1eq_lm_solver(self):
        """Tests the mean / variance with a single estimating equation.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        estr = GMMEstimator(psi, init=[0, ])
        estr.estimate()

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.asymptotic_variance,
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_mean_variance_2eq(self):
        """Tests the mean / variance with two estimating equations.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        estr = GMMEstimator(psi, init=[0, 0])
        estr.estimate()

        # Checking mean estimate
        npt.assert_allclose(estr.theta[0],
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.theta[1],
                            estr.asymptotic_variance[0][0],
                            atol=1e-6)
        npt.assert_allclose(estr.theta[1],
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_mean_variance_2eq_lm_solver(self):
        """Tests the mean / variance with two estimating equations.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        estr = GMMEstimator(psi, init=[0, 0])
        estr.estimate()

        # Checking mean estimate
        npt.assert_allclose(estr.theta[0],
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.theta[1],
                            estr.asymptotic_variance[0][0],
                            atol=1e-6)
        npt.assert_allclose(estr.theta[1],
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_ratio_estimator(self):
        """Tests the ratio with a single estimating equation.
        """
        # Data sets
        data = pd.DataFrame()
        data['Y'] = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])
        data['X'] = np.array([1, 5, 3, 5, 1, 4, 1, 2, 5, 1, 2, 12, 1, 8])

        def psi(theta):
            return data['Y'] - data['X'] * theta

        estr = GMMEstimator(psi, init=[0, ])
        estr.estimate()

        # Closed form solutions from SB
        theta = np.mean(data['Y']) / np.mean(data['X'])
        var = (1 / np.mean(data['X']) ** 2) * np.mean((data['Y'] - theta * data['X']) ** 2)

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.asymptotic_variance,
                            var,
                            atol=1e-6)

    def test_alt_ratio_estimator(self):
        """Tests the alternative ratio with three estimating equations.
        """
        # Data sets
        data = pd.DataFrame()
        data['Y'] = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])
        data['X'] = np.array([1, 5, 3, 5, 1, 4, 1, 2, 5, 1, 2, 12, 1, 8])
        data['C'] = 1

        def psi(theta):
            return (data['Y'] - theta[0],
                    data['X'] - theta[1],
                    data['C'] * theta[0] - theta[1] * theta[2])

        estr = GMMEstimator(psi, init=[0, 0, 0])
        estr.estimate()

        # Closed form solutions from SB
        theta = np.mean(data['Y']) / np.mean(data['X'])
        var = (1 / np.mean(data['X']) ** 2) * np.mean((data['Y'] - theta * data['X']) ** 2)

        # Checking mean estimate
        npt.assert_allclose(estr.theta[-1],
                            theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.asymptotic_variance[-1][-1],
                            var,
                            atol=1e-5)

    def test_alt_ratio_estimator_lm_solver(self):
        """Tests the alternative ratio with three estimating equations.
        """
        # Data sets
        data = pd.DataFrame()
        data['Y'] = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])
        data['X'] = np.array([1, 5, 3, 5, 1, 4, 1, 2, 5, 1, 2, 12, 1, 8])
        data['C'] = 1

        def psi(theta):
            return (data['Y'] - theta[0],
                    data['X'] - theta[1],
                    data['C'] * theta[0] - theta[1] * theta[2])

        estr = GMMEstimator(psi, init=[0, 0, 0])
        estr.estimate()

        # Closed form solutions from SB
        theta = np.mean(data['Y']) / np.mean(data['X'])
        var = (1 / np.mean(data['X']) ** 2) * np.mean((data['Y'] - theta * data['X']) ** 2)

        # Checking mean estimate
        npt.assert_allclose(estr.theta[-1],
                            theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.asymptotic_variance[-1][-1],
                            var,
                            atol=1e-5)

    def test_ols(self):
        """Tests linear regression by-hand with a single estimating equation.
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2 * data['X'] - 1 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        def psi_regression(theta):
            x = np.asarray(data[['C', 'X', 'Z']])
            y = np.asarray(data['Y'])[:, None]
            beta = np.asarray(theta)[:, None]
            return ((y - np.dot(x, beta)) * x).T

        estr = GMMEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        estr.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(estr.confidence_intervals(),
                            np.asarray(glm.conf_int()),
                            atol=1e-6)

        # Checking confidence bands
        npt.assert_allclose(estr.confidence_bands(method='bonferroni'),
                            estr.confidence_intervals(alpha=0.05 / len(estr.theta)),
                            atol=1e-6)

        # Checking Z-scores
        npt.assert_allclose(estr.z_scores(null=0),
                            np.asarray(glm.tvalues),
                            atol=1e-6)

        # Checking P-values
        npt.assert_allclose(estr.p_values(null=0),
                            np.asarray(glm.pvalues),
                            atol=1e-6)

        # Checking S-values
        npt.assert_allclose(estr.s_values(null=0),
                            -1 * np.log2(np.asarray(glm.pvalues)),
                            atol=1e-4)

        # Checking influence function
        ifunc = estr.influence_functions()
        npt.assert_allclose(np.dot(ifunc.T, ifunc) / (data.shape[0]**2),
                            estr.variance,
                            atol=1e-6)

    def test_logistic(self):
        """Tests linear regression by-hand with a single estimating equation.
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2 * data['X'] - 1 * data['Z']), size=n)
        data['C'] = 1

        def psi_regression(theta):
            x = np.asarray(data[['C', 'X', 'Z']])
            y = np.asarray(data['Y'])[:, None]
            beta = np.asarray(theta)[:, None]
            return ((y - inverse_logit(np.dot(x, beta))) * x).T

        estr = GMMEstimator(psi_regression, init=[0., 0., 0.])
        estr.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, family=sm.families.Binomial()).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(estr.confidence_intervals(),
                            np.asarray(glm.conf_int()),
                            atol=1e-6)

        # Checking Z-scores
        npt.assert_allclose(estr.z_scores(null=0),
                            np.asarray(glm.tvalues),
                            atol=1e-6)

        # Checking P-values
        npt.assert_allclose(estr.p_values(null=0),
                            np.asarray(glm.pvalues),
                            atol=1e-6)

        # Checking S-values
        npt.assert_allclose(estr.s_values(null=0),
                            -1 * np.log2(np.asarray(glm.pvalues)),
                            atol=1e-4)

        # Checking influence function
        ifunc = estr.influence_functions()
        npt.assert_allclose(np.dot(ifunc.T, ifunc) / (data.shape[0]**2),
                            estr.variance,
                            atol=1e-6)

    def test_custom_solver(self):
        """Test the use of a user-specified minimization algorithm.
        """
        # Generating some generic data for the mean
        y = np.random.normal(size=1000)

        # This is the stacked estimating equations that we are solving!!
        def psi(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        # This is the custom estimating equation
        def custom_solver(stacked_equations, init):
            options = {"maxiter": 1000}
            opt = minimize(stacked_equations,
                           x0=np.asarray(init),
                           method='cg', options=options, tol=1e-9)
            return opt.x

        # Estimating the M-Estimator
        estr = GMMEstimator(psi, init=[0, 0])
        estr.estimate(solver=custom_solver)

        # Checking mean estimate
        npt.assert_allclose(estr.theta[0],
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.theta[1],
                            estr.asymptotic_variance[0][0],
                            atol=1e-6)
        npt.assert_allclose(estr.theta[1],
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_no_pderiv_overwrite(self):
        """Test for error found in v0.1b2 (when the `dx` argument was added).
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        estr1 = GMMEstimator(psi, init=[0, ])
        estr1.estimate(dx=1e-9)

        estr2 = GMMEstimator(psi, init=[0, ])
        estr2.estimate(dx=10)

        npt.assert_allclose(estr1.theta, estr2.theta)

    def test_subset_params(self, data_c):
        # Creating data set
        data_c['A1'] = 1

        # Setting up data from M-estimation
        x = np.asarray(data_c[['I', 'A', 'W', 'V']])
        x1 = np.asarray(data_c[['I', 'A1', 'W', 'V']])
        y = np.asarray(data_c['Y'])

        # IPW mean estimating equation
        def psi(theta):
            ee_reg = ee_regression(theta=theta[1:],
                                   X=x, y=y,
                                   model='linear')
            ee_mean = np.dot(x1, theta[1:]) - theta[0]
            return np.vstack((ee_mean,
                              ee_reg))

        # Full solve
        init = [0, ] + [0, ] * x.shape[1]
        ns = GMMEstimator(psi, init=init)
        ns.estimate(deriv_method='exact')

        # Subset solve (using previous regression solutions)
        init = [0, ] + list(ns.theta[1:])
        ys = GMMEstimator(psi, init=init, subset=[0, ])
        ys.estimate(deriv_method='exact')

        # Check point estimates are all close
        npt.assert_allclose(ns.theta, ys.theta)

        # Check variance estimates are all close
        npt.assert_allclose(ns.bread, ys.bread, atol=1e-9)
        npt.assert_allclose(ns.meat, ys.meat, atol=1e-9)
        npt.assert_allclose(ns.variance, ys.variance, atol=1e-9)

    def test_subset_params2(self, data_c):
        # Creating data set
        data_c['A1'] = 1

        # Setting up data from M-estimation
        x = np.asarray(data_c[['I', 'A', 'W', 'V']])
        x1 = np.asarray(data_c[['I', 'A1', 'W', 'V']])
        y = np.asarray(data_c['Y'])

        # IPW mean estimating equation
        def psi(theta):
            ee_reg = ee_regression(theta=theta[1:],
                                   X=x, y=y,
                                   model='linear')
            ee_mean = np.dot(x1, theta[1:]) - theta[0]
            return np.vstack((ee_mean,
                              ee_reg))

        # Full solve
        init = [0, ] + [0, ] * x.shape[1]
        ns = GMMEstimator(psi, init=init)
        ns.estimate(deriv_method='exact')

        # Subset solve (using previous regression solutions)
        init = [0, 0, ] + list(ns.theta[2:4]) + [0, ]
        ys = GMMEstimator(psi, init=init, subset=[0, 1, 4])
        ys.estimate(deriv_method='exact')

        # Check point estimates are all close
        npt.assert_allclose(ns.theta, ys.theta)

        # Check variance estimates are all close
        npt.assert_allclose(ns.bread, ys.bread, atol=1e-9)
        npt.assert_allclose(ns.meat, ys.meat, atol=1e-9)
        npt.assert_allclose(ns.variance, ys.variance, atol=1e-9)


# TODO need to do over-identification
class TestGMMEstimationOverID:

    # Data was generated according to the following:
    # np.random.seed(9242)
    # n = 90
    # d0 = pd.DataFrame()
    # d0['W'] = np.random.binomial(n=1, p=0.25, size=n)
    # d0['Z1'] = np.random.normal(scale=0.5, size=n)
    # d0['Z2'] = np.random.normal(scale=0.5, size=n)
    # d0['A'] = d0['Z1'] + d0['Z2'] + np.random.normal(size=n)
    # d0['Y'] = 2*d0['A'] - 1*d0['W']*d0['A'] + np.random.normal(scale=1.0, size=n)
    # d0['S'] = 0
    # d0 = np.round(d0, 2)
    #
    # d1 = pd.DataFrame()
    # d1['W'] = np.random.binomial(n=1, p=0.75, size=n)
    # d1['S'] = 1
    # d1 = np.round(d1, 2)

    @pytest.fixture
    def data_iv(self):
        d = pd.DataFrame()
        d['W'] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
                  0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Z1'] = [0.36, 0.82, 1.04, -0.12, -0.27, -0.24, 0.41, 0.97, 0.38, -0.16, 1.1, -0.22, 0.44, -0.25, -0.67,
                   -0.27, -0.58, 0.97, -0.38, -0.14, -0.25, -0.12, 0.2, 0.4, 0.16, -0.24, 0.01, -0.03, -0.23, -0.06,
                   0.21, 0.56, 0.06, 0.47, -0.44, 0.3, -0.38, 0.35, -0.11, 0.1, 0.34, -1.14, -0.82, 0.37, -0.52, 0.33,
                   -0.19, -0.33, 0.01, -0.58, -0.31, 0.18, 0.6, 0.49, 0.08, -0.1, 0.82, 0.01, -0.84, -0.28, -0.28,
                   0.26, 0.54, 0.44, -0.17, -0.5, 0.42, 0.08, -0.27, -0.21, -0.17, -0.58, 0.68, 0.42, 0.45, -0.35,
                   -0.39, 0.22, 0.69, 0.44, 0.64, -0.18, 0.04, 0.57, 0.08, 0.2, -0.19, 0.09, -0.05, 0.13]
        d['Z2'] = [0., -0.21, -0.51, 0.54, -0.43, -1.13, 0.19, -0.51, -0.38, 0.08, 0.62, 0.25, 0.05, 0.08, 0.77, -0.35,
                   -0.17, 0.04, -0.04, 0.44, 1.01, 0.36, -0.19, 0.47, 0.45, 0.14, 0.16, 0.08, 0.15, -0.37, -0.07, 0.01,
                   -0.57, 0.43, 0.12, -0.38, 0.41, -0.61, 0.06, 0.05, -0.77, -0.37, -0.01, -0.32, -0.64, -0.64, 0.26,
                   0.3, -0.08, 0.25, -0.05, -0.23, -0.12, 0.25, 0.01, -0.04, 0.53, 0.28, 0.89, 0.51, -0.66, -0.65, 0.54,
                   0.47, -0.68, -0.01, 0.56, -0.05, 0.72, -0.77, -0.2, -0.64, -0.19, -0.28, 0.72, -0.17, 0.35, -0.16,
                   0., -0.2, -0.61, -0.9, -0.89, -0.07, 0.64, 0.42, 0.32, 0.53, -1.44, 0.46]
        d['A'] = [0.69, -0.23, -0.29, 0.16, -0.16, -1.84, 0.84, -0.9, 2.54, 0.57, 2.18, -0.3, 0.75, -0.83, -1.44, -1.68,
                  -1.95, 0.26, -0.88, -1.19, -1.73, -1.11, -1.07, 0.56, 0.8, 1.48, -1.37, -1.95, 0.4, 2.39, -0.5, 1.58,
                  -0.58, -1.08, 0.51, -0.9, 0.59, -0.28, 0.72, -1.42, -1.09, -0.93, -1.69, -0.87, -0.78, 0.13, 0.46,
                  -0.48, -0.34, 1.82, -0.6, -0.06, -0.67, 0.95, -1.35, -0.3, 3.22, 1.63, 1.31, -2.1, -0.72, 0.16, 0.55,
                  1.13, -0.8, -1.36, 1.05, 0.87, -0.98, -1.46, 0.5, -0.05, 0.23, -0.01, -0.25, -1.42, 0.53, 0.37, -0.61,
                  1.29, 0.89, -2.17, -0.83, 0.48, 0.37, 1.28, 1.26, 0.7, -2.11, 1.33]
        d['Y'] = [1.73, -0.83, 0.98, -0.06, 0.44, -4.64, 2.12, -3.23, 5.73, -0.36, 4.84, -2.34, 1.95, -0.27, -1.81,
                  -2.36, -3.17, 0.28, -2.48, -2.54, -3.19, -0.31, -2.33, 0.96, 1.4, 1.48, -1.93, -4.07, 0.55, 1.99,
                  -0.5, 3.04, -0.64, -1.21, 0.94, -0.43, 1.52, 0.1, 0.72, 0.15, -0.69, -1.74, -2.84, 0.53, -1.28, -0.9,
                  1.52, -0.84, -0.05, 3.6, 0.86, 0.03, 1.21, 1.1, -1.81, -1.28, 5.88, 2.57, 3.55, -1.82, -0.99, -0.86,
                  0.48, 3.01, -1.84, -3.36, 1.05, 1.43, -0.98, -1.17, 0.53, -0.75, -0.84, -1.4, -1.69, -1.11, -1.55,
                  1.79, -0.86, 0.78, 1.46, -4.09, -0.79, 2.09, 0.67, 3.38, 3.02, 2.94, -3.89, 1.4]
        d['S'] = 0
        d['C'] = 1
        return d

    @pytest.fixture
    def data_target(self):
        d = pd.DataFrame()
        d['W'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                  1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        d['Z1'] = -9999
        d['Z2'] = -9999
        d['A'] = -9999
        d['Y'] = -9999
        d['S'] = 1
        d['C'] = 1
        return d

    @pytest.fixture
    def data_transport(self, data_iv, data_target):
        d0 = data_iv
        d1 = data_target
        d = pd.concat([d1, d0], ignore_index=True)
        return d

    def test_error_underid1(self, data_iv):
        d = data_iv
        z1 = d['Z1']
        a = d['A']
        y = d['Y']

        def psi(theta):
            return z1*(y - theta[0]*a)

        estr = GMMEstimator(psi, init=[0, 0])
        with pytest.raises(ValueError, match="should be less than or equal to"):
            estr.estimate()

    def test_error_underid2(self, data_iv):
        d = data_iv
        z1 = d['Z1']
        z2 = d['Z2']
        a = d['A']
        y = d['Y']

        def psi(theta):
            ee_z1 = z1*(y - theta[0]*a)
            ee_z2 = z2*(y - theta[0]*a)
            return np.vstack([ee_z1, ee_z2])

        estr = GMMEstimator(psi, init=[0, 0, 0])
        with pytest.raises(ValueError, match="3 initial values and the `stacked_equations` function returns 2"):
            estr.estimate()

    def test_warn_overid_maxiter(self, data_iv):
        d = data_iv
        z1 = d['Z1']
        z2 = d['Z2']
        a = d['A']
        y = d['Y']

        def psi(theta):
            ee_z1 = z1*(y - theta[0]*a)
            ee_z2 = z2*(y - theta[0]*a)
            return np.vstack([ee_z1, ee_z2])

        estr = GMMEstimator(psi, init=[0, ], overid_maxiter=1)
        with pytest.warns(UserWarning, match="exceeded for the iterative GMM updating"):
            estr.estimate()

    def test_overid_iv(self, data_iv):
        d = data_iv
        z1 = d['Z1']
        z2 = d['Z2']
        a = d['A']
        y = d['Y']

        def psi_overid(theta):
            ee_z1 = z1*(y - theta[0]*a)
            ee_z2 = z2*(y - theta[0]*a)
            return np.vstack([ee_z1, ee_z2])

        estr = GMMEstimator(psi_overid, init=[0, ])
        estr.estimate()

        # Checking point estimate
        npt.assert_allclose(estr.theta, [1.934432, ], atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, [[0.02519052, ], ], atol=1e-5)

    def test_overid_iv_target(self, data_transport):
        d = data_transport
        W = np.asarray(d[['C', 'W']])
        s = np.asarray(d['S'])
        z1 = d['Z1']
        z2 = d['Z2']
        a = d['A']
        y = d['Y']

        def psi_overid(theta):
            alpha = theta[0]
            beta = theta[1:]

            # Calculating inverse odds of sampling weights
            pi_s = inverse_logit(np.dot(W, beta))
            iosw = (1 - s) * pi_s / (1 - pi_s)

            # Sampling model
            ee_sample = ee_regression(theta=beta, y=s, X=W, model='logistic')
            ee_z1 = z1 * (y - alpha * a) * iosw * (1 - s)
            ee_z2 = z2 * (y - alpha * a) * iosw * (1 - s)
            return np.vstack([ee_z1, ee_z2, ee_sample])

        estr = GMMEstimator(psi_overid, init=[0, 0, 0])
        estr.estimate()

        # Checking point estimate
        npt.assert_allclose(estr.theta, [1.831228, -1.764904, 3.124276], atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, [[0.160789, -0.005212, 0.009804],
                                            [-0.005212,  0.100592, -0.102601],
                                            [0.009804, -0.102601,  0.168713], ],
                            atol=1e-5)

    def test_overid_iv_subset(self, data_transport):
        d = data_transport
        W = np.asarray(d[['C', 'W']])
        s = np.asarray(d['S'])
        z1 = d['Z1']
        z2 = d['Z2']
        a = d['A']
        y = d['Y']

        def psi_overid(theta):
            alpha = theta[0]
            beta = theta[1:]

            # Calculating inverse odds of sampling weights
            pi_s = inverse_logit(np.dot(W, beta))
            iosw = (1 - s) * pi_s / (1 - pi_s)

            # Sampling model
            ee_sample = ee_regression(theta=beta, y=s, X=W, model='logistic')
            ee_z1 = z1 * (y - alpha * a) * iosw * (1 - s)
            ee_z2 = z2 * (y - alpha * a) * iosw * (1 - s)
            return np.vstack([ee_z1, ee_z2, ee_sample])

        estr = GMMEstimator(psi_overid, init=[0, -1.08227033, 2.18464809], subset=[0, ])
        estr.estimate()

        # Checking point estimate
        npt.assert_allclose(estr.theta, [1.870714, -1.08227033, 2.18464809], atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, [[0.109367, -0.00225699, 0.00485698],
                                            [-0.00225699, 0.04731588, -0.04802412],
                                            [0.00485698, -0.04802412, 0.09727285], ],
                            atol=1e-5)
