import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic
from scipy.optimize import root

from delicatessen import MEstimator
from delicatessen.utilities import inverse_logit

np.random.seed(236461)


class TestMEstimation:

    def test_error_nan(self):
        """Checks for an error when estimating equations return a NaN at the init values
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, np.nan])

        def psi(theta):
            return y - theta

        mestimator = MEstimator(psi, init=[0, ])
        with pytest.raises(ValueError, match="at least one np.nan"):
            mestimator.estimate()

    def test_error_rootfinder1(self):
        """Checks for an error when an invalid root finder is provided
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        mestimator = MEstimator(psi, init=[0, ])
        with pytest.raises(ValueError, match="The solver 'not-avail'"):
            mestimator.estimate(solver='not-avail')

    def test_error_rootfinder2(self):
        """Check that user-specified solver has correct arguments
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        def custom_solver(stacked_equations):
            options = {"maxiter": 1000}
            opt = root(stacked_equations, x0=np.asarray([0, ]),
                       method='lm', tol=1e-9, options=options)
            return opt.x

        mestimator = MEstimator(psi, init=[0, ])
        with pytest.raises(TypeError, match="The user-specified root-finding `solver` must be a function"):
            mestimator.estimate(solver=custom_solver)

    def test_error_rootfinder3(self):
        """Check that user-specified solver returns something besides None
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        def custom_solver(stacked_equations, init):
            options = {"maxiter": 1000}
            opt = root(stacked_equations, x0=np.asarray(init),
                       method='lm', tol=1e-9, options=options)

        mestimator = MEstimator(psi, init=[0, ])
        with pytest.raises(ValueError, match="must return the solution to the"):
            mestimator.estimate(solver=custom_solver)

    def test_error_bad_inits1(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta[0]

        mestimator = MEstimator(psi, init=[0, 0])
        with pytest.raises(ValueError, match="initial values and the number of rows returned by `stacked_equations`"):
            mestimator.estimate()

    def test_error_bad_inits2(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return np.asarray((y - theta[0],
                               y**2 - theta[1]))

        mestimator = MEstimator(psi, init=[0, 0, 0])
        with pytest.raises(ValueError, match="initial values and the number of rows returned by `stacked_equations`"):
            mestimator.estimate()

    def test_error_bad_inits3(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return np.asarray((y - theta[0],
                               y**2 - theta[1]))

        mestimator = MEstimator(psi, init=[0, ])
        with pytest.raises(IndexError):
            mestimator.estimate()

    def test_error_dimensions(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return np.zeros((2, 3, 4))

        mestimator = MEstimator(psi, init=[0, 0, 0])
        with pytest.raises(ValueError, match="A 2-dimensional array is expected"):
            mestimator.estimate()

    def test_mean_variance_1eq(self):
        """Tests the mean / variance with a single estimating equation.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        mestimator = MEstimator(psi, init=[0, ])
        mestimator.estimate()

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.asymptotic_variance,
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_mean_variance_1eq_lm_solver(self):
        """Tests the mean / variance with a single estimating equation.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        mestimator = MEstimator(psi, init=[0, ])
        mestimator.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.asymptotic_variance,
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_mean_variance_2eq(self):
        """Tests the mean / variance with two estimating equations.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        mestimator = MEstimator(psi, init=[0, 0])
        mestimator.estimate()

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[0],
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.theta[1],
                            mestimator.asymptotic_variance[0][0],
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[1],
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_mean_variance_2eq_lm_solver(self):
        """Tests the mean / variance with two estimating equations.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        mestimator = MEstimator(psi, init=[0, 0])
        mestimator.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[0],
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.theta[1],
                            mestimator.asymptotic_variance[0][0],
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[1],
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
            return data['Y'] - data['X']*theta

        mestimator = MEstimator(psi, init=[0, ])
        mestimator.estimate()

        # Closed form solutions from SB
        theta = np.mean(data['Y']) / np.mean(data['X'])
        var = (1 / np.mean(data['X']) ** 2) * np.mean((data['Y'] - theta * data['X']) ** 2)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.asymptotic_variance,
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

        mestimator = MEstimator(psi, init=[0, 0, 0])
        mestimator.estimate()

        # Closed form solutions from SB
        theta = np.mean(data['Y']) / np.mean(data['X'])
        var = (1 / np.mean(data['X']) ** 2) * np.mean((data['Y'] - theta * data['X']) ** 2)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[-1],
                            theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.asymptotic_variance[-1][-1],
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

        mestimator = MEstimator(psi, init=[0, 0, 0])
        mestimator.estimate(solver='lm')

        # Closed form solutions from SB
        theta = np.mean(data['Y']) / np.mean(data['X'])
        var = (1 / np.mean(data['X']) ** 2) * np.mean((data['Y'] - theta * data['X']) ** 2)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[-1],
                            theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.asymptotic_variance[-1][-1],
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

        mestimator = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(mestimator.confidence_intervals(),
                            np.asarray(glm.conf_int()),
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

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, family=sm.families.Binomial()).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(mestimator.confidence_intervals(),
                            np.asarray(glm.conf_int()),
                            atol=1e-6)

    def test_custom_solver(self):
        """Test the use of a user-specified root-finding algorithm.
        """
        # Generating some generic data for the mean
        y = np.random.normal(size=1000)

        # This is the stacked estimating equations that we are solving!!
        def psi(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        # This is the custom estimating equation
        def custom_solver(stacked_equations, init):
            options = {"maxiter": 1000}
            opt = root(stacked_equations,
                       x0=np.asarray(init),
                       method='lm',
                       tol=1e-9,
                       options=options)
            return opt.x

        # Estimating the M-Estimator
        mestimator = MEstimator(psi, init=[0, 0])
        mestimator.estimate(solver=custom_solver)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[0],
                            np.mean(y),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.theta[1],
                            mestimator.asymptotic_variance[0][0],
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[1],
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_no_pderiv_overwrite(self):
        """Test for error found in v0.1b2 (when the `dx` argument was added).
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        estr1 = MEstimator(psi, init=[0, ])
        estr1.estimate(dx=1e-9)

        estr2 = MEstimator(psi, init=[0, ])
        estr2.estimate(dx=10)

        npt.assert_allclose(estr1.theta, estr2.theta)



