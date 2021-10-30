import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic

from deli import MEstimator
from deli.estimating_equations import (ee_mean, ee_mean_variance, ee_mean_robust,
                                       ee_linear_regression, ee_logistic_regression,
                                       )
from deli.utilities import inverse_logit


class TestMEstimation:

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


class TestEstimatingEquations:

    def test_mean(self):
        """Tests mean with the built-in estimating equation.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi1(theta):
            return y - theta

        mcee = MEstimator(psi1, init=[0, ])
        mcee.estimate()

        def psi2(theta):
            return ee_mean(theta, y=y)

        mpee = MEstimator(psi2, init=[0, ])
        mpee.estimate()

        # Checking mean estimate
        npt.assert_allclose(mcee.theta,
                            mpee.theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mcee.asymptotic_variance,
                            mpee.asymptotic_variance,
                            atol=1e-6)

    def test_mean_variance(self):
        """Tests mean-variance with the built-in estimating equations.
        """
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi1(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        mcee = MEstimator(psi1, init=[0, 0, ])
        mcee.estimate()

        def psi2(theta):
            return ee_mean_variance(theta=theta, y=y)

        mpee = MEstimator(psi2, init=[0, 0, ])
        mpee.estimate()

        # Checking mean estimate
        npt.assert_allclose(mcee.theta,
                            mpee.theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mcee.asymptotic_variance,
                            mpee.asymptotic_variance,
                            atol=1e-6)

    def test_ols(self):
        """Tests linear regression with the built-in estimating equation.
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        def psi_regression(theta):
            x = np.asarray(data[['C', 'X', 'Z']])
            y = np.asarray(data['Y'])[:, None]
            beta = np.asarray(theta)[:, None]
            return ((y - np.dot(x, beta)) * x).T

        mcee = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        mcee.estimate()

        def psi_builtin_regression(theta):
            return ee_linear_regression(theta,
                                        X=data[['C', 'X', 'Z']],
                                        y=data['Y'])

        mpee = MEstimator(psi_builtin_regression, init=[0.1, 0.1, 0.1])
        mpee.estimate()

        # Checking mean estimate
        npt.assert_allclose(mcee.theta,
                            mpee.theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mcee.variance,
                            mpee.variance,
                            atol=1e-6)

    def test_logitic(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1

        def psi_regression(theta):
            x = np.asarray(data[['C', 'X', 'Z']])
            y = np.asarray(data['Y'])[:, None]
            beta = np.asarray(theta)[:, None]
            return ((y - inverse_logit(np.dot(x, beta))) * x).T

        mcee = MEstimator(psi_regression, init=[0., 0., 0.])
        mcee.estimate()

        def psi_builtin_regression(theta):
            return ee_logistic_regression(theta,
                                          X=data[['C', 'X', 'Z']],
                                          y=data['Y'])

        mpee = MEstimator(psi_builtin_regression, init=[0., 0., 0.])
        mpee.estimate()

        # Checking mean estimate
        npt.assert_allclose(mcee.theta,
                            mpee.theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mcee.variance,
                            mpee.variance,
                            atol=1e-6)
