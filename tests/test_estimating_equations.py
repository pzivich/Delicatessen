import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic

from delicatessen import MEstimator
from delicatessen.estimating_equations import (ee_mean, ee_mean_variance, ee_mean_robust,
                                               ee_linear_regression, ee_logistic_regression,
                                               ee_gformula, ee_ipw, ee_aipw)
from delicatessen.utilities import inverse_logit


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

    def test_mean_robust(self):
        y = [-10, -1, 2, 3, -2, 0, 3, 5, 12]
        yk = [-6, -1, 2, 3, -2, 0, 3, 5, 6]

        def psi(theta):
            return ee_mean_robust(theta=theta, y=y, k=6)

        mestimator = MEstimator(psi, init=[0, ])
        mestimator.estimate()

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[0],
                            np.mean(yk),
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

        def psi_builtin_regression(theta):
            return ee_linear_regression(theta,
                                        X=data[['C', 'X', 'Z']],
                                        y=data['Y'])

        mpee = MEstimator(psi_builtin_regression, init=[0.1, 0.1, 0.1])
        mpee.estimate()

        # Statsmodels function equivalent
        glm = smf.glm("Y ~ X + Z", data).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mpee.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mpee.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_wls(self):
        """Tests weighted linear regression by-hand with a single estimating equation.
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2 * data['X'] - 1 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi_regression(theta):
            return ee_linear_regression(theta,
                                        X=data[['C', 'X', 'Z']],
                                        y=data['Y'],
                                        weights=data['w'])

        mestimator = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, freq_weights=data['w']).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(glm.params),
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

    def test_weighted_logistic(self):
        """Tests weighted logistic regression by-hand with a single estimating equation.
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi_regression(theta):
            return ee_logistic_regression(theta,
                                          X=data[['C', 'X', 'Z']],
                                          y=data['Y'],
                                          weights=data['w'])

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, freq_weights=data['w'], family=sm.families.Binomial()).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

    @pytest.fixture
    def causal_data(self):
        np.random.seed(1205811)
        n = 1000
        df = pd.DataFrame()
        # Covariates
        df['W'] = np.random.binomial(1, p=0.5, size=n)
        df['A'] = np.random.binomial(1, p=(0.25 + 0.5 * df['W']), size=n)
        df['C'] = 1

        # Potential outcomes
        df['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5 * df['W']), size=n)
        df['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5 * df['W'] - 0.1 * 1), size=n)

        # Applying causal consistency
        df['Y'] = (1 - df['A']) * df['Ya0'] + df['A'] * df['Ya1']
        return df

    def test_gformula(self, causal_data):
        # M-estimation
        def psi(theta):
            return ee_gformula(theta,
                               X=causal_data[['C', 'A', 'W']],
                               y=causal_data['Y'],
                               treat_index=1)

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand g-formula with statsmodels
        glm = sm.GLM(causal_data['Y'], causal_data[['C', 'A', 'W']],
                     family=sm.families.Binomial()).fit()
        cd = causal_data[['C', 'A', 'W']].copy()
        cd['A'] = 1
        ya1 = glm.predict(cd)
        cd['A'] = 0
        ya0 = glm.predict(cd)

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestimator.theta[3:],
                            np.asarray(glm.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestimator.theta[0],
                            np.mean(ya1) - np.mean(ya0),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[1],
                            np.mean(ya1),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[2],
                            np.mean(ya0),
                            atol=1e-6)

    def test_ipw(self, causal_data):
        # M-estimation
        def psi(theta):
            return ee_ipw(theta,
                          X=causal_data[['C', 'A', 'W']],
                          y=causal_data['Y'],
                          treat_index=1)

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(causal_data['A'], causal_data[['C', 'W']],
                     family=sm.families.Binomial()).fit()
        pi = glm.predict()
        ya1 = causal_data['A'] * causal_data['Y'] / pi
        ya0 = (1-causal_data['A']) * causal_data['Y'] / (1-pi)

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestimator.theta[3:],
                            np.asarray(glm.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestimator.theta[0],
                            np.mean(ya1) - np.mean(ya0),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[1],
                            np.mean(ya1),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[2],
                            np.mean(ya0),
                            atol=1e-6)

    def test_aipw(self, causal_data):
        # M-estimation
        def psi_builtin_regression(theta):
            return ee_aipw(theta,
                           X=causal_data[['C', 'A', 'W']],
                           y=causal_data['Y'],
                           treat_index=1)

        mestimator = MEstimator(psi_builtin_regression, init=[0., 0.5, 0.5,   # Parameters of interest
                                                              0., 0., 0.,     # Outcome nuisance model
                                                              0., 0.])        # Treatment nuisance model
        mestimator.estimate(solver='lm', tolerance=1e-12)

        # By-hand IPW estimator with statsmodels
        pi_m = sm.GLM(causal_data['A'], causal_data[['C', 'W']],
                      family=sm.families.Binomial()).fit()
        y_m = sm.GLM(causal_data['Y'], causal_data[['C', 'A', 'W']],
                     family=sm.families.Binomial()).fit()
        # Predicting coefficients
        pi = pi_m.predict()
        cd = causal_data[['C', 'A', 'W']].copy()
        cd['A'] = 1
        ya1 = y_m.predict(cd)
        cd['A'] = 0
        ya0 = y_m.predict(cd)
        # AIPW estimator
        ya1_star = causal_data['Y'] * causal_data['A'] / pi - ya1 * (causal_data['A'] - pi) / pi
        ya0_star = causal_data['Y'] * (1-causal_data['A']) / (1-pi) - ya0 * (pi - causal_data['A']) / (1-pi)
        # AIPW variance estimator!
        var_ate = np.nanvar((ya1_star - ya0_star) - np.mean(ya1_star - ya0_star), ddof=1) / causal_data.shape[0]
        var_r1 = np.nanvar(ya1_star - np.mean(ya1_star), ddof=1) / causal_data.shape[0]
        var_r0 = np.nanvar(ya0_star - np.mean(ya0_star), ddof=1) / causal_data.shape[0]

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestimator.theta[3:6],
                            np.asarray(y_m.params),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[6:],
                            np.asarray(pi_m.params),
                            atol=1e-6)

        # Checking mean estimates
        npt.assert_allclose(mestimator.theta[0],
                            np.mean(ya1_star) - np.mean(ya0_star),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[1],
                            np.mean(ya1_star),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[2],
                            np.mean(ya0_star),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestimator.variance[0, 0],
                            var_ate,
                            atol=1e-6)
        npt.assert_allclose(mestimator.variance[1, 1],
                            var_r1,
                            atol=1e-6)
        npt.assert_allclose(mestimator.variance[2, 2],
                            var_r0,
                            atol=1e-6)

