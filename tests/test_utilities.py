####################################################################################################################
# Tests for provided user-accessible utility functions
####################################################################################################################

import pytest
import numpy as np
import scipy as sp
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic
from lifelines import ExponentialFitter, WeibullFitter, WeibullAFTFitter

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_mean, ee_regression, ee_survival_model, ee_aft
from delicatessen.utilities import (identity, inverse_logit, logit,
                                    polygamma, digamma,
                                    robust_loss_functions,
                                    aggregate_efuncs,
                                    spline,
                                    regression_predictions, survival_predictions,
                                    aft_predictions_individual, aft_predictions_function,
                                    additive_design_matrix)

np.random.seed(80958151)


@pytest.fixture
def design_matrix():
    array = np.array([[1, 1, 1, 1, 1],
                      [1, 5, 10, 15, 20],
                      [1, 2, 3, 4, 5], ])
    return array.T


class TestFunctions:

    def test_identity_var(self):
        """Checks the logit transformation of a single observation
        """
        npt.assert_allclose(identity(0.5), 0.5)
        npt.assert_allclose(identity(0.25), 0.25)
        npt.assert_allclose(identity(0.75), 0.75)

    def test_logit_var(self):
        """Checks the logit transformation of a single observation
        """
        npt.assert_allclose(logit(0.5), 0.)
        npt.assert_allclose(logit(0.25), -1.098612288668)
        npt.assert_allclose(logit(0.75), 1.098612288668)

    def test_invlogit_var(self):
        """Checks the inverse logit transformation of a single observation
        """
        npt.assert_allclose(inverse_logit(0.), 0.5)
        npt.assert_allclose(inverse_logit(-1.098612288668), 0.25)
        npt.assert_allclose(inverse_logit(1.098612288668), 0.75)

    def test_logit_backtransform(self):
        """Checks the interative applications of logit and expit transformations of a single observation
        """
        original = 0.6521
        original_logit = logit(original)

        # Transform 1st order
        npt.assert_allclose(inverse_logit(original_logit), original)
        npt.assert_allclose(logit(inverse_logit(original_logit)), original_logit)

        # Transform 2nd order
        npt.assert_allclose(inverse_logit(logit(inverse_logit(original_logit))), original)
        npt.assert_allclose(logit(inverse_logit(logit(inverse_logit(original_logit)))), original_logit)

        # Transform 3rd order
        npt.assert_allclose(inverse_logit(logit(inverse_logit(logit(inverse_logit(original_logit))))),
                            original)
        npt.assert_allclose(logit(inverse_logit(logit(inverse_logit(logit(inverse_logit(original_logit)))))),
                            original_logit)

    def test_identity_array(self):
        """Checks the identity transformation of an array
        """
        vals = np.array([0.5, 0.25, 0.75, 0.5])
        npt.assert_allclose(identity(vals), vals)

    def test_logit_array(self):
        """Checks the inverse logit transformation of an array
        """
        prbs = np.array([0.5, 0.25, 0.75, 0.5])
        odds = np.array([0., -1.098612288668, 1.098612288668, 0.])

        npt.assert_allclose(logit(prbs), odds)

    def test_expit_array(self):
        """Checks the inverse logit transformation of an array
        """
        prbs = np.array([0.5, 0.25, 0.75, 0.5])
        odds = np.array([0., -1.098612288668, 1.098612288668, 0.])

        npt.assert_allclose(inverse_logit(odds), prbs)

    def test_identity_list(self):
        """Checks the identity transformation of a list
        """
        vals = [0.5, 0.25, 0.75, 0.5]
        npt.assert_allclose(identity(vals), vals)

    def test_logit_list(self):
        """Checks the logit transformation of a list
        """
        prbs = [0.5, 0.25, 0.75, 0.5]
        odds = [0., -1.098612288668, 1.098612288668, 0.]

        npt.assert_allclose(logit(prbs), odds)

    def test_expit_list(self):
        """Checks the inverse logit transformation of a list
        """
        prbs = [0.5, 0.25, 0.75, 0.5]
        odds = [0., -1.098612288668, 1.098612288668, 0.]

        npt.assert_allclose(logit(prbs), odds)

    def test_polygamma(self):
        """Checks the polygamma wrapper function
        """
        y = np.array([-1, 0, 2, 3, 12, -58101, 5091244])

        # Single input check
        npt.assert_allclose(polygamma(n=1, x=y[0]),
                            sp.special.polygamma(n=1, x=y[0]))
        npt.assert_allclose(polygamma(n=3, x=y[-1]),
                            sp.special.polygamma(n=3, x=y[-1]))

        # Multiple input check
        npt.assert_allclose(polygamma(n=2, x=y),
                            sp.special.polygamma(n=2, x=y))
        npt.assert_allclose(polygamma(n=4, x=y),
                            sp.special.polygamma(n=4, x=y))

        # Multiple input check into 2D array
        y = y[:, None]
        npt.assert_allclose(polygamma(n=1, x=y),
                            sp.special.polygamma(n=1, x=y))
        npt.assert_allclose(polygamma(n=4, x=y),
                            sp.special.polygamma(n=4, x=y))

    def test_digamma(self):
        """Checks the digamma wrapper function
        """
        y = np.array([-1, 0, 2, 3, 12, -58101, 5091244])

        # Single input check
        v2 = sp.special.digamma(y[0])
        npt.assert_allclose(digamma(y[0]), v2)
        v2 = sp.special.digamma(y[-1])
        npt.assert_allclose(digamma(y[-1]), v2)

        # Multiple input check
        v2 = sp.special.digamma(y)
        npt.assert_allclose(digamma(z=y), v2)

    def test_inverse_logit_array(self):
        """Checks the inverse inverse logit transformation of an array
        """
        prbs = np.array([0.5, 0.25, 0.75, 0.5])
        odds = np.array([0., -1.098612288668, 1.098612288668, 0.])

        npt.assert_allclose(inverse_logit(odds), prbs)

    def test_rloss_huber(self):
        """Checks the robust loss function: Huber's
        """
        residuals = np.array([-5, 1, 0, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='huber', k=4)
        byhand = np.array([-4, 1, 0, -2, 4, 3])

        npt.assert_allclose(func, byhand)

    def test_rloss_tukey(self):
        """Checks the robust loss function: Tukey's biweight
        """
        residuals = np.array([-5, 1, 0.1, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='tukey', k=4)

        byhand = np.array([-5 * 0,
                           1 * ((1-(1/4)**2)**2),
                           0.1 * ((1-(0.1/4)**2)**2),
                           -2 * ((1-(2/4)**2)**2),
                           8 * 0,
                           3 * ((1-(3/4)**2)**2)])

        npt.assert_allclose(func, byhand)

    def test_rloss_andrew(self):
        """Checks the robust loss function: Andrew's Sine
        """
        residuals = np.array([-5, 1, 0.1, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='andrew', k=1)

        byhand = np.array([-5 * 0,
                           np.sin(1 / 1),
                           np.sin(0.1 / 1),
                           np.sin(-2 / 1),
                           8 * 0,
                           np.sin(3 / 1)])

        npt.assert_allclose(func, byhand)

    def test_rloss_hampel(self):
        """Checks the robust loss function: Hampel
        """
        residuals = np.array([-5, 1, 1.5, -1.3, 0.1, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='hampel',
                                     k=4, a=1, b=2)

        byhand = np.array([-5 * 0,
                           1,
                           1,
                           -1,
                           0.1,
                           (-4 + 2)/(-4 + 2)*-1,
                           8 * 0,
                           (4 - 3)/(4 - 2)*1])

        npt.assert_allclose(func, byhand)

    def test_rloss_hampel_error(self):
        residuals = np.array([-5, 1, 1.5, -1.3, 0.1, -2, 8, 3])

        # All parameters are specified
        with pytest.raises(ValueError, match="requires the optional"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=4)

        # Ordering of parameters
        with pytest.raises(ValueError, match="requires that a < b < k"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=-4, a=1, b=2)
        with pytest.raises(ValueError, match="requires that a < b < k"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=4, a=1, b=-2)
        with pytest.raises(ValueError, match="requires that a < b < k"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=1.5, a=1, b=2)

    def test_spline_unnormed(self):
        vars = [1, 5, 10, 15, 20]

        # Spline setup 1
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 4.0], ]).T
        returned = spline(variable=vars, knots=[16, ], power=1, restricted=False, normalized=False)
        npt.assert_allclose(returned, expected)

        # Spline setup 2
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 16.0], ]).T
        returned = spline(variable=vars, knots=[16, ], power=2, restricted=False, normalized=False)
        npt.assert_allclose(returned, expected)

        # Spline setup 3
        expected = np.array([[0.0, 0.0, 0.0, 5.0**1.5, 10.0**1.5],
                             [0.0, 0.0, 0.0, 0.0, 4.0**1.5]]).T
        returned = spline(variable=vars, knots=[10, 16], power=1.5, restricted=False, normalized=False)
        npt.assert_allclose(returned, expected)

        # Spline setup 4
        expected = np.array([[0.0, 0.0, 0.0, 5.0, 6.0], ]).T
        returned = spline(variable=vars, knots=[10, 16], power=1, restricted=True, normalized=False)
        npt.assert_allclose(returned, expected)

        # Spline setup 5
        expected = np.array([[0.0, 0.0, 5.0**2, 10.0**2, 15.0**2 - 4.0**2], ]).T
        returned = spline(variable=vars, knots=[5, 16], power=2, restricted=True, normalized=False)
        npt.assert_allclose(returned, expected)

    def test_spline_normed(self):
        vars = [1, 5, 10, 15, 20]

        # Spline setup 1
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 4.0], ]).T / 16
        returned = spline(variable=vars, knots=[16, ], power=1, restricted=False, normalized=True)
        npt.assert_allclose(returned, expected)

        # Spline setup 2
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 16.0], ]).T / 16**2
        returned = spline(variable=vars, knots=[16, ], power=2, restricted=False, normalized=True)
        npt.assert_allclose(returned, expected)

        # Spline setup 3
        expected = np.array([[0.0, 0.0, 0.0, 5.0**1.5, 10.0**1.5],
                             [0.0, 0.0, 0.0, 0.0, 4.0**1.5]]).T / (16 - 10)**1.5
        returned = spline(variable=vars, knots=[10, 16], power=1.5, restricted=False, normalized=True)
        npt.assert_allclose(returned, expected)

        # Spline setup 4
        expected = np.array([[0.0, 0.0, 0.0, 5.0, 6.0], ]).T / (16 - 10)
        returned = spline(variable=vars, knots=[10, 16], power=1, restricted=True, normalized=True)
        npt.assert_allclose(returned, expected)

        # Spline setup 5
        expected = np.array([[0.0, 0.0, 5.0**2, 10.0**2, 15.0**2 - 4.0**2], ]).T / (16 - 5)**2
        returned = spline(variable=vars, knots=[5, 16], power=2, restricted=True, normalized=True)
        npt.assert_allclose(returned, expected)


class TestGroupedData:

    @pytest.fixture
    def data(self):
        d = pd.DataFrame()
        d['X'] = [1, 0, 1, 0, 1, 1, 1, 0, 0, 1]
        d['Y'] = [1, -1, 0, 3, 2, -3, -2, -1, 1, 0]
        d['group'] = [1, 1, 1, 2, 2, 3, 4, 4, 5, 6]
        d['C'] = 1
        return d

    def test_error_1dim(self, data):
        group = list(data['group'])[:-1]

        def psi(theta):
            return aggregate_efuncs(ee_mean(theta=theta, y=data['Y']),
                                    group=group)

        estr = MEstimator(psi, init=[0., ])
        with pytest.raises(ValueError, match="vector must match the number of units"):
            estr.estimate()

    def test_error_2dim(self, data):
        group = list(data['group'])[:-1]

        def psi(theta):
            return aggregate_efuncs(ee_regression(theta=theta, X=data[['C', 'X']], y=data['Y'], model='linear'),
                                    group=group)

        estr = MEstimator(psi, init=[0., 0.])
        with pytest.raises(ValueError, match="vector must match the number of units"):
            estr.estimate()

    def test_grouped_mean(self, data):

        def psi(theta):
            return aggregate_efuncs(ee_mean(theta=theta, y=data['Y']),
                                    group=data['group'])

        estr = MEstimator(psi, init=[0., ])
        estr.estimate()

        gee = smf.gee("Y ~ 1", "group", data,
                      cov_struct=sm.cov_struct.Independence(),
                      family=sm.families.Gaussian()).fit()

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(gee.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.variance,
                            np.asarray(gee.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(estr.confidence_intervals(),
                            np.asarray(gee.conf_int()),
                            atol=1e-6)

    def test_grouped_regression(self, data):

        def psi(theta):
            return aggregate_efuncs(ee_regression(theta=theta, X=data[['C', 'X']], y=data['Y'], model='linear'),
                                    group=data['group'])

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate()

        gee = smf.gee("Y ~ X", "group", data,
                      cov_struct=sm.cov_struct.Independence(),
                      family=sm.families.Gaussian()).fit()

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(gee.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.variance,
                            np.asarray(gee.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(estr.confidence_intervals(),
                            np.asarray(gee.conf_int()),
                            atol=1e-6)


class TestPredictions:

    @pytest.fixture
    def bcancer(self):
        # Collett Breast Cancer data set
        d = pd.DataFrame()
        d['X'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        d['delta'] = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                      0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        d['t'] = [23, 47, 69, 70, 71, 100, 101, 148, 181, 198, 208, 212, 224, 5, 8, 10, 13, 18, 24, 26, 26, 31, 35, 40,
                  41, 48, 50, 59, 61, 68, 71, 76, 105, 107, 109, 113, 116, 118, 143, 154, 162, 188, 212, 217, 225]
        d['C'] = 1
        return d

    def test_regression_predictions_linear(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2*data['X'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        p = pd.DataFrame()
        p['X'] = np.linspace(-2.5, 2.5, 20)
        p['C'] = 1

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X']], y=data['Y'],
                                 model='linear')

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate(solver='lm')
        returned = regression_predictions(X=p[['C', 'X']], theta=estr.theta, covariance=estr.variance)

        # Statsmodels function equivalent
        glm = smf.glm("Y ~ X", data).fit(cov_type="HC1")
        expected = np.asarray(glm.get_prediction(p).summary_frame())
        expected[:, 1] = expected[:, 1]**2

        npt.assert_allclose(returned, expected, atol=1e-6)

    def test_regression_predictions_logit(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(-0.5 + 2*data['X']), size=n)
        data['C'] = 1

        p = pd.DataFrame()
        p['X'] = np.linspace(-2.5, 2.5, 20)
        p['C'] = 1

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X']], y=data['Y'],
                                 model='logistic')

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate(solver='lm')
        returned = regression_predictions(X=p[['C', 'X']], theta=estr.theta, covariance=estr.variance)

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X", data, family=sm.families.Binomial()).fit(cov_type="HC1")
        expected = np.asarray(glm.get_prediction(p).summary_frame())

        npt.assert_allclose(returned[:, 0], logit(expected[:, 0]), atol=1e-6)
        npt.assert_allclose(returned[:, 2], logit(expected[:, 2]), atol=1e-6)
        npt.assert_allclose(returned[:, 3], logit(expected[:, 3]), atol=1e-6)

    def test_regression_predictions_poisson(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(-1 + 2*data['X']), size=n)
        data['C'] = 1

        p = pd.DataFrame()
        p['X'] = np.linspace(-2.5, 2.5, 20)
        p['C'] = 1

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X']], y=data['Y'],
                                 model='poisson')

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate(solver='lm')
        returned = regression_predictions(X=p[['C', 'X']], theta=estr.theta, covariance=estr.variance)

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X", data, family=sm.families.Poisson()).fit(cov_type="HC1")
        expected = np.asarray(glm.get_prediction(p).summary_frame())

        npt.assert_allclose(returned[:, 0], np.log(expected[:, 0]), atol=1e-6)
        npt.assert_allclose(returned[:, 2], np.log(expected[:, 2]), atol=1e-6)
        npt.assert_allclose(returned[:, 3], np.log(expected[:, 3]), atol=1e-6)

    def test_survival_exp_survival(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'exponential'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[1., ])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='survival')

        # Exponential fitter
        exf = ExponentialFitter()
        exf.fit(times, events)
        results = np.asarray(exf.survival_function_at_times(times=times_to_predict))

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_exp_risk(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'exponential'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[1., ])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='risk')

        # Exponential fitter
        exf = ExponentialFitter()
        exf.fit(times, events)
        results = exf.cumulative_density_at_times(times=times_to_predict)

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_exp_hazard(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'exponential'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[1., ])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='hazard')

        # Exponential fitter
        exf = ExponentialFitter()
        exf.fit(times, events)
        results = np.asarray(exf.summary['coef'])[0]

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            1 / results,
                            atol=1e-5)

    def test_survival_exp_chazard(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'exponential'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[1., ])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='cumulative_hazard')

        # Exponential fitter
        exf = ExponentialFitter()
        exf.fit(times, events)
        results = exf.cumulative_hazard_at_times(times=times_to_predict)

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_weibull_survival(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'weibull'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[0.05, 1.])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='survival')

        # Lifelines Weibull for comparison
        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.survival_function_at_times(times=times_to_predict))

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_weibull_risk(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'weibull'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[0.05, 1.])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='risk')

        # Lifelines Weibull for comparison
        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.cumulative_density_at_times(times=times_to_predict))

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_weibull_hazard(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'weibull'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[0.05, 1.])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='hazard')

        # Lifelines Weibull for comparison
        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.hazard_at_times(times=times_to_predict))

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_weibull_chazard(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'weibull'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[0.05, 1.])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='cumulative_hazard')

        # Lifelines Weibull for comparison
        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.cumulative_hazard_at_times(times=times_to_predict))

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_weibull_density(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        times_to_predict = [10, 20, 30, 50, 80, 90, 100, 150, 175, 200, 250]
        dist = 'weibull'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        estr = MEstimator(psi, init=[0.05, 1.])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='density')

        # Lifelines Weibull for comparison
        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.density_at_times(times=times_to_predict))

        # Checking mean estimate
        npt.assert_allclose(s_hat[:, 0],
                            results,
                            atol=1e-5)

    def test_survival_model_variance(self, bcancer):
        times, events = bcancer['t'], bcancer['delta']
        n = bcancer.shape[0]
        times_to_predict = [10, 50, 150, 250]
        dist = 'weibull'

        def psi(theta):
            return ee_survival_model(theta=theta, t=times, delta=events,
                                     distribution=dist)

        # Built-in functionality via Delta Method
        estr = MEstimator(psi, init=[0.05, 1.])
        estr.estimate()
        s_hat = survival_predictions(times=times_to_predict, theta=estr.theta, covariance=estr.variance,
                                     distribution=dist, measure='survival')

        # Stacked estimating functions version of delta method
        def convert_to_survival(t, theta):
            lambd, gamma = theta[0], theta[1]
            return np.exp(-lambd * (t ** gamma))

        def psi(theta):
            ee_sm = ee_survival_model(theta=theta[:2], t=times, delta=events, distribution=dist)
            ee_sp = []
            for t, p in zip(times_to_predict, theta[2:]):
                s_t = convert_to_survival(t=t, theta=theta[:2])
                ee_sp.append(np.ones(n) * s_t - p)
            return np.vstack([ee_sm, ee_sp])

        estr = MEstimator(psi, init=[0.05, 1., 0.5, 0.5, 0.5, 0.5])
        estr.estimate(deriv_method='exact')
        ci = estr.confidence_intervals()
        byhand = np.asarray([estr.theta[2:], np.diag(estr.variance)[2:], ci[2:, 0], ci[2:, 1]]).T

        # Checking outputs are equal
        npt.assert_allclose(s_hat,
                            byhand,
                            atol=1e-7)

    def test_aft_ind_survival(self, bcancer):
        dist = 'weibull'
        measure = 'survival'
        times, events = bcancer['t'], bcancer['delta']
        dmatrix = bcancer[['C', 'X']]
        times_to_predict = [10, 50, 150, 200]

        # M-estimator with built-in Weibull AFT
        def psi(theta):
            return ee_aft(theta=theta, X=dmatrix, t=times, delta=events, distribution=dist)

        estr = MEstimator(psi, init=[6, -1, 0])
        estr.estimate()

        s_i_hat = aft_predictions_individual(X=dmatrix, times=times_to_predict, theta=estr.theta,
                                             distribution=dist, measure=measure)

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(bcancer[['X', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_survival_function(bcancer, times=times_to_predict)

        # Checking mean estimate
        npt.assert_allclose(s_i_hat,
                            np.asarray(preds).T,
                            atol=1e-5)

    def test_aft_ind_risk(self, bcancer):
        dist = 'weibull'
        measure = 'risk'
        times, events = bcancer['t'], bcancer['delta']
        dmatrix = bcancer[['C', 'X']]
        times_to_predict = [10, 50, 150, 250]

        # M-estimator with built-in Weibull AFT
        def psi(theta):
            return ee_aft(theta=theta, X=dmatrix, t=times, delta=events, distribution=dist)

        estr = MEstimator(psi, init=[6, -1, 0])
        estr.estimate()

        s_i_hat = aft_predictions_individual(X=dmatrix, times=times_to_predict, theta=estr.theta,
                                             distribution=dist, measure=measure)

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(bcancer[['X', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = 1 - waft.predict_survival_function(bcancer, times=times_to_predict)

        # Checking mean estimate
        npt.assert_allclose(s_i_hat,
                            np.asarray(preds).T,
                            atol=1e-5)

    def test_aft_ind_hazard(self, bcancer):
        dist = 'weibull'
        measure = 'hazard'
        times, events = bcancer['t'], bcancer['delta']
        dmatrix = bcancer[['C', 'X']]
        times_to_predict = [10, 50, 150, 250]

        # M-estimator with built-in Weibull AFT
        def psi(theta):
            return ee_aft(theta=theta, X=dmatrix, t=times, delta=events, distribution=dist)

        estr = MEstimator(psi, init=[6, -1, 0])
        estr.estimate()

        s_i_hat = aft_predictions_individual(X=dmatrix, times=times_to_predict, theta=estr.theta,
                                             distribution=dist, measure=measure)

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(bcancer[['X', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_hazard(bcancer, times=times_to_predict)

        # Checking mean estimate
        npt.assert_allclose(s_i_hat,
                            np.asarray(preds).T,
                            atol=1e-5)

    def test_aft_ind_chazard(self, bcancer):
        dist = 'weibull'
        measure = 'cumulative_hazard'
        times, events = bcancer['t'], bcancer['delta']
        dmatrix = bcancer[['C', 'X']]
        times_to_predict = [10, 50, 150, 250]

        # M-estimator with built-in Weibull AFT
        def psi(theta):
            return ee_aft(theta=theta, X=dmatrix, t=times, delta=events, distribution=dist)

        estr = MEstimator(psi, init=[6, -1, 0])
        estr.estimate()

        s_i_hat = aft_predictions_individual(X=dmatrix, times=times_to_predict, theta=estr.theta,
                                             distribution=dist, measure=measure)

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(bcancer[['X', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_cumulative_hazard(bcancer, times=times_to_predict)

        # Checking mean estimate
        npt.assert_allclose(s_i_hat,
                            np.asarray(preds).T,
                            atol=5e-5)

    def test_aft_f_variance(self, bcancer):
        dist = 'weibull'
        measure = 'survival'
        times, events = bcancer['t'], bcancer['delta']
        dmatrix = bcancer[['C', 'X']]
        times_to_predict = [10, 50, 150, 250]
        n = bcancer.shape[0]

        def psi(theta):
            return ee_aft(theta=theta, X=dmatrix, t=times, delta=events, distribution=dist)

        estr = MEstimator(psi, init=[6, -1, 0])
        estr.estimate()
        s_t_hat = aft_predictions_function(X=[[1, 0], ], times=times_to_predict,
                                           theta=estr.theta, covariance=estr.variance,
                                           distribution=dist, measure=measure)

        # Stacked estimating functions version of delta method
        def psi(theta):
            ee_sm = ee_aft(theta=theta[:3], X=dmatrix, t=times, delta=events, distribution=dist)
            param = aft_predictions_individual(X=[[1, 0]], times=times_to_predict,
                                               theta=theta[:3], distribution=dist, measure=measure)
            ee_sp = []
            for p_t, p in zip(theta[3:], param[0]):
                ee_sp.append(np.ones(n) * p - p_t)
            return np.vstack([ee_sm, ee_sp])

        estr = MEstimator(psi, init=[5.8, -1., -0.1, 0.96, 0.85, 0.64, 0.5])
        estr.estimate(deriv_method='exact')
        ci = estr.confidence_intervals()
        byhand = np.asarray([estr.theta[3:], np.diag(estr.variance)[3:], ci[3:, 0], ci[3:, 1]]).T

        # Checking outputs are equal
        npt.assert_allclose(s_t_hat,
                            byhand,
                            atol=1e-6)


class TestDesignMatrix:

    def test_adm_error_noknots(self, design_matrix):
        # Testing error when dictionary with no knots given
        specs = [None,
                 {"power": 2},
                 None]
        with pytest.raises(ValueError, match="`knots` must be"):
            additive_design_matrix(X=design_matrix,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_error_negknots(self, design_matrix):
        # Testing error when dictionary with negative number of knots given
        specs = [None,
                 {"knots": 2},
                 None]
        with pytest.raises(TypeError, match="not iterable"):
            additive_design_matrix(X=design_matrix,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_error_misalign(self, design_matrix):
        # Testing error matrix is misaligned
        # Testing warning for extra spline arguments
        specs = [None,
                 {"knots": [-1, 1]},
                 None]
        with pytest.raises(ValueError, match="number of input"):
            additive_design_matrix(X=design_matrix.T,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_warn_extras(self, design_matrix):
        # Testing warning for extra spline arguments
        specs = [None,
                 {"knots": [-1, 1], "extra": 4},
                 None]
        with pytest.warns(UserWarning, match="following keys"):
            additive_design_matrix(X=design_matrix,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_defaults(self, design_matrix):
        # Testing default additive design matrix specifications
        specs = [None,
                 {"knots": [5, 16]},
                 {"knots": [2, 4]}]
        expected_matrix = np.array([[1, 1, 1, 1, 1],
                                    [1, 5, 10, 15, 20],
                                    [0.0, 0.0, 5.0**3, 10.0**3, 15.0**3 - 4.0**3],
                                    [1, 2, 3, 4, 5],
                                    [0, 0, 1, 2**3, 3**3 - 1]]).T
        expected_penalty = np.array([0, 0, 0, 0, 0])
        returned_matrix, returned_penalty = additive_design_matrix(X=design_matrix,
                                                                   specifications=specs,
                                                                   return_penalty=True)
        npt.assert_allclose(returned_matrix, expected_matrix)
        npt.assert_allclose(returned_penalty, expected_penalty)

    def test_adm_v1(self, design_matrix):
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False, "normalized": False},
                 None]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4],
                             [1, 2, 3, 4, 5]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_v2(self, design_matrix):
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False, "normalized": False},
                 {"knots": [2, 4], "power": 2, "natural": True, "normalized": False}]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4],
                             [1, 2, 3, 4, 5],
                             [0, 0, 1, 4, 3**2 - 1]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_v3(self, design_matrix):
        # Testing non-ascending order correction for knots
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False, "normalized": False},
                 {"knots": [4, 2], "power": 2, "natural": True, "normalized": False}]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4],
                             [1, 2, 3, 4, 5],
                             [0, 0, 1, 4, 3**2 - 1]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_v4(self, design_matrix):
        # Testing number of generated rows
        specs = [None,
                 {"knots": [16, ], "power": 2, "natural": False, "normalized": False},
                 {"knots": [2, 4], "power": 1, "natural": False, "normalized": False}]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4**2],
                             [1, 2, 3, 4, 5],
                             [0, 0, 1, 2, 3],
                             [0, 0, 0, 0, 1]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_v5(self, design_matrix):
        # Testing number of generated rows
        specs = [None,
                 {"knots": [16, ], "power": 2, "natural": False, "normalized": True},
                 {"knots": [2, 4], "power": 1, "natural": False, "normalized": True}]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0/16**2, 0, 4**2/16**2],
                             [1, 2, 3, 4, 5],
                             [0, 0, 1/(4-2), 2/(4-2), 3/(4-2)],
                             [0, 0, 0, 0, 1/(4-2)]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_penalty(self, design_matrix):
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False, "penalty": 3},
                 {"knots": [2, 3, 4], "power": 2, "natural": True, "penalty": 5}]
        expected = np.array([0, 0, 3, 0, 5, 5])
        design, returned = additive_design_matrix(X=design_matrix,
                                                  specifications=specs,
                                                  return_penalty=True)
        npt.assert_allclose(returned, expected)
