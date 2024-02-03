import pytest
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import DomainWarning
from scipy.stats import logistic
from lifelines import ExponentialFitter, WeibullFitter, WeibullAFTFitter

from delicatessen import MEstimator
from delicatessen.estimating_equations import (ee_mean, ee_mean_variance, ee_mean_robust,
                                               # Regression models
                                               ee_regression, ee_glm, ee_robust_regression, ee_ridge_regression,
                                               ee_additive_regression, ee_elasticnet_regression,
                                               # Survival models
                                               ee_exponential_model, ee_exponential_measure, ee_weibull_model,
                                               ee_weibull_measure, ee_aft_weibull, ee_aft_weibull_measure,
                                               # Dose-Response
                                               ee_2p_logistic, ee_3p_logistic, ee_4p_logistic, ee_effective_dose_delta,
                                               # Measurement error
                                               ee_rogan_gladen,
                                               # Causal inference
                                               ee_gformula, ee_ipw, ee_ipw_msm, ee_aipw, ee_gestimation_snmm,
                                               ee_mean_sensitivity_analysis)
from delicatessen.data import load_inderjit
from delicatessen.utilities import additive_design_matrix, inverse_logit


np.random.seed(236461)

# TODO write more compact data generation (as functions instead of unique creations in g-estimation)


class TestEstimatingEquationsBase:

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

        def byhand(theta):
            k = 6
            ee_rm = np.array(y) - theta
            ee_rm = np.where(ee_rm > k, k, ee_rm)
            ee_rm = np.where(ee_rm < -k, -k, ee_rm)
            return ee_rm

        ref = MEstimator(byhand, init=[0, ])
        ref.estimate()

        def psi(theta):
            return ee_mean_robust(theta=theta, y=y, k=6)

        mestimator = MEstimator(psi, init=[0, ])
        mestimator.estimate()

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[0],
                            ref.theta[0],
                            atol=1e-9)

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


class TestEstimatingEquationsRegression:

    def test_error_regression(self):
        """Test for error raised when incorrect regression name is provided
        """
        n = 100
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = data['x1'] + np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + np.random.normal(size=n)
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])

        def psi(theta):
            return ee_regression(theta, X=Xvals, y=yvals, model=748)

        estr = MEstimator(psi, init=[5, 1, 1])
        with pytest.raises(ValueError, match="The model argument"):
            estr.estimate(solver='lm')

        def psi(theta):
            return ee_regression(theta, X=Xvals, y=yvals, model='magic')

        estr = MEstimator(psi, init=[5, 1, 1])
        with pytest.raises(ValueError, match="Invalid input"):
            estr.estimate(solver='lm')

    def test_error_robust(self):
        n = 1000
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = data['x1'] + np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + np.random.normal(size=n)
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])

        def psi(theta):
            return ee_robust_regression(theta, X=Xvals, y=yvals, model='logistic', k=5)

        estr = MEstimator(psi, init=[5, 1, 1])
        with pytest.raises(ValueError, match="only supports linear"):
            estr.estimate(solver='lm')

    def test_error_penalized(self):
        n = 1000
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = data['x1'] + np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + np.random.normal(size=n)
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])

        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear',
                                       penalty=[0.5, 5.], weights=None)

        estr = MEstimator(psi, init=[5, 1, 1])
        with pytest.raises(ValueError, match="The penalty term must"):
            estr.estimate(solver='lm')

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
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='linear')

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

        # Checking confidence interval estimates
        npt.assert_allclose(mpee.confidence_intervals(),
                            np.asarray(glm.conf_int()),
                            atol=1e-6)

    def test_ols_offset(self):
        """Tests linear regression with the built-in estimating equation and an offset term.
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        def psi_builtin_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 offset=data['C'],
                                 model='linear')

        mpee = MEstimator(psi_builtin_regression, init=[0.1, 0.1, 0.1])
        mpee.estimate()

        # Statsmodels function equivalent
        glm = smf.glm("Y ~ X + Z", data, offset=data['C']).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mpee.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mpee.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(mpee.confidence_intervals(),
                            np.asarray(glm.conf_int()),
                            atol=1e-6)

    def test_wls(self):
        """Tests weighted linear regression
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2 * data['X'] - 1 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='linear', weights=data['w'])

        mestimator = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, freq_weights=data['w']).fit(cov_type="cluster",
                                                                     cov_kwds={"groups": data.index,
                                                                               "use_correction": False})

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

    def test_wls_offset(self):
        """Tests weighted linear regression with an offset term
        """
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2 * data['X'] - 1 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='linear', weights=data['w'], offset=data['C'])

        mestimator = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, freq_weights=data['w'],
                      offset=data['C']).fit(cov_type="cluster",
                                            cov_kwds={"groups": data.index,
                                                      "use_correction": False})

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

    def test_ridge_ols(self):
        """Tests the ridge (L2) variation of the linear regression built-in estimating equation
        """
        n = 1000
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = data['x1'] + np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + np.random.normal(size=n)
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])

        # Penalty of 0.5
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear', penalty=0.5, weights=None)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge = sm.OLS(yvals, Xvals).fit_regularized(L1_wt=0., alpha=0.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

        # Penalty of 5.0
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear', penalty=5.0, weights=None)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge = sm.OLS(yvals, Xvals).fit_regularized(L1_wt=0., alpha=5. / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

        # Testing array of penalty terms
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear', penalty=[0., 5., 2.], weights=None)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge = sm.OLS(yvals, Xvals).fit_regularized(L1_wt=0., alpha=np.array([0., 5., 2.]) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

    def test_ridge_ols_offset(self):
        """Tests the ridge (L2) variation of the linear regression built-in estimating equation with an offset term
        """
        n = 10000
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = data['x1'] + np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + np.random.normal(size=n)
        data['off'] = np.random.uniform(0.4, 0.6, size=n)
        # data['off'] = 1
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])

        # Penalty of 0.5
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear',
                                       penalty=[0., 5., 2.],
                                       weights=None, offset=data['off']
                                       )

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge = sm.OLS(yvals - data['off'], Xvals,).fit_regularized(L1_wt=0.,
                                                                    alpha=np.array([0., 5., 2.]) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

    def test_ridge_wls(self):
        """Tests the ridge (L2) variation of the weighted linear regression built-in estimating equation
        """
        n = 1000
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = data['x1'] + np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + np.random.normal(size=n)
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])
        weights = np.random.uniform(0.1, 2.5, size=n)

        # Penalty of 0.5
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear', penalty=0.5, weights=weights)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        wridge = sm.WLS(yvals, Xvals, weights=weights).fit_regularized(L1_wt=0., alpha=0.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(wridge.params),
                            atol=1e-6)

        # Penalty of 5.0
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear', penalty=5.0, weights=weights)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        wridge = sm.WLS(yvals, Xvals, weights=weights).fit_regularized(L1_wt=0., alpha=5. / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(wridge.params),
                            atol=1e-6)

        # Testing array of penalty terms
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear', penalty=[0., 5., 2.], weights=weights)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        wridge = sm.WLS(yvals, Xvals, weights=weights).fit_regularized(L1_wt=0.,
                                                                       alpha=np.array([0., 5., 2.]) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(wridge.params),
                            atol=1e-6)

    def test_additive_ols(self):
        n = 1000
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + data['x2'] + 0.01*data['x2'] + np.random.normal(size=n)
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 3}, {"knots": [-2, -1, 0, 1, 2], "penalty": 5}]

        # Testing array of penalty terms
        def psi(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals, model='linear', specifications=spec)

        estr = MEstimator(psi, init=[5, 1, 0, 0, 1, 0, 0, 0, 0])
        estr.estimate(solver='lm')

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        ridge = sm.OLS(yvals, design).fit_regularized(L1_wt=0., alpha=np.array(penalty)/Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

    def test_additive_wls(self):
        n = 1000
        data = pd.DataFrame()
        data['x1'] = np.random.normal(size=n)
        data['x2'] = np.random.normal(scale=0.1, size=n)
        data['c'] = 1
        data['y'] = 5 + data['x1'] + data['x2'] + 0.01*data['x2'] + np.random.normal(size=n)
        Xvals = np.asarray(data[['c', 'x1', 'x2']])
        yvals = np.asarray(data['y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 3}, {"knots": [-2, -1, 0, 1, 2], "penalty": 5}]
        weights = np.random.uniform(0.1, 2.5, size=n)

        # Testing array of penalty terms
        def psi(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals, model='linear',
                                          specifications=spec, weights=weights)

        estr = MEstimator(psi, init=[5, 1, 0, 0, 1, 0, 0, 0, 0])
        estr.estimate(solver='lm')

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        ridge = sm.WLS(yvals, design, weights=weights).fit_regularized(L1_wt=0.,
                                                                       alpha=np.array(penalty)/Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

    def test_logistic(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1

        def psi_builtin_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='logistic')

        mpee = MEstimator(psi_builtin_regression, init=[0., 0., 0.])
        mpee.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, family=sm.families.Binomial()).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mpee.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mpee.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(mpee.confidence_intervals(),
                            np.asarray(glm.conf_int()),
                            atol=1e-6)

    def test_logistic_offset(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        data['off'] = np.random.uniform(0.4, 0.6, size=n)

        def psi_builtin_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='logistic', offset=data['off'])

        mpee = MEstimator(psi_builtin_regression, init=[0., 0., 0.])
        mpee.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, family=sm.families.Binomial(), offset=data['off']).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mpee.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mpee.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(mpee.confidence_intervals(),
                            np.asarray(glm.conf_int()),
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
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='logistic', weights=data['w'])

        mestimator = MEstimator(psi_regression, init=[0., 2., -1.])
        mestimator.estimate(solver='lm')

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, freq_weights=data['w'],
                      family=sm.families.Binomial()).fit(cov_type="cluster",
                                                         cov_kwds={"groups": data.index,
                                                                   "use_correction": False})

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

    def test_ridge_logistic(self):
        """Tests ridge logistic regression by-hand with a single estimating equation.
        """
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='logistic', penalty=0.5, weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        f = sm.families.Binomial()
        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=0.5 / Xvals.shape[0], tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=5e-4)

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='logistic', penalty=5., weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='hybr', tolerance=1e-12)

        f = sm.families.Binomial()
        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=5. / Xvals.shape[0], tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=5e-4)

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='logistic', penalty=[0., 5., 2.], weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='hybr', tolerance=1e-12)

        f = sm.families.Binomial()
        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=np.array([0., 5., 2.]) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=5e-4)

    def test_additive_logistic(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 3}, {"knots": [-2, -1, 0, 1, 2], "penalty": 5}]

        def psi_regression(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals,
                                          model='logistic',
                                          specifications=spec)

        mestimator = MEstimator(psi_regression, init=[0., 2., 0., 0., 1., 0., 0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-14)

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        f = sm.families.Binomial()
        lgt = sm.GLM(yvals, design, family=f).fit_regularized(L1_wt=0., alpha=np.array(penalty)/Xvals.shape[0],
                                                              tol=1e-14)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-3)

    def test_poisson(self):
        """Tests Poisson regression by-hand with a single estimating equation.
        """
        np.random.seed(20212345)
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(1 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1

        def psi_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='poisson')

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm')

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, family=sm.families.Poisson()).fit(cov_type="HC1")

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

    def test_weighted_poisson(self):
        """Tests weighted Poisson regression by-hand with a single estimating equation.
        """
        np.random.seed(1234)
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(1 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 3, size=n)

        def psi_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='poisson', weights=data['w'])

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm')

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", data, freq_weights=data['w'],
                      family=sm.families.Poisson()).fit(cov_type="cluster",
                                                        cov_kwds={"groups": data.index,
                                                                  "use_correction": False})

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

    def test_ridge_poisson(self):
        """Tests ridge Poisson regression by-hand with a single estimating equation.
        """
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(1 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='poisson', penalty=0.5, weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        f = sm.families.Poisson()
        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=0.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-6)

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='poisson', penalty=2.5, weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=2.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-6)

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='poisson', penalty=[0., 5., 2.5], weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=np.asarray([0., 5., 2.5]) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-6)

    def test_ridge_wpoisson(self):
        """Tests weighted ridge Poisson regression by-hand with a single estimating equation.
        """
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(1 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])
        weights = np.random.uniform(0.5, 2, size=n)

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='poisson', penalty=0.5, weights=weights)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        f = sm.families.Poisson()
        lgt = sm.GLM(yvals, Xvals, family=f, freq_weights=weights).fit_regularized(L1_wt=0., alpha=0.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-5)

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='poisson', penalty=2.5, weights=weights)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        lgt = sm.GLM(yvals, Xvals, family=f, freq_weights=weights).fit_regularized(L1_wt=0., alpha=2.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=2e-5)

        def psi_regression(theta):
            return ee_ridge_regression(theta,
                                       X=Xvals, y=yvals,
                                       model='poisson', penalty=[0., 5., 2.5], weights=weights)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        lgt = sm.GLM(yvals, Xvals, family=f, freq_weights=weights).fit_regularized(L1_wt=0.,
                                                                                   alpha=np.asarray([0., 5., 2.5]
                                                                                                    ) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=5e-4)

    def test_additive_poisson(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(1 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 3}, {"knots": [-2, -1, 0, 1, 2], "penalty": 5}]

        def psi_regression(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals,
                                          model='poisson',
                                          specifications=spec)

        mestimator = MEstimator(psi_regression, init=[0., 2., 0., 0., 1., 0., 0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        f = sm.families.Poisson()
        lgt = sm.GLM(yvals, design, family=f).fit_regularized(L1_wt=0., alpha=np.array(penalty)/Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-5)

    def test_glm_normal_identity(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, -1, 4, 3, 3, 1, -2, 4, -2, 3, 6, 6, 8, 7, 1, -2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='normal', link='identity')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm')

        fam = sm.families.Gaussian(sm.families.links.identity())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_normal_log(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, -1, 4, 3, 3, 1, -2, 4, -2, 3, 6, 6, 8, 7, 1, -2, 5]
        d['Y'] = d['Y'] + 5
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='normal', link='log')

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Gaussian(sm.families.links.log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_poisson_identity(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='poisson', link='identity')

        mestr = MEstimator(psi, init=[1., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Poisson(sm.families.links.identity())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_poisson_log(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='poisson', link='log')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Poisson(sm.families.links.log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_poisson_sqrt(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='poisson', link='sqrt')

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Poisson(sm.families.links.sqrt())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_logit(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='logit')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.logit())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_log(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='log')

        mestr = MEstimator(psi, init=[-.9, 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_identity(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='identity')

        mestr = MEstimator(psi, init=[.2, 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.identity())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_probit(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='probit')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.probit())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_cauchy(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='cauchy')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.cauchy())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_cloglog(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='cloglog')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.cloglog())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_loglog(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='loglog')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.loglog())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_logit_offset(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1
        d['F'] = [1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3]

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='logit',
                          offset=d['F'])

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.logit())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], offset=d['F'], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_gamma_log(self):
        # Example data comes from R's MASS library
        d = pd.DataFrame()
        d['X'] = np.log([5, 10, 15, 20, 30, 40, 60, 80, 100])
        d['Y'] = [118, 58, 42, 35, 27, 25, 21, 19, 18]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X']], y=d['Y'],
                          distribution='gamma', link='log')

        mestr = MEstimator(psi, init=[0., 0., 1.])
        mestr.estimate(solver='lm', maxiter=5000)

        # Log-Gamma with statsmodels (only includes scale parameters)
        fam = sm.families.Gamma(sm.families.links.Log())
        glm = sm.GLM(d['Y'], d[['I', 'X']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta[0:2],
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance[0:2, 0:2],
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

        # Previously solved log Gamma shape parameter using MASS
        # library(MASS)
        # clotting <- data.frame(
        #     u = c(5,10,15,20,30,40,60,80,100),
        #     lot1 = c(118,58,42,35,27,25,21,19,18),
        #     lot2 = c(69,35,26,21,18,16,13,12,12))
        # clot1 <- glm(lot1 ~ log(u), data = clotting, family = Gamma(link='log'))
        # summary(clot1)
        # gamma.shape(clot1)
        alpha_param = np.log(55.51389)

        # Checking SNM parameters
        npt.assert_allclose(mestr.theta[2],
                            alpha_param,
                            atol=1e-4)

    def test_glm_gamma_identity(self):
        # Example data comes from R's MASS library
        d = pd.DataFrame()
        d['X'] = np.log([5, 10, 15, 20, 30, 40, 60, 80, 100])
        d['Y'] = [118, 58, 42, 35, 27, 25, 21, 19, 18]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X']], y=d['Y'],
                          distribution='gamma', link='identity')

        mestr = MEstimator(psi, init=[100., -10., 10.])
        mestr.estimate(solver='lm', maxiter=5000)

        # Log-Gamma with statsmodels (only includes scale parameters)
        fam = sm.families.Gamma(sm.families.links.identity())
        glm = sm.GLM(d['Y'], d[['I', 'X']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta[0:2],
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance[0:2, 0:2],
                            np.asarray(glm.cov_params()),
                            rtol=1e-4)

        # Previously solved log Gamma shape parameter using MASS
        # library(MASS)
        # clotting <- data.frame(
        #     u = c(5,10,15,20,30,40,60,80,100),
        #     lot1 = c(118,58,42,35,27,25,21,19,18),
        #     lot2 = c(69,35,26,21,18,16,13,12,12))
        # clot1 <- glm(lot1 ~ log(u), data = clotting, family = Gamma(link='identity'))
        # summary(clot1)
        # gamma.shape(clot1)
        alpha_param = np.log(14.956340)

        # Checking SNM parameters
        npt.assert_allclose(mestr.theta[2],
                            alpha_param,
                            atol=1e-4)

    def test_glm_gamma_inverse(self):
        # Example data comes from R's MASS library
        d = pd.DataFrame()
        d['X'] = np.log([5, 10, 15, 20, 30, 40, 60, 80, 100])
        d['Y'] = [118, 58, 42, 35, 27, 25, 21, 19, 18]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X']], y=d['Y'],
                          distribution='gamma', link='inverse')

        mestr = MEstimator(psi, init=[0.1, 0.1, 2.])
        mestr.estimate(solver='lm', maxiter=5000)

        # Log-Gamma with statsmodels (only includes scale parameters)
        fam = sm.families.Gamma(sm.families.links.inverse_power())
        glm = sm.GLM(d['Y'], d[['I', 'X']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta[0:2],
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance[0:2, 0:2],
                            np.asarray(glm.cov_params()),
                            rtol=1e-5)

        # Previously solved log Gamma shape parameter using MASS
        # library(MASS)
        # clotting <- data.frame(
        #     u = c(5,10,15,20,30,40,60,80,100),
        #     lot1 = c(118,58,42,35,27,25,21,19,18),
        #     lot2 = c(69,35,26,21,18,16,13,12,12))
        # clot1 <- glm(lot1 ~ log(u), data = clotting, family = Gamma(link='inverse'))
        # summary(clot1)
        # gamma.shape(clot1)
        alpha_param = np.log(538.1315)

        # Checking SNM parameters
        npt.assert_allclose(mestr.theta[2],
                            alpha_param,
                            atol=1e-4)

    def test_glm_gamma_log_weighted(self):
        # Example data comes from R's MASS library
        d = pd.DataFrame()
        d['X'] = np.log([5, 10, 15, 20, 30, 40, 60, 80, 100])
        d['Y'] = [118, 58, 42, 35, 27, 25, 21, 19, 18]
        d['I'] = 1
        d['w'] = [1, 2, 1, 3, 1, 6, 1, 2, 3]

        # Using the weights
        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X']], y=d['Y'],
                          distribution='gamma', link='log',
                          weights=d['w'])

        westr = MEstimator(psi, init=[0., 0., 0.])
        westr.estimate(solver='lm', maxiter=5000)

        # Using an expanded data frame instead
        ld = pd.DataFrame(np.repeat(d.values,
                                    d['w'],
                                    axis=0),
                          columns=d.columns)

        def psi(theta):
            return ee_glm(theta, X=ld[['I', 'X']], y=ld['Y'],
                          distribution='gamma', link='log')

        uestr = MEstimator(psi, init=[0., 0., 0.])
        uestr.estimate(solver='lm', maxiter=5000)

        # Checking mean estimate
        npt.assert_allclose(westr.theta,
                            uestr.theta,
                            atol=1e-6)

    def test_glm_invnormal_identity(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='inverse_normal', link='identity')

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.InverseGaussian(sm.families.links.identity())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_invnormal_log(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='inverse_normal', link='log',
                          hyperparameter=1.5)

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.InverseGaussian(sm.families.links.log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_log(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=1.5)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Tweedie(sm.families.links.log(), var_power=1.5)
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_error(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=-1)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        with pytest.raises(ValueError, match="non-negative"):
            mestr.estimate(solver='lm', maxiter=5000)

    def test_glm_tweedie_lessthan1(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [0, 0, 0, 0, 0, 15, 15, 25, 25, 45, 0, 0, 0, 0, 15, 15, 15, 25, 25, 35]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=0.5)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Tweedie(sm.families.links.log(), var_power=0.5)
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_identity(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=1.5)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Tweedie(sm.families.links.log(), var_power=1.5)
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_poisson(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        # Using the weights
        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=1)

        testr = MEstimator(psi, init=[0., 0., 0.])
        testr.estimate(solver='lm', maxiter=5000)

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='poisson', link='log')

        sestr = MEstimator(psi, init=[0., 0., 0.])
        sestr.estimate(solver='lm', maxiter=5000)

        # Checking mean estimate
        npt.assert_allclose(testr.theta,
                            sestr.theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(testr.variance,
                            sestr.variance,
                            rtol=1e-6)

    def test_glm_tweedie_gamma(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        # Using the weights
        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=2)

        testr = MEstimator(psi, init=[0., 0., 0.])
        testr.estimate(solver='lm', maxiter=5000)

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='gamma', link='log')

        sestr = MEstimator(psi, init=[0., 0., 0., 0.])
        sestr.estimate(solver='lm', maxiter=5000)

        # Checking mean estimate
        npt.assert_allclose(testr.theta,
                            sestr.theta[0:3],
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(testr.variance,
                            sestr.variance[0:3, 0:3],
                            atol=1e-6)

    def test_glm_tweedie_invnormal(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
        d['I'] = 1

        # Using the weights
        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=3)

        testr = MEstimator(psi, init=[0., 0., 0.])
        testr.estimate(solver='lm', maxiter=5000)

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='inverse_normal', link='log')

        sestr = MEstimator(psi, init=[0., 0., 0.])
        sestr.estimate(solver='lm', maxiter=5000)

        # Checking mean estimate
        npt.assert_allclose(testr.theta,
                            sestr.theta,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(testr.variance,
                            sestr.variance,
                            rtol=1e-6)

    def test_glm_nb_log(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [0, 0, 0, 0, 0, 15, 15, 25, 25, 45, 0, 0, 0, 0, 15, 15, 15, 25, 25, 35]
        d['I'] = 1

        # M-estimation negative binomial
        def psi(theta):
            ef_glm = ee_glm(theta[:-1], X=d[['I', 'X', 'Z']], y=d['Y'],
                            distribution='nb', link='log')
            # Transform with exponentiation to compare back to statsmodels
            ef_ta = np.ones(d.shape[0])*(np.exp(theta[-2]) - theta[-1])
            return np.vstack([ef_glm, ef_ta])

        mestr = MEstimator(psi, init=[0., 0., 0., -2., 1.])
        mestr.estimate(solver='lm', maxiter=5000)

        # Negative Binomial using statsmodels
        nb = sm.NegativeBinomial(d['Y'], d[['I', 'X', 'Z']],
                                 loglike_method='nb2').fit(cov_type="HC1",
                                                           tol=1e-10,
                                                           method='newton')

        # Checking mean estimate
        npt.assert_allclose(list(mestr.theta[:3]) + [mestr.theta[-1], ],
                            np.asarray(nb.params),
                            atol=1e-6)

        # Checking variance estimates
        cov_mat = np.asarray(nb.cov_params())
        npt.assert_allclose(mestr.variance[:3, :3],
                            np.asarray(cov_mat[:3, :3]),
                            atol=1e-6)

        # Checking covariance for nuisance parameters (since there is an extra transform)
        npt.assert_allclose(mestr.variance[-1, -1],
                            np.asarray(cov_mat[-1, -1]),
                            atol=1e-4)
        npt.assert_allclose(mestr.variance[0, -1],
                            np.asarray(cov_mat[0, -1]),
                            atol=1e-4)
        npt.assert_allclose(mestr.variance[1, -1],
                            np.asarray(cov_mat[1, -1]),
                            atol=1e-4)
        npt.assert_allclose(mestr.variance[2, -1],
                            np.asarray(cov_mat[2, -1]),
                            atol=1e-4)

    def test_glm_nb_identity(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [0, 0, 0, 0, 0, 15, 15, 25, 25, 45, 0, 0, 0, 0, 15, 15, 15, 25, 25, 35]
        d['I'] = 1

        # M-estimation negative binomial
        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='nb', link='identity')

        mestr = MEstimator(psi, init=[20., 0., 0., 1.])
        mestr.estimate(solver='lm', maxiter=5000)

        # Negative Binomial using statsmodels
        fam = sm.families.NegativeBinomial(sm.families.links.identity(), alpha=np.exp(mestr.theta[-1]))
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta[:3],
                            np.asarray(glm.params),
                            atol=1e-5)

        # Checking variance estimates
        # Can't check variances since statsmodels NB ignores the uncertainty in alpha

    def test_elasticnet(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 1 + 2*data['X'] - 1*data['Z'] + np.random.normal(size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])

        def psi_ridge(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='linear',
                                       penalty=[0., 5., 2.])

        ridge = MEstimator(psi_ridge, init=[0., 0., 0.])
        ridge.estimate(solver='lm')

        def psi_elastic(theta):
            return ee_elasticnet_regression(theta, X=Xvals, y=yvals,
                                            model='linear',
                                            penalty=[0., 5., 2.], ratio=0)

        elastic = MEstimator(psi_elastic, init=[0., 0., 0.])
        elastic.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(ridge.theta,
                            elastic.theta,
                            atol=1e-5)

    def test_elasticnet_offset(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 1 + 2*data['X'] - 1*data['Z'] + np.random.normal(size=n)
        data['C'] = 1
        data['off'] = np.random.uniform(0.4, 0.6, size=n)
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])

        def psi_ridge(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='linear',
                                       penalty=[0., 5., 2.],
                                       offset=data['off'])

        ridge = MEstimator(psi_ridge, init=[0., 0., 0.])
        ridge.estimate(solver='lm')

        def psi_elastic(theta):
            return ee_elasticnet_regression(theta, X=Xvals, y=yvals,
                                            model='linear',
                                            penalty=[0., 5., 2.], ratio=0,
                                            offset=data['off'])

        elastic = MEstimator(psi_elastic, init=[0., 0., 0.])
        elastic.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(ridge.theta,
                            elastic.theta,
                            atol=1e-5)

    def test_robust_linear(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 1 + 2*data['X'] - 1*data['Z'] + np.random.normal(size=n)
        data['C'] = 1
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])

        def psi_linear(theta):
            return ee_regression(theta, X=Xvals, y=yvals,
                                 model='linear')

        linear = MEstimator(psi_linear, init=[0., 0., 0.])
        linear.estimate(solver='lm')

        def psi_robust(theta):
            return ee_robust_regression(theta, X=Xvals, y=yvals,
                                        model='linear', k=20)

        robust = MEstimator(psi_robust, init=[0., 0., 0.])
        robust.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(linear.theta,
                            robust.theta,
                            atol=1e-6)

    def test_robust_linear_offset(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 1 + 2*data['X'] - 1*data['Z'] + np.random.normal(size=n)
        data['C'] = 1
        data['off'] = np.random.uniform(0.4, 0.6, size=n)
        Xvals = np.asarray(data[['C', 'X', 'Z']])
        yvals = np.asarray(data['Y'])

        def psi_linear(theta):
            return ee_regression(theta, X=Xvals, y=yvals,
                                 model='linear', offset=data['off'])

        linear = MEstimator(psi_linear, init=[0., 0., 0.])
        linear.estimate(solver='lm')

        def psi_robust(theta):
            return ee_robust_regression(theta, X=Xvals, y=yvals,
                                        model='linear', k=20, offset=data['off'])

        robust = MEstimator(psi_robust, init=[0., 0., 0.])
        robust.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(linear.theta,
                            robust.theta,
                            atol=1e-6)


class TestEstimatingEquationsSurvival:

    @pytest.fixture
    def surv_data(self):
        np.random.seed(1123211)
        n = 200
        d = pd.DataFrame()
        d['C'] = np.random.weibull(a=1, size=n)
        d['C'] = np.where(d['C'] > 5, 5, d['C'])
        d['T'] = 0.8 * np.random.weibull(a=0.75, size=n)
        d['delta'] = np.where(d['T'] < d['C'], 1, 0)
        d['t'] = np.where(d['delta'] == 1, d['T'], d['C'])
        return np.asarray(d['t']), np.asarray(d['delta'])

    @pytest.fixture
    def data(self):
        np.random.seed(131313131)
        n = 200
        d = pd.DataFrame()
        d['X'] = np.random.binomial(n=1, p=0.5, size=n)
        d['W'] = np.random.binomial(n=1, p=0.5, size=n)
        d['T'] = (1 / 1.25 + 1 / np.exp(0.5) * d['X']) * np.random.weibull(a=0.75, size=n)
        d['C'] = np.random.weibull(a=1, size=n)
        d['C'] = np.where(d['C'] > 10, 10, d['C'])

        d['delta'] = np.where(d['T'] < d['C'], 1, 0)
        d['t'] = np.where(d['delta'] == 1, d['T'], d['C'])
        d['weight'] = np.random.uniform(1, 5, size=n)
        return d

    def test_exponential_model(self, surv_data):
        """Tests exponential model estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            return ee_exponential_model(theta=theta[0],
                                        t=times, delta=events)

        mestimator = MEstimator(psi, init=[1.])
        mestimator.estimate(solver="lm")

        exf = ExponentialFitter()
        exf.fit(times, events)
        results = np.asarray(exf.summary[['coef', 'se(coef)', 'coef lower 95%', 'coef upper 95%']])

        # Checking mean estimate
        npt.assert_allclose(1 / mestimator.theta[0],
                            np.asarray(results[0, 0]),
                            atol=1e-5)

        # No robust variance for lifeline's ExponentialFitter, so not checking against
        # Checking variance estimates
        # npt.assert_allclose(np.sqrt(np.diag(mestimator.variance)),
        #                     np.asarray(results[0, 1]),
        #                     atol=1e-6)

        # Checking confidence interval estimates
        # npt.assert_allclose(mestimator.confidence_intervals(),
        #                     np.asarray(results[0, 2:]),
        #                     atol=1e-5)

    def test_exponential_survival(self, surv_data):
        """Tests exponential measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_exp = ee_exponential_model(theta=theta[0],
                                          t=times, delta=events)
            ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                             times=[0.5, 1, 2, 3], n=times.shape[0],
                                             measure="survival")
            return np.vstack((ee_exp, ee_surv))

        mestimator = MEstimator(psi, init=[1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        exf = ExponentialFitter()
        exf.fit(times, events)
        results = np.asarray(exf.survival_function_at_times(times=[0.5, 1, 2, 3]))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[1:],
                            results,
                            atol=1e-5)

    def test_exponential_risk(self, surv_data):
        """Tests exponential measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_exp = ee_exponential_model(theta=theta[0],
                                          t=times, delta=events)
            ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                             times=[0.5, 1, 2, 3], n=times.shape[0],
                                             measure="risk")
            return np.vstack((ee_exp, ee_surv))

        mestimator = MEstimator(psi, init=[1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        exf = ExponentialFitter()
        exf.fit(times, events)
        results = exf.cumulative_density_at_times(times=[0.5, 1, 2, 3])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[1:],
                            results,
                            atol=1e-5)

    def test_exponential_hazard(self, surv_data):
        """Tests exponential measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_exp = ee_exponential_model(theta=theta[0],
                                          t=times, delta=events)
            ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                             times=[0.5, 1, 2, 3], n=times.shape[0],
                                             measure="hazard")
            return np.vstack((ee_exp, ee_surv))

        mestimator = MEstimator(psi, init=[1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        exf = ExponentialFitter()
        exf.fit(times, events)
        results = np.asarray(exf.summary['coef'])[0]

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[1:],
                            [1/results]*4,
                            atol=1e-5)

    def test_exponential_cumulative_hazard(self, surv_data):
        """Tests exponential measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_exp = ee_exponential_model(theta=theta[0],
                                          t=times, delta=events)
            ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                             times=[0.5, 1, 2, 3], n=times.shape[0],
                                             measure="cumulative_hazard")
            return np.vstack((ee_exp, ee_surv))

        mestimator = MEstimator(psi, init=[1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        exf = ExponentialFitter()
        exf.fit(times, events)
        results = exf.cumulative_hazard_at_times(times=[0.5, 1, 2, 3])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[1:],
                            results,
                            atol=1e-5)

    def test_exponential_density(self, surv_data):
        """Tests exponential measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_exp = ee_exponential_model(theta=theta[0],
                                          t=times, delta=events)
            ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                             times=[0.5, 1, 2, 3], n=times.shape[0],
                                             measure="density")
            return np.vstack((ee_exp, ee_surv))

        mestimator = MEstimator(psi, init=[1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        # NOTICE: lifelines fails here (some problem with the derivative), so skipping comparison
        #   the density measure is still covered by the Weibull density prediction (so not a testing coverage problem)
        # exf = ExponentialFitter()
        # exf.fit(times, events)
        # results = exf.density_at_times(times=[0.5, 1, 2, 3])
        #
        # # Checking mean estimate
        # npt.assert_allclose(mestimator.theta[1:],
        #                     results,
        #                     atol=1e-5)
        pass

    def test_weibull_model(self, surv_data):
        """Tests Weibull model estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            return ee_weibull_model(theta=theta,
                                    t=times, delta=events)

        mestimator = MEstimator(psi, init=[1., 1.])
        mestimator.estimate(solver="lm")

        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.summary[['coef', 'se(coef)', 'coef lower 95%', 'coef upper 95%']])

        # Checking mean estimate
        npt.assert_allclose([(1 / mestimator.theta[0])**(1/mestimator.theta[1]), mestimator.theta[1]],
                            np.asarray(results[:, 0]),
                            atol=1e-4)

        # No robust variance for lifeline's WeibullFitter, so not checking against
        # Checking variance estimates
        # npt.assert_allclose(np.sqrt(np.diag(mestimator.variance)),
        #                     np.asarray(results[0, 1]),
        #                     atol=1e-6)

        # Checking confidence interval estimates
        # npt.assert_allclose(mestimator.confidence_intervals(),
        #                     np.asarray(results[0, 2:]),
        #                     atol=1e-5)

    def test_weibull_survival(self, surv_data):
        """Tests Weibull measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_wbl = ee_weibull_model(theta=theta[0:2],
                                      t=times, delta=events)
            ee_surv = ee_weibull_measure(theta[2:], scale=theta[0], shape=theta[1],
                                         times=[0.5, 1, 2, 3], n=times.shape[0],
                                         measure="survival")
            return np.vstack((ee_wbl, ee_surv))

        mestimator = MEstimator(psi, init=[1., 1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.survival_function_at_times(times=[0.5, 1, 2, 3]))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[2:],
                            results,
                            atol=1e-5)

    def test_weibull_risk(self, surv_data):
        """Tests Weibull measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_wbl = ee_weibull_model(theta=theta[0:2],
                                      t=times, delta=events)
            ee_surv = ee_weibull_measure(theta[2:], scale=theta[0], shape=theta[1],
                                         times=[0.5, 1, 2, 3], n=times.shape[0],
                                         measure="risk")
            return np.vstack((ee_wbl, ee_surv))

        mestimator = MEstimator(psi, init=[1., 1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.cumulative_density_at_times(times=[0.5, 1, 2, 3]))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[2:],
                            results,
                            atol=1e-5)

    def test_weibull_hazard(self, surv_data):
        """Tests Weibull measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_wbl = ee_weibull_model(theta=theta[0:2],
                                      t=times, delta=events)
            ee_surv = ee_weibull_measure(theta[2:], scale=theta[0], shape=theta[1],
                                         times=[0.5, 1, 2, 3], n=times.shape[0],
                                         measure="hazard")
            return np.vstack((ee_wbl, ee_surv))

        mestimator = MEstimator(psi, init=[1., 1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.hazard_at_times(times=[0.5, 1, 2, 3]))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[2:],
                            results,
                            atol=1e-4)

    def test_weibull_cumulative_hazard(self, surv_data):
        """Tests Weibull measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_wbl = ee_weibull_model(theta=theta[0:2],
                                      t=times, delta=events)
            ee_surv = ee_weibull_measure(theta[2:], scale=theta[0], shape=theta[1],
                                         times=[0.5, 1, 2, 3], n=times.shape[0],
                                         measure="cumulative_hazard")
            return np.vstack((ee_wbl, ee_surv))

        mestimator = MEstimator(psi, init=[1., 1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.cumulative_hazard_at_times(times=[0.5, 1, 2, 3]))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[2:],
                            results,
                            atol=1e-4)

    def test_weibull_density(self, surv_data):
        """Tests Weibull measures estimating equation to lifelines.
        """
        times, events = surv_data

        def psi(theta):
            ee_wbl = ee_weibull_model(theta=theta[0:2],
                                      t=times, delta=events)
            ee_surv = ee_weibull_measure(theta[2:], scale=theta[0], shape=theta[1],
                                         times=[0.5, 1, 2, 3], n=times.shape[0],
                                         measure="density")
            return np.vstack((ee_wbl, ee_surv))

        mestimator = MEstimator(psi, init=[1., 1., 0.5, 0.5, 0.5, 0.5])
        mestimator.estimate(solver="lm")

        wbf = WeibullFitter()
        wbf.fit(times, events)
        results = np.asarray(wbf.density_at_times(times=[0.5, 1, 2, 3]))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[2:],
                            results,
                            atol=1e-5)

    def test_weibull_aft(self, data):
        """Tests Weibull AFT estimating equation to lifelines.
        """
        def psi(theta):
            return ee_aft_weibull(theta=theta,
                                  t=data['t'], delta=data['delta'], X=data[['X', 'W']])

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[-.5, 0.7, 0., 0.])
        mestimator.estimate(solver="lm")

        # Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data[['X', 'W', 't', 'delta']], 't', 'delta',
                 ancillary=False, robust=True)
        results = np.asarray(waft.summary[['coef', 'se(coef)', 'coef lower 95%', 'coef upper 95%']])
        results = results[[2, 1, 0, 3], :]

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(results[:, 0]),
                            atol=1e-5)

        # Checking variance estimates
        npt.assert_allclose(np.sqrt(np.diag(mestimator.variance)),
                            np.asarray(results[:, 1]),
                            atol=1e-6)

        # Checking confidence interval estimates
        npt.assert_allclose(mestimator.confidence_intervals(),
                            np.asarray(results[:, 2:]),
                            atol=1e-5)

    def test_weighted_weibull_aft(self, data):
        """Tests weighted Weibull AFT estimating equation to lifelines.
        """
        def psi(theta):
            return ee_aft_weibull(theta=theta, weights=data['weight'],
                                  t=data['t'], delta=data['delta'], X=data[['X', 'W']])

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[0., 0., 0., 0.])
        mestimator.estimate(solver="lm")

        # Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data[['X', 'W', 't', 'delta', 'weight']], 't', 'delta',
                 weights_col='weight', ancillary=False, robust=True)
        results = np.asarray(waft.summary[['coef', 'se(coef)', 'coef lower 95%', 'coef upper 95%']])
        results = results[[2, 1, 0, 3], :]

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(results[:, 0]),
                            atol=1e-5)

        # No variance check, since lifelines uses a different estimator

    def test_weibull_aft_survival(self, data):
        """Tests predicted survival at several time points for Weibull AFT estimating equation to lifelines.
        """
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data['t'], delta=data['delta'], X=data[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='survival',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[-.5, 0.7, 0., -.2, ] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_survival_function(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_risk(self, data):
        """Tests predicted risk at several time points for Weibull AFT estimating equation to lifelines.
        """
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data['t'], delta=data['delta'], X=data[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='risk',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[-.5, 0.7, 0., -.2, ] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = 1 - waft.predict_survival_function(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_density(self, data):
        """Tests predicted density at several time points for Weibull AFT estimating equation to lifelines.
        """
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data['t'], delta=data['delta'], X=data[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='density',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[-.5, 0.7, 0., -.2, ] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = (waft.predict_survival_function(dta.iloc[0], times=times_to_eval)
                 * waft.predict_hazard(dta.iloc[0], times=times_to_eval))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_hazard(self, data):
        """Tests predicted hazard at several time points for Weibull AFT estimating equation to lifelines.
        """
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data['t'], delta=data['delta'], X=data[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='hazard',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[-.5, 0.7, 0., -.2, ] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_hazard(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_cumulative_hazard(self, data):
        """Tests predicted cumulative hazard at several time points for Weibull AFT estimating equation to lifelines.
        """
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data['t'], delta=data['delta'], X=data[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='cumulative_hazard',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[-.5, 0.7, 0., -.2, ] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_cumulative_hazard(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-4)


class TestEstimatingEquationsDoseResponse:

    def test_4pl(self):
        """Test the 4 parameter log-logistic model using Inderjit et al. (2002)

        Compares against R's drc library:

        library(drc)
        library(sandwich)
        library(lmtest)

        data(ryegrass)
        rgll4 = drm(rootl ~ conc, data=ryegrass, fct=LL.4())
        coeftest(rgll4, vcov=sandwich)
        """
        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_4p_logistic(theta=theta, X=dose_data, y=resp_data)

        # Optimization procedure
        mestimator = MEstimator(psi, init=[0, 2, 1, 10])
        mestimator.estimate(solver='lm')

        # R optimization from Ritz et al.
        comparison_theta = np.asarray([0.48141, 3.05795, 2.98222, 7.79296])
        comparison_var = np.asarray([0.12779, 0.26741, 0.47438, 0.15311])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            comparison_theta,
                            atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(mestimator.variance)**0.5,
                            comparison_var,
                            atol=1e-4)

    def test_3pl(self):
        """Test the 3 parameter log-logistic model using Inderjit et al. (2002)

        Compares against R's drc library:

        library(drc)
        library(sandwich)
        library(lmtest)

        data(ryegrass)
        rgll3 = drm(rootl ~ conc, data=ryegrass, fct=LL.3())
        coeftest(rgll3, vcov=sandwich)
        """
        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_3p_logistic(theta=theta, X=dose_data, y=resp_data,
                                  lower=0)

        # Optimization procedure
        mestimator = MEstimator(psi, init=[2, 1, 10])
        mestimator.estimate(solver='lm')

        # R optimization from Ritz et al.
        comparison_theta = np.asarray([3.26336, 2.47033, 7.85543])
        comparison_var = np.asarray([0.26572, 0.29238, 0.15397])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            comparison_theta,
                            atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(mestimator.variance)**0.5,
                            comparison_var,
                            atol=1e-5)

    def test_2pl(self):
        """Test the 2 parameter log-logistic model using Inderjit et al. (2002)

        Compares against R's drc library:

        library(drc)
        library(sandwich)
        library(lmtest)

        data(ryegrass)
        rgll2 = drm(rootl ~ conc, data=ryegrass, fct=LL.2(upper=8))
        coeftest(rgll2, vcov=sandwich)
        """
        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_2p_logistic(theta=theta, X=dose_data, y=resp_data,
                                  lower=0, upper=8)

        # Optimization procedure
        mestimator = MEstimator(psi, init=[2, 1])
        mestimator.estimate(solver='lm')

        # R optimization from Ritz et al.
        comparison_theta = np.asarray([3.19946, 2.38220])
        comparison_var = np.asarray([0.24290, 0.27937])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            comparison_theta,
                            atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(mestimator.variance)**0.5,
                            comparison_var,
                            atol=1e-5)

    def test_3pl_ed_delta(self):
        """Test the ED(alpha) calculation with the 3 parameter log-logistic model using Inderjit et al. (2002)

        Compares against R's drc library:

        library(drc)
        library(sandwich)

        data(ryegrass)
        rgll3 = drm(rootl ~ conc, data=ryegrass, fct=LL.3())
        ED(rgll3, c(5, 10, 50), interval='delta', vcov=sandwich)
        """
        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            lower_limit = 0
            pl3 = ee_3p_logistic(theta=theta, X=dose_data, y=resp_data,
                                 lower=lower_limit)
            ed05 = ee_effective_dose_delta(theta[3], y=resp_data, delta=0.05,
                                           steepness=theta[0], ed50=theta[1],
                                           lower=lower_limit, upper=theta[2])
            ed10 = ee_effective_dose_delta(theta[4], y=resp_data, delta=0.10,
                                           steepness=theta[0], ed50=theta[1],
                                           lower=lower_limit, upper=theta[2])
            ed50 = ee_effective_dose_delta(theta[5], y=resp_data, delta=0.50,
                                           steepness=theta[0], ed50=theta[1],
                                           lower=lower_limit, upper=theta[2])
            return np.vstack((pl3,
                              ed05,
                              ed10,
                              ed50))

        # Optimization procedure
        mestimator = MEstimator(psi, init=[2, 1, 10, 1, 1, 2])
        mestimator.estimate(solver='lm')

        # R optimization from Ritz et al.
        comparison_theta = np.asarray([0.99088, 1.34086, 3.26336])
        comparison_var = np.asarray([0.12397, 0.13134, 0.26572])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[-3:],
                            comparison_theta,
                            atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(mestimator.variance)[-3:]**0.5,
                            comparison_var,
                            atol=1e-5)


class TestEstimatingEquationsMeasurement:

    @pytest.fixture
    def cole2023_data(self):
        # Data from Cole et al. (2023)
        d1 = pd.DataFrame()
        d1['Y_star'] = [0, 1]
        d1['Y'] = [np.nan, np.nan]
        d1['n'] = [950-680, 680]
        d1 = pd.DataFrame(np.repeat(d1.values, d1['n'], axis=0), columns=d1.columns)
        d1 = d1[['Y', 'Y_star']].copy()
        d1['S'] = 1

        d0 = pd.DataFrame()
        d0['Y_star'] = [0, 1, 0, 1]
        d0['Y'] = [0, 0, 1, 1]
        d0['n'] = [71, 18, 38, 203]
        d0 = pd.DataFrame(np.repeat(d0.values, d0['n'], axis=0), columns=d0.columns)
        d0 = d0[['Y', 'Y_star']].copy()
        d0['S'] = 0

        d = pd.concat([d0, d1], axis=0)
        d['C'] = 1
        return d

    @pytest.fixture
    def cole_roses_data(self):
        # Data from Cole et al.'s rejoinder (202x)
        d = pd.DataFrame()
        d['Y_star'] = [0, 1, 0, 1] + [np.nan, np.nan] + [1, 0, 1, 0]
        d['Y'] = [np.nan, np.nan, np.nan, np.nan] + [np.nan, np.nan] + [1, 1, 0, 0]
        d['W'] = [0, 0, 1, 1] + [0, 1] + [np.nan, np.nan, np.nan, np.nan]
        d['n'] = [266, 67, 400, 267] + [333, 167] + [180, 20, 60, 240]
        d['S'] = [1, 1, 1, 1] + [2, 2] + [3, 3, 3, 3]
        d = pd.DataFrame(np.repeat(d.values, d['n'], axis=0), columns=d.columns)
        d['C'] = 1
        return d

    def test_rogan_gladen(self, cole2023_data):
        # Replicate Cole et al. 2023 Rogan-Gladen example as a test

        def psi(theta):
            return ee_rogan_gladen(theta,
                                   y=cole2023_data['Y'],
                                   y_star=cole2023_data['Y_star'],
                                   r=cole2023_data['S'])

        estr = MEstimator(psi, init=[0.5, 0.5, .75, .75])
        estr.estimate(solver='lm')

        reference_theta = [0.80231396, 0.71578947, 0.84232365, 0.79775281]
        reference_covar = [[ 1.56147627e-03, 3.34556962e-04, -6.90781941e-04, 5.59893648e-04],
                           [ 3.34556962e-04, 2.14142010e-04,  0.,             0.],
                           [-6.90781941e-04, 0.,              5.51097661e-04, 0.],
                           [ 5.59893648e-04, 0.,              0.,             1.81284545e-03]]

        # Checking mean estimate
        npt.assert_allclose(estr.theta, reference_theta,
                            atol=1e-6)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, reference_covar,
                            atol=1e-6)

    def test_rogan_gladen_weights(self, cole_roses_data):
        # Replicate Cole et al. 202x Rejoinder example as a test
        dr = cole_roses_data
        y_no_nan = np.asarray(dr['Y'].fillna(-9))
        ystar_no_nan = np.asarray(dr['Y_star'].fillna(-9))
        w_no_nan = np.asarray(dr[['C', 'W']].fillna(-9))
        s1 = np.where(dr['S'] == 1, 1, 0)
        s2 = np.where(dr['S'] == 2, 1, 0)
        s3 = np.where(dr['S'] == 3, 1, 0)

        def psi(theta):
            param = theta[:4]
            beta = theta[4:]

            # Inverse odds weights model
            ee_sm = ee_regression(beta, X=w_no_nan, y=s2,
                                  model='logistic')
            ee_sm = ee_sm * (1 - s3)
            pi_s = inverse_logit(np.dot(w_no_nan, beta))
            iosw = s1 * pi_s / (1-pi_s) + s3

            # Rogan-Gladen
            ee_rg = ee_rogan_gladen(param,
                                    y=y_no_nan,
                                    y_star=ystar_no_nan,
                                    r=s1,
                                    weights=iosw)
            ee_rg = ee_rg * (1 - s2)
            return np.vstack([ee_rg, ee_sm])

        estr = MEstimator(psi, init=[0.5, 0.5, .75, .75, 0., 0.])
        estr.estimate(solver='lm')

        reference_theta = [0.0967144999, 0.267700150, 0.9, 0.8]
        reference_covar = [[ 1.45149504e-03,  3.88376666e-04, -6.21735965e-05,  6.88217424e-04],
                           [ 3.88376666e-04,  2.71863653e-04,  0.,              0.],
                           [-6.21735965e-05,  0.,              4.49999988e-04,  0.],
                           [ 6.88217424e-04,  0.,              0.,              5.33333316e-04]]

        # Checking mean estimate
        npt.assert_allclose(estr.theta[0:4], reference_theta,
                            atol=1e-6)

        # Checking variance estimate
        npt.assert_allclose(estr.variance[0:4, 0:4], reference_covar,
                            atol=1e-6)


class TestEstimatingEquationsCausal:

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

    @pytest.fixture
    def causal_miss_data(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  ] + [np.nan, ]*15
        d['I'] = 1
        return d

    def test_gformula(self, causal_data):
        d1 = causal_data.copy()
        d1['A'] = 1
        d0 = causal_data.copy()
        d0['A'] = 0

        # M-estimation
        def psi(theta):
            return ee_gformula(theta,
                               y=causal_data['Y'],
                               X=causal_data[['C', 'A', 'W']],
                               X1=d1[['C', 'A', 'W']],
                               X0=d0[['C', 'A', 'W']])

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

    def test_gcomp_bad_dimensions_error(self, causal_data):
        d1 = causal_data.copy()
        d1['A'] = 1
        d0 = causal_data.copy()
        d0['A'] = 0

        # M-estimation
        def psi(theta):
            return ee_gformula(theta,
                               y=causal_data['Y'],
                               X=causal_data[['C', 'A', 'W']],
                               X1=d1[['C', 'W']])

        mestimator = MEstimator(psi, init=[0.5, 0., 0., 0.])
        with pytest.raises(ValueError, match="The dimensions of X and X1"):
            mestimator.estimate(solver='lm')

        def psi(theta):
            return ee_gformula(theta,
                               y=causal_data['Y'],
                               X=causal_data[['C', 'A', 'W']],
                               X1=d1[['C', 'A', 'W']],
                               X0=d0[['C', 'A']])

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0.])
        with pytest.raises(ValueError, match="The dimensions of X and X0"):
            mestimator.estimate(solver='lm')

    def test_ipw(self, causal_data):
        # M-estimation
        def psi(theta):
            return ee_ipw(theta,
                          y=causal_data['Y'],
                          A=causal_data['A'],
                          W=causal_data[['C', 'W']])

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

    def test_ipw_truncate(self, causal_data):
        # M-estimation
        def psi(theta):
            return ee_ipw(theta,
                          y=causal_data['Y'],
                          A=causal_data['A'],
                          W=causal_data[['C', 'W']],
                          truncate=(0.1, 0.5))

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(causal_data['A'], causal_data[['C', 'W']],
                     family=sm.families.Binomial()).fit()
        pi = glm.predict()
        pi = np.clip(pi, 0.1, 0.5)
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

    def test_ipw_truncate_error(self, causal_data):
        # M-estimation
        def psi(theta):
            return ee_ipw(theta,
                          y=causal_data['Y'],
                          A=causal_data['A'],
                          W=causal_data[['C', 'W']],
                          truncate=(0.99, 0.01))

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0.])
        with pytest.raises(ValueError, match="truncate values"):
            mestimator.estimate()

    def test_ipw_weights(self, causal_miss_data):
        # Setting up data
        d = causal_miss_data
        W = d[['I', 'V', 'W']]
        X = d[['I', 'A']]
        a = d['A']
        y = d['Y']
        r = np.where(d['Y'].isna(), 0, 1)

        # M-estimation
        def psi(theta):
            # Separating parameters out
            alpha = theta[:3 + W.shape[1]]  # MSM & PS
            gamma = theta[3 + W.shape[1]:]  # Missing score

            # Estimating equation for IPMW
            ee_ms = ee_regression(theta=gamma,
                                  X=X, y=r,
                                  model='logistic')
            pi_m = inverse_logit(np.dot(X, gamma))
            ipmw = r / pi_m

            # Estimating equations for MSM and PS
            ee_msm = ee_ipw(theta=alpha, y=y, A=a, W=W,
                            weights=ipmw)
            ee_msm = np.nan_to_num(ee_msm, copy=False, nan=0.)
            return np.vstack([ee_msm, ee_ms])

        init_vals = [0., 3., 3., ] + [0., 0., 0.] + [0., 0.]
        mestr = MEstimator(psi, init=init_vals)
        mestr.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm_ps = sm.GLM(a, W, family=sm.families.Binomial()).fit()
        pi_a = glm_ps.predict()
        glm_ms = sm.GLM(r, X, family=sm.families.Binomial()).fit()
        pi_m = glm_ms.predict()
        ipmw = r / pi_m

        ya1 = d['A'] * d['Y'] / pi_a * ipmw
        ya0 = (1-d['A']) * d['Y'] / (1-pi_a) * ipmw

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestr.theta[3:6],
                            np.asarray(glm_ps.params),
                            atol=1e-6)
        npt.assert_allclose(mestr.theta[6:],
                            np.asarray(glm_ms.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestr.theta[0],
                            np.mean(ya1) - np.mean(ya0),
                            atol=1e-6)
        npt.assert_allclose(mestr.theta[1],
                            np.mean(ya1),
                            atol=1e-6)
        npt.assert_allclose(mestr.theta[2],
                            np.mean(ya0),
                            atol=1e-6)

    def test_ipw_msm(self, causal_data):
        # M-estimation
        def psi(theta):
            return ee_ipw_msm(theta,
                              y=causal_data['Y'],
                              A=causal_data['A'],
                              W=causal_data[['C', 'W']],
                              V=causal_data[['C', 'A']],
                              link='logit', distribution='binomial')

        mestimator = MEstimator(psi, init=[0., 0., 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(causal_data['A'], causal_data[['C', 'W']],
                     family=sm.families.Binomial()).fit()
        pi = glm.predict()
        ipw = np.where(causal_data['A'] == 1, 1/pi, 1/(1-pi))
        msm = sm.GEE(causal_data['Y'], causal_data[['C', 'A']],
                     family=sm.families.Binomial(), weights=ipw,
                     groups=causal_data.index).fit()

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestimator.theta[2:],
                            np.asarray(glm.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestimator.theta[0:2],
                            msm.params,
                            atol=1e-6)

    def test_ipw_msm_truncate(self, causal_data):
        # M-estimation
        def psi(theta):
            return ee_ipw_msm(theta,
                              y=causal_data['Y'], A=causal_data['A'],
                              W=causal_data[['C', 'W']], V=causal_data[['C', 'A']],
                              link='logit', distribution='binomial', truncate=(0.1, 0.5))

        mestimator = MEstimator(psi, init=[0., 0., 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(causal_data['A'], causal_data[['C', 'W']],
                     family=sm.families.Binomial()).fit()
        pi = glm.predict()
        pi = np.clip(pi, 0.1, 0.5)
        ipw = np.where(causal_data['A'] == 1, 1/pi, 1/(1-pi))
        msm = sm.GEE(causal_data['Y'], causal_data[['C', 'A']],
                     family=sm.families.Binomial(), weights=ipw,
                     groups=causal_data.index).fit()

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestimator.theta[2:],
                            np.asarray(glm.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestimator.theta[0:2],
                            msm.params,
                            atol=1e-6)

    def test_ipw_msm_weights(self, causal_miss_data):
        # Setting up data
        d = causal_miss_data.copy()
        W = d[['I', 'V', 'W']]
        X = d[['I', 'A']]
        msm = d[['I', 'A']]
        a = d['A']
        y = d['Y']
        r = np.where(d['Y'].isna(), 0, 1)

        # M-estimation
        def psi(theta):
            # Separating parameters out
            alpha = theta[:2 + W.shape[1]]  # MSM & PS
            gamma = theta[2 + W.shape[1]:]  # Missing score

            # Estimating equation for IPMW
            ee_ms = ee_regression(theta=gamma,
                                  X=X, y=r,
                                  model='logistic')
            pi_m = inverse_logit(np.dot(X, gamma))
            ipmw = r / pi_m

            # Estimating equations for MSM and PS
            ee_msm = ee_ipw_msm(alpha,
                                y=y, A=a, W=W, V=msm,
                                link='log', distribution='poisson',
                                weights=ipmw)
            ee_msm = np.nan_to_num(ee_msm, copy=False, nan=0.)
            return np.vstack([ee_msm, ee_ms])

        init_vals = [0., 0., ] + [0., 0., 0.] + [0., 0.]
        mestr = MEstimator(psi, init=init_vals)
        mestr.estimate(solver='lm', maxiter=5000)

        # By-hand IPW estimator with statsmodels
        glm_ps = sm.GLM(a, W, family=sm.families.Binomial()).fit()
        pi_a = glm_ps.predict()
        iptw = np.where(a == 1, 1/pi_a, 1/(1-pi_a))
        glm_ms = sm.GLM(r, X, family=sm.families.Binomial()).fit()
        pi_m = glm_ms.predict()
        ipmw = r / pi_m
        d['ipw'] = ipmw*iptw

        dcc = d.dropna()
        msm = sm.GEE(dcc['Y'], dcc[['I', 'A']], weights=dcc['ipw'],
                     family=sm.families.Poisson(), groups=dcc.index).fit()

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestr.theta[2:5],
                            np.asarray(glm_ps.params),
                            atol=1e-6)
        npt.assert_allclose(mestr.theta[5:],
                            np.asarray(glm_ms.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestr.theta[0:2],
                            msm.params,
                            atol=1e-6)

    def test_aipw(self, causal_data):
        d1 = causal_data.copy()
        d1['A'] = 1
        d0 = causal_data.copy()
        d0['A'] = 0

        # M-estimation
        def psi_builtin_regression(theta):
            return ee_aipw(theta,
                           y=causal_data['Y'],
                           A=causal_data['A'],
                           W=causal_data[['C', 'W']],
                           X=causal_data[['C', 'A', 'W']],
                           X1=d1[['C', 'A', 'W']],
                           X0=d0[['C', 'A', 'W']])

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
        npt.assert_allclose(mestimator.theta[3:5],
                            np.asarray(pi_m.params),
                            atol=1e-6)
        npt.assert_allclose(mestimator.theta[5:],
                            np.asarray(y_m.params),
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

    def test_gestimaton_linear(self):
        # Generating a simple test data set to compare with
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5,
                  3, 5, 1, 5, 2, 5, 2, 1, 4, 2, 3, 4, 2, 5, 5]
        d['I'] = 1

        # M-estimator
        def psi(theta):
            return ee_gestimation_snmm(theta=theta,
                                       y=d['Y'], A=d['A'],
                                       W=d[['I', 'V', 'W']],
                                       V=d[['I', 'V']],
                                       model='linear')

        mestr = MEstimator(psi, init=[0., ] * 5)
        mestr.estimate(solver='lm')

        # Previously solved SNM using zEpid
        snm_params = [0.499264398938, -0.400700107829]

        # Previously solved variance
        snm_var = [[0.5309405889911, -0.5497523512113, -0.08318452996742, 0.02706302088976,  0.02666983404626],
                   [-0.5497523512113,  0.9114476962859,  0.20188395780370, -8.543908429690e-04, -0.08753380928183],
                   [-0.08318452996742, 0.2018839578037,  0.84188596592780, -0.2000575145713, -0.2959629818969],
                   [0.02706302088976, -8.543908429690e-04, -0.2000575145713, 0.4010462043495, -0.03045175600959],
                   [0.02666983404626, -0.08753380928183, -0.2959629818969, -0.03045175600959,  0.1505645314902]]

        # Checking SNM parameters
        npt.assert_allclose(mestr.theta[0:2],
                            snm_params,
                            atol=1e-7)

        # Checking variance
        npt.assert_allclose(mestr.variance,
                            snm_var,
                            atol=1e-4)

    def test_gestimation_linear_weighted(self, causal_miss_data):
        # Setting up data for estimation equation
        d = causal_miss_data
        W = d[['I', 'V', 'W']]
        X = d[['I', 'A']]
        snm = d[['I', 'V']]
        a = d['A']
        y = d['Y']
        r = np.where(d['Y'].isna(), 0, 1)

        # M-estimator
        def psi(theta):
            # Dividing parameters into corresponding estimation equations
            alpha = theta[:snm.shape[1] + W.shape[1]]  # SNM and PS
            gamma = theta[snm.shape[1] + W.shape[1]:]  # Missing scores

            # Estimating equation for IPMW
            ee_ms = ee_regression(theta=gamma,  # Missing score
                                  X=X, y=r,  # ... observed data
                                  model='logistic')  # ... logit model
            pi_m = inverse_logit(np.dot(X, gamma))  # Predicted prob
            ipmw = r / pi_m  # Missing weights

            # Estimating equations for PS
            ee_snm = ee_gestimation_snmm(theta=alpha,
                                         y=y, A=a,
                                         W=W, V=snm,
                                         model='linear',
                                         weights=ipmw)
            # Setting rows with missing Y's as zero (no contribution)
            ee_snm = np.nan_to_num(ee_snm, copy=False, nan=0.)

            return np.vstack([ee_snm, ee_ms])

        init_values = [0., 0.] + [0., ]*3 + [0., ]*2
        mestr = MEstimator(psi, init=init_values)
        mestr.estimate(solver='lm')

        # Comparison using zEpid
        # from zepid.causal.snm import GEstimationSNM
        # snm = GEstimationSNM(d, exposure='A', outcome='Y')
        # snm.exposure_model("V + W")
        # snm.missing_model("A", stabilized=False)
        # snm.structural_nested_model("A + A:V")
        # snm.fit()
        snm_params = [0.478757884654, -0.366956950389,
                      2.406293643425, -0.750506882472, -0.816256976888,
                      1.252762968495, -0.878069519054]

        # Previously solved variance
        snm_var = [[0.833658518465, -0.863011204849],
                   [-0.863011204849, 1.397401367295]]

        # Checking SNM parameters
        npt.assert_allclose(mestr.theta,
                            snm_params,
                            atol=1e-7)

        # Checking variance
        npt.assert_allclose(mestr.variance[:2, :2],
                            snm_var,
                            atol=1e-4)

    def test_robins_sensitivity_mean(self):
        d = pd.DataFrame()
        d['I'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        d['X'] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        d['Y'] = [7, 2, 5, np.nan, 1, 4, 8, np.nan, 1, np.nan]
        d['delta'] = np.where(d['Y'].isna(), 0, 1)

        def q_function(y_vals, alpha):
            y_no_miss = np.where(np.isnan(y_vals), 0, y_vals)
            return alpha * y_no_miss

        ####
        # Checking with alpha=0.5
        def psi(theta):
            return ee_mean_sensitivity_analysis(theta=theta,
                                                y=d['Y'], delta=d['delta'], X=d[['I', 'X']],
                                                q_eval=q_function(d['Y'], alpha=0.5),
                                                H_function=inverse_logit)

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm')

        # Checking point estimates
        npt.assert_allclose(mestr.theta,
                            [4.42581577, -3.72866034, 3.74204493],
                            atol=1e-6)

        # Checking variance estimates
        var_closed = [[0.77498021, 0.00455995, -0.04176666],
                      [0.00455995, 1.0032351, -0.94005101],
                      [-0.04176666, -0.94005101, 2.26235294]]
        npt.assert_allclose(mestr.variance,
                            var_closed,
                            atol=1e-6)

        ####
        # Checking with alpha=-0.5
        def psi(theta):
            return ee_mean_sensitivity_analysis(theta=theta,
                                                y=d['Y'], delta=d['delta'], X=d[['I', 'X']],
                                                q_eval=q_function(d['Y'], alpha=-0.5),
                                                H_function=inverse_logit)

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm')

        # Checking point estimates
        npt.assert_allclose(mestr.theta,
                            [4.876513, 3.579188, 0.140411],
                            atol=1e-6)

        # Checking variance estimates
        var_closed = [[0.75110762, 0.05973157, -0.11159881],
                      [0.05973157, 0.88333613, -0.36712181],
                      [-0.11159881, -0.36712181, 0.52362185]]
        npt.assert_allclose(mestr.variance,
                            var_closed,
                            atol=1e-6)

        ####
        # Checking complete-case analysis
        def psi(theta):
            return ee_mean_sensitivity_analysis(theta=theta,
                                                y=d['Y'], delta=d['delta'], X=d[['I', ]],
                                                q_eval=q_function(d['Y'], alpha=0.),
                                                H_function=inverse_logit)

        mestr = MEstimator(psi, init=[0., 0.])
        mestr.estimate(solver='lm')

        # Checking point estimates
        npt.assert_allclose(mestr.theta[0],
                            np.nanmean(d['Y']),
                            atol=1e-6)
