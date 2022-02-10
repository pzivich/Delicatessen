import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic
from sklearn.linear_model import Ridge

from delicatessen import MEstimator
from delicatessen.estimating_equations import (ee_mean, ee_mean_variance, ee_mean_robust,
                                               ee_linear_regression, ee_logistic_regression, ee_ridge_linear_regression,
                                               ee_2p_logistic, ee_3p_logistic, ee_4p_logistic, ee_effective_dose_delta,
                                               ee_gformula, ee_ipw, ee_aipw)
from delicatessen.data import load_inderjit
from delicatessen.utilities import inverse_logit


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


class TestEstimatingEquationsRegression:

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

        # Checking confidence interval estimates
        npt.assert_allclose(mpee.confidence_intervals(),
                            np.asarray(glm.conf_int()),
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
            return ee_ridge_linear_regression(theta, X=Xvals, y=yvals, penalty=0.5, weights=None)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge_skl = Ridge(alpha=.5, fit_intercept=False).fit(X=Xvals, y=yvals)

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge_skl.coef_),
                            atol=1e-6)

        # Penalty of 5.0
        def psi(theta):
            return ee_ridge_linear_regression(theta, X=Xvals, y=yvals, penalty=5.0, weights=None)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge_skl = Ridge(alpha=5.0, fit_intercept=False).fit(X=Xvals, y=yvals)

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge_skl.coef_),
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
            return ee_ridge_linear_regression(theta, X=Xvals, y=yvals, penalty=0.5, weights=weights)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge_skl = Ridge(alpha=.5, fit_intercept=False).fit(X=Xvals, y=yvals, sample_weight=weights)

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge_skl.coef_),
                            atol=1e-6)

        # Penalty of 5.0
        def psi(theta):
            return ee_ridge_linear_regression(theta, X=Xvals, y=yvals, penalty=5.0, weights=weights)

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge_skl = Ridge(alpha=5.0, fit_intercept=False).fit(X=Xvals, y=yvals, sample_weight=weights)

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge_skl.coef_),
                            atol=1e-6)

    def test_logitic(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1

        def psi_builtin_regression(theta):
            return ee_logistic_regression(theta,
                                          X=data[['C', 'X', 'Z']],
                                          y=data['Y'])

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

