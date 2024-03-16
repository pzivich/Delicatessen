####################################################################################################################
# Tests for built-in estimating equations -- regression
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from delicatessen import MEstimator
from delicatessen.estimating_equations import (ee_regression, ee_glm, ee_mlogit,
                                               ee_robust_regression, ee_ridge_regression,
                                               ee_lasso_regression, ee_elasticnet_regression,
                                               ee_additive_regression)
from delicatessen.utilities import additive_design_matrix


@pytest.fixture
def data_c():
    d = pd.DataFrame()
    d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
    d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d['Y'] = [1, 5, 1, 9, -1, 4, 3, 3, 1, -2, 4, -2, 3, 6, 6, 8, 7, 1, -2, 5]
    d['F'] = [1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3]
    d['w'] = [2, 1, 5, 1, 2, 7, 3, 1, 2, 2, 3, 9, 1, 1, 1, 5, 1, 1, 6, 2]
    d['I'] = 1
    return d


@pytest.fixture
def data_b():
    d = pd.DataFrame()
    d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
    d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d['Y'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
    d['F'] = [1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3]
    d['w'] = [2, 1, 5, 1, 2, 7, 3, 1, 2, 2, 3, 9, 1, 1, 1, 5, 1, 1, 6, 2]
    d['I'] = 1
    return d


@pytest.fixture
def data_cp():
    d = pd.DataFrame()
    d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
    d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d['Y'] = [1, 5, 1, 9, 1, 4, 3, 3, 1, 2, 4, 2, 3, 6, 6, 8, 7, 1, 2, 5]
    d['F'] = [1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3]
    d['w'] = [2, 1, 5, 1, 2, 7, 3, 1, 2, 2, 3, 9, 1, 1, 1, 5, 1, 1, 6, 2]
    d['I'] = 1
    return d


class TestEstimatingEquationsRegression:

    @pytest.fixture
    def data_m(self):
        d = pd.DataFrame()
        d['W0'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        d['Y0'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        d['Y1'] = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
        d['Y2'] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        d['offset'] = [.1, -.1, .2, -.2, -.2, .1, .1, .1, -.1, .1, .1, .1, .1, -.1]
        d['C'] = 1
        return d

    @pytest.fixture
    def data_mw(self):
        d = pd.DataFrame()
        d['W0'] = [0, 0, 0, 1, 1, 1]
        d['Y0'] = [1, 0, 0, 1, 0, 0]
        d['Y1'] = [0, 1, 0, 0, 1, 0]
        d['Y2'] = [0, 0, 1, 0, 0, 1]
        d['weight'] = [4, 2, 3, 1, 2, 2]
        d['C'] = 1
        return d

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

    def test_ols(self, data_c):
        """Tests linear regression with the built-in estimating equation.
        """
        d = data_c

        def psi_builtin_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='linear')

        estr = MEstimator(psi_builtin_regression, init=[0., 0., 0.])
        estr.estimate()

        # Statsmodels function equivalent
        glm = smf.glm("Y ~ X + Z", d).fit(cov_type="HC1")

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

    def test_ols_offset(self, data_c):
        """Tests linear regression with the built-in estimating equation and an offset term.
        """
        d = data_c

        def psi_builtin_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 offset=d['F'], model='linear')

        mpee = MEstimator(psi_builtin_regression, init=[0.1, 0.1, 0.1])
        mpee.estimate()

        # Statsmodels function equivalent
        glm = smf.glm("Y ~ X + Z", d, offset=d['F']).fit(cov_type="HC1")

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

    def test_wls(self, data_c):
        """Tests weighted linear regression
        """
        d = data_c

        def psi_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='linear', weights=d['w'])

        mestimator = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, freq_weights=d['w']).fit(cov_type="cluster",
                                                               cov_kwds={"groups": d.index, "use_correction": False})

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

    def test_wls_offset(self, data_c):
        """Tests weighted linear regression with an offset term
        """
        d = data_c

        def psi_regression(theta):
            return ee_regression(theta,
                                 X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='linear', weights=d['w'], offset=d['F'])

        mestimator = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
        mestimator.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, freq_weights=d['w'],
                      offset=d['F']).fit(cov_type="cluster",
                                         cov_kwds={"groups": d.index, "use_correction": False})

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

    def test_logistic(self, data_b):
        d = data_b

        def psi_builtin_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='logistic')

        mpee = MEstimator(psi_builtin_regression, init=[0., 0., 0.])
        mpee.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, family=sm.families.Binomial()).fit(cov_type="HC1")

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

    def test_logistic_offset(self, data_b):
        d = data_b

        def psi_builtin_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='logistic', offset=d['F'])

        mpee = MEstimator(psi_builtin_regression, init=[0., 0., 0.])
        mpee.estimate()

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, family=sm.families.Binomial(), offset=d['F']).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mpee.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mpee.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-5)

        # Checking confidence interval estimates
        npt.assert_allclose(mpee.confidence_intervals(),
                            np.asarray(glm.conf_int()),
                            atol=1e-5)

    def test_weighted_logistic(self, data_b):
        d = data_b

        def psi_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='logistic', weights=d['w'])

        mestimator = MEstimator(psi_regression, init=[0., 2., -1.])
        mestimator.estimate(solver='lm')

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, freq_weights=d['w'],
                      family=sm.families.Binomial()).fit(cov_type="cluster",
                                                         cov_kwds={"groups": d.index,
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

    def test_poisson(self, data_cp):
        d = data_cp

        def psi_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='poisson')

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm')

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, family=sm.families.Poisson()).fit(cov_type="HC1")

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

    def test_poisson_offset(self, data_cp):
        d = data_cp

        def psi_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='poisson', offset=d['F'])

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm')

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, offset=d['F'], family=sm.families.Poisson()).fit(cov_type="HC1")

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

    def test_weighted_poisson(self, data_cp):
        d = data_cp

        def psi_regression(theta):
            return ee_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                 model='poisson', weights=d['w'])

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm')

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X + Z", d, freq_weights=d['w'],
                      family=sm.families.Poisson()).fit(cov_type="cluster",
                                                        cov_kwds={"groups": d.index,
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

    def test_mlogit(self, data_m):
        # Setup data
        d = data_m
        y = np.asarray(d[['Y0', 'Y1', 'Y2']])
        X = np.asarray(d[['C', 'W0']])

        # M-estimator
        def psi(theta):
            return ee_mlogit(theta=theta, X=X, y=y)

        estr = MEstimator(psi, init=[0., 0., 0., 0.])
        estr.estimate(solver='lm')

        # Statsmodels as the reference
        model = sm.MNLogit(y, X)
        fm = model.fit(cov_type="HC1")
        ref_params = list(fm.params[:, 0]) + list(fm.params[:, 1])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            ref_params,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr.variance,
                            np.asarray(fm.cov_params()),
                            atol=1e-5)

    def test_mlogit_weights(self, data_m, data_mw):
        # Setup data
        d = data_m
        y = np.asarray(d[['Y0', 'Y1', 'Y2']])
        X = np.asarray(d[['C', 'W0']])

        # M-estimator for reference
        def psi(theta):
            return ee_mlogit(theta=theta, X=X, y=y)

        estr1 = MEstimator(psi, init=[0., 0., 0., 0.])
        estr1.estimate(solver='lm')
        ref_par = np.asarray(estr1.theta)

        # Setup data
        d = data_mw
        y = np.asarray(d[['Y0', 'Y1', 'Y2']])
        X = np.asarray(d[['C', 'W0']])
        w = np.asarray(d['weight'])

        # M-estimator for reference
        def psi(theta):
            return ee_mlogit(theta=theta, X=X, y=y, weights=w)

        estr2 = MEstimator(psi, init=[0., 0., 0., 0.])
        estr2.estimate(solver='lm')
        wgt_par = np.asarray(estr2.theta)

        # Checking mean estimate
        npt.assert_allclose(wgt_par,
                            ref_par,
                            atol=1e-6)

    def test_mlogit_offset(self, data_m):
        # Setup data
        d = data_m
        y = np.asarray(d[['Y0', 'Y1', 'Y2']])
        X = np.asarray(d[['C', 'W0']])

        # M-estimator
        def psi(theta):
            return ee_mlogit(theta=theta, X=X, y=y,
                             offset=d['offset'])

        estr = MEstimator(psi, init=[0., 0., 0., 0.])
        estr.estimate(solver='lm', tolerance=1e-12)

        # SAS as the reference
        # NOTE: I cannot find a variance reference. SAS doesn't offer it for proc logistic and genmod does not support
        #       unranked multinomial logistic. R's mlogit doesn't allow the specification of the offset. So, I can only
        #       compare coefficients here.
        # SAS code used for the comparison:
        # data dat;
        #     input w y offset weight;
        #     datalines;
        # 0 1   0.1 1
        # 0 1  -0.1 1
        # 0 1   0.2 2
        # 0 1  -0.2 1
        # 0 2  -0.2 3
        # 0 2   0.1 4
        # 0 3   0.1 2
        # 0 3   0.1 1
        # 0 3  -0.1 1
        # 1 1   0.1 1
        # 1 2   0.1 1
        # 1 2   0.1 1
        # 1 3   0.1 1
        # 1 3  -0.1 5
        # ;
        # run;
        # proc logistic data = dat;
        # 	class y (ref = "1") / param = ref;
        # 	model y = w / link = glogit offset = offset gconv=1e-12;
        # 	ods output ParameterEstimates=or;
        # run;
        # proc print data=or label;
        #     format _numeric_ 12.10;
        #     var _numeric_;
        #     title "Odds Ratio Estimates";
        # run;
        ref_params = [-0.6920875177, 1.3271570373, -0.2866227784, 0.9216939568]

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            ref_params,
                            atol=5e-5)


class TestEstimatingEquationsRegressionRobust:

    @pytest.fixture
    def data_r(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [1, 5, 1, 9, -1, 4, 3, 3, 1, -2, 4, -2, 3, 6, 6, 8, 7, 1, -2, 15]
        d['F'] = [1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 3]
        d['w'] = [2, 1, 5, 1, 2, 7, 3, 1, 2, 2, 3, 9, 1, 1, 1, 5, 1, 1, 6, 2]
        d['I'] = 1
        return d

    def test_error_robust(self, data_r):
        d = data_r
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

        def psi(theta):
            return ee_robust_regression(theta, X=Xvals, y=yvals, model='logistic', k=5)

        estr = MEstimator(psi, init=[5, 1, 1])
        with pytest.raises(ValueError, match="only supports linear"):
            estr.estimate(solver='lm')

    def test_robust_eq_to_linear(self, data_r):
        d = data_r
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

        def psi_linear(theta):
            return ee_regression(theta, X=Xvals, y=yvals,
                                 model='linear')

        linear = MEstimator(psi_linear, init=[0., 0., 0.])
        linear.estimate(solver='lm')

        def psi_robust(theta):
            return ee_robust_regression(theta, X=Xvals, y=yvals,
                                        model='linear', k=200)

        robust = MEstimator(psi_robust, init=[0., 0., 0.])
        robust.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(linear.theta,
                            robust.theta,
                            atol=1e-6)

    def test_robust_linear_offset(self, data_r):
        d = data_r
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])
        offset = np.asarray(d['F'])

        def psi_linear(theta):
            return ee_regression(theta, X=Xvals, y=yvals,
                                 model='linear', offset=offset)

        linear = MEstimator(psi_linear, init=[0., 0., 0.])
        linear.estimate(solver='lm')

        def psi_robust(theta):
            return ee_robust_regression(theta, X=Xvals, y=yvals,
                                        model='linear', k=200, offset=offset)

        robust = MEstimator(psi_robust, init=[0., 0., 0.])
        robust.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(linear.theta,
                            robust.theta,
                            atol=1e-6)


class TestEstimatingEquationsRegressionPenalty:

    def test_error_penalized(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear',
                                       penalty=[0.5, 5.], weights=None)

        estr = MEstimator(psi, init=[5, 1, 1])
        with pytest.raises(ValueError, match="The penalty term must"):
            estr.estimate(solver='lm')

    def test_ridge_ols(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

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

    def test_ridge_ols_offset(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

        # Penalty of 0.5
        def psi(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals, model='linear',
                                       penalty=[0., 5., 2.], weights=None, offset=d['F'])

        estr = MEstimator(psi, init=[5, 1, 1])
        estr.estimate(solver='lm')
        ridge = sm.OLS(yvals - d['F'], Xvals,).fit_regularized(L1_wt=0.,
                                                               alpha=np.array([0., 5., 2.]) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

    def test_ridge_wls(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])
        weights = np.asarray(d['w'])

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

    def test_additive_ols(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 15}, None]

        # Testing array of penalty terms
        def psi(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals, model='linear', specifications=spec)

        estr = MEstimator(psi, init=[5, 1, 0, 0, 0])
        estr.estimate(solver='lm')

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        ridge = sm.OLS(yvals, design).fit_regularized(L1_wt=0., alpha=np.array(penalty)/Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

    def test_additive_wls(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])
        weights = np.asarray(d['w'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 15}, None]

        # Testing array of penalty terms
        def psi(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals, model='linear',
                                          specifications=spec, weights=weights)

        estr = MEstimator(psi, init=[5, 1, 0, 0, 0])
        estr.estimate(solver='lm')

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        ridge = sm.WLS(yvals, design, weights=weights).fit_regularized(L1_wt=0.,
                                                                       alpha=np.array(penalty)/Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(estr.theta,
                            np.asarray(ridge.params),
                            atol=1e-6)

    def test_lasso_ols(self, data_c):
        """Tests linear regression with the built-in estimating equation.
        """
        d = data_c

        def psi_builtin_regression(theta):
            return ee_lasso_regression(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                                       model='linear', penalty=1.)

        estr = MEstimator(psi_builtin_regression, init=[0., 0., 0.])
        estr.estimate()

    def test_ridge_logistic(self, data_b):
        d = data_b
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

        def psi_regression(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='logistic', penalty=0.5, weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        f = sm.families.Binomial()
        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=0.5 / Xvals.shape[0], tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-5)

        def psi_regression(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='logistic', penalty=15., weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='hybr', tolerance=1e-12)

        f = sm.families.Binomial()
        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=15. / Xvals.shape[0], tol=1e-12)

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

    def test_additive_logistic(self, data_b):
        d = data_b
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 15}, None]

        def psi_regression(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals,
                                          model='logistic',
                                          specifications=spec)

        mestimator = MEstimator(psi_regression, init=[0., 2., 0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        f = sm.families.Binomial()
        lgt = sm.GLM(yvals, design, family=f).fit_regularized(L1_wt=0., alpha=np.array(penalty)/Xvals.shape[0],
                                                              tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-3)

    def test_ridge_poisson(self, data_cp):
        d = data_cp
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

        def psi_regression(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='poisson', penalty=0.5, weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        f = sm.families.Poisson()
        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=0.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-5)

        def psi_regression(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='poisson', penalty=2.5, weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=2.5 / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-5)

        def psi_regression(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='poisson', penalty=[0., 5., 2.5], weights=None)

        mestimator = MEstimator(psi_regression, init=[0., 0., 0.])
        mestimator.estimate(solver='lm', tolerance=1e-12)

        lgt = sm.GLM(yvals, Xvals, family=f).fit_regularized(L1_wt=0., alpha=np.asarray([0., 5., 2.5]) / Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-5)

    def test_additive_poisson(self, data_cp):
        d = data_cp
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])
        spec = [None, {"knots": [-1, 0, 1], "penalty": 3}, None]

        def psi_regression(theta):
            return ee_additive_regression(theta, X=Xvals, y=yvals,
                                          model='poisson',
                                          specifications=spec)

        mestimator = MEstimator(psi_regression, init=[0., 2., 0., 0., 0.])
        mestimator.estimate(solver='lm')

        design, penalty = additive_design_matrix(X=Xvals, specifications=spec, return_penalty=True)
        f = sm.families.Poisson()
        lgt = sm.GLM(yvals, design, family=f).fit_regularized(L1_wt=0., alpha=np.array(penalty)/Xvals.shape[0])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(lgt.params),
                            atol=1e-5)

    def test_elasticnet(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

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

    def test_elasticnet_offset(self, data_c):
        d = data_c
        Xvals = np.asarray(d[['I', 'X', 'Z']])
        yvals = np.asarray(d['Y'])

        def psi_ridge(theta):
            return ee_ridge_regression(theta, X=Xvals, y=yvals,
                                       model='linear', penalty=[0., 5., 2.], offset=d['F'])

        ridge = MEstimator(psi_ridge, init=[0., 0., 0.])
        ridge.estimate(solver='lm')

        def psi_elastic(theta):
            return ee_elasticnet_regression(theta, X=Xvals, y=yvals,
                                            model='linear', penalty=[0., 5., 2.], ratio=0, offset=d['F'])

        elastic = MEstimator(psi_elastic, init=[0., 0., 0.])
        elastic.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(ridge.theta,
                            elastic.theta,
                            atol=1e-5)


class TestEstimatingEquationsGLM:

    @pytest.fixture
    def data_cpp(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [0, 0, 0, 0, 0, 15, 15, 25, 25, 45, 0, 0, 0, 0, 15, 15, 15, 25, 25, 35]
        d['I'] = 1
        return d

    @pytest.fixture
    def data_g(self):
        # Example data comes from R's MASS library
        d = pd.DataFrame()
        d['X'] = np.log([5, 10, 15, 20, 30, 40, 60, 80, 100])
        d['Y'] = [118, 58, 42, 35, 27, 25, 21, 19, 18]
        d['w'] = [1, 2, 1, 3, 1, 6, 1, 2, 3]
        d['I'] = 1
        return d

    def test_glm_normal_identity(self, data_c):
        d = data_c

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

    def test_glm_normal_log(self, data_c):
        d = data_c

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='normal', link='log')

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Gaussian(sm.families.links.Log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_poisson_identity(self, data_cp):
        d = data_cp

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

    def test_glm_poisson_log(self, data_cp):
        d = data_cp

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='poisson', link='log')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Poisson(sm.families.links.Log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_poisson_sqrt(self, data_cp):
        d = data_cp

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='poisson', link='sqrt')

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Poisson(sm.families.links.Sqrt())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_logit(self, data_b):
        d = data_b

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='logit')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.Logit())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_log(self, data_b):
        d = data_b

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='log')

        mestr = MEstimator(psi, init=[-.9, 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.Log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_identity(self, data_b):
        d = data_b

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

    def test_glm_binomial_probit(self, data_b):
        d = data_b

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='probit')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.Probit())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_cauchy(self, data_b):
        d = data_b

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='cauchy')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.Cauchy())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_cloglog(self, data_b):
        d = data_b

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

    def test_glm_binomial_loglog(self, data_b):
        d = data_b

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='loglog')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.LogLog())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_binomial_logit_offset(self, data_b):
        d = data_b

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='binomial', link='logit',
                          offset=d['F'])

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Binomial(sm.families.links.Logit())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], offset=d['F'], family=fam).fit(cov_type="HC1", tol=1e-12)

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_gamma_log(self, data_g):
        d = data_g

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

    def test_glm_gamma_identity(self, data_g):
        d = data_g

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

    def test_glm_gamma_inverse(self, data_g):
        d = data_g

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

    def test_glm_gamma_log_weighted(self, data_g):
        d = data_g

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

    def test_glm_invnormal_identity(self, data_cp):
        d = data_cp

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

    def test_glm_invnormal_log(self, data_cp):
        d = data_cp

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='inverse_normal', link='log')

        mestr = MEstimator(psi, init=[0., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.InverseGaussian(sm.families.links.Log())
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_log(self, data_cp):
        d = data_cp

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=1.5)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Tweedie(sm.families.links.Log(), var_power=1.5)
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_error(self, data_cp):
        d = data_cp

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=-1)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        with pytest.raises(ValueError, match="non-negative"):
            mestr.estimate(solver='lm', maxiter=5000)

    def test_glm_tweedie_lessthan1(self, data_cpp):
        d = data_cpp

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=0.5)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Tweedie(sm.families.links.Log(), var_power=0.5)
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_identity(self, data_cp):
        d = data_cp

        def psi(theta):
            return ee_glm(theta, X=d[['I', 'X', 'Z']], y=d['Y'],
                          distribution='tweedie', link='log',
                          hyperparameter=1.5)

        mestr = MEstimator(psi, init=[2., 0., 0.])
        mestr.estimate(solver='lm', maxiter=5000)

        fam = sm.families.Tweedie(sm.families.links.Log(), var_power=1.5)
        glm = sm.GLM(d['Y'], d[['I', 'X', 'Z']], family=fam).fit(cov_type="HC1")

        # Checking mean estimate
        npt.assert_allclose(mestr.theta,
                            np.asarray(glm.params),
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.variance,
                            np.asarray(glm.cov_params()),
                            atol=1e-6)

    def test_glm_tweedie_poisson(self, data_cp):
        d = data_cp

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

    def test_glm_tweedie_gamma(self, data_cp):
        d = data_cp

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

    def test_glm_tweedie_invnormal(self, data_cp):
        d = data_cp

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

    def test_glm_nb_log(self, data_cpp):
        d = data_cpp

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

    def test_glm_nb_identity(self, data_cpp):
        d = data_cpp

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
