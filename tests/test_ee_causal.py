####################################################################################################################
# Tests for built-in estimating equations -- causal
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm

from delicatessen import MEstimator, GMMEstimator
from delicatessen.estimating_equations import (ee_regression,
                                               ee_gformula, ee_ipw, ee_ipw_msm, ee_aipw, ee_gestimation_snmm,
                                               ee_iv_causal, ee_2sls,
                                               ee_mean_sensitivity_analysis)
from delicatessen.utilities import inverse_logit


class TestEstimatingEquationsGMethods:

    @pytest.fixture
    def data_causal_b(self):
        d = pd.DataFrame()
        d['W'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        d['V'] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        d['Y'] = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
                  1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        d['I'] = 1
        return d

    @pytest.fixture
    def data_causal_c(self):
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

    @pytest.fixture
    def data_causal_m(self):
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

    def test_gformula(self, data_causal_b):
        d = data_causal_b
        d1 = d.copy()
        d1['A'] = 1
        d0 = d.copy()
        d0['A'] = 0

        # M-estimation
        def psi(theta):
            return ee_gformula(theta,
                               y=d['Y'], X=d[['I', 'A', 'W']],
                               X1=d1[['I', 'A', 'W']],
                               X0=d0[['I', 'A', 'W']])

        estr = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0.])
        estr.estimate(solver='lm')

        # By-hand g-formula with statsmodels
        glm = sm.GLM(d['Y'], d[['I', 'A', 'W']], family=sm.families.Binomial()).fit()
        cd = d[['I', 'A', 'W']].copy()
        cd['A'] = 1
        ya1 = glm.predict(cd)
        cd['A'] = 0
        ya0 = glm.predict(cd)

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(estr.theta[3:],
                            np.asarray(glm.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(estr.theta[0],
                            np.mean(ya1) - np.mean(ya0),
                            atol=1e-6)
        npt.assert_allclose(estr.theta[1],
                            np.mean(ya1),
                            atol=1e-6)
        npt.assert_allclose(estr.theta[2],
                            np.mean(ya0),
                            atol=1e-6)

    def test_gcomp_bad_dimensions_error(self, data_causal_b):
        d = data_causal_b
        d1 = d.copy()
        d1['A'] = 1
        d0 = d.copy()
        d0['A'] = 0

        # M-estimation
        def psi(theta):
            return ee_gformula(theta,
                               y=d['Y'], X=d[['I', 'A', 'W']],
                               X1=d1[['I', 'W']])

        mestimator = MEstimator(psi, init=[0.5, 0., 0., 0.])
        with pytest.raises(ValueError, match="The dimensions of X and X1"):
            mestimator.estimate(solver='lm')

        def psi(theta):
            return ee_gformula(theta,
                               y=d['Y'], X=d[['I', 'A', 'W']],
                               X1=d1[['I', 'A', 'W']],
                               X0=d0[['I', 'A']])

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0.])
        with pytest.raises(ValueError, match="The dimensions of X and X0"):
            mestimator.estimate(solver='lm')

    def test_ipw(self, data_causal_b):
        d = data_causal_b

        def psi(theta):
            return ee_ipw(theta, y=d['Y'], A=d['A'], W=d[['I', 'W']])

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(d['A'], d[['I', 'W']], family=sm.families.Binomial()).fit()
        pi = glm.predict()
        ya1 = d['A'] * d['Y'] / pi
        ya0 = (1-d['A']) * d['Y'] / (1-pi)

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

    def test_ipw_truncate(self, data_causal_b):
        d = data_causal_b

        def psi(theta):
            return ee_ipw(theta, y=d['Y'], A=d['A'], W=d[['I', 'W']],
                          truncate=(0.1, 0.5))

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(d['A'], d[['I', 'W']], family=sm.families.Binomial()).fit()
        pi = glm.predict()
        pi = np.clip(pi, 0.1, 0.5)
        ya1 = d['A'] * d['Y'] / pi
        ya0 = (1-d['A']) * d['Y'] / (1-pi)

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

    def test_ipw_truncate_error(self, data_causal_b):
        d = data_causal_b

        def psi(theta):
            return ee_ipw(theta, y=d['Y'], A=d['A'], W=d[['I', 'W']],
                          truncate=(0.99, 0.01))

        mestimator = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0.])
        with pytest.raises(ValueError, match="truncate values"):
            mestimator.estimate()

    def test_ipw_weights(self, data_causal_m):
        # Setting up data
        d = data_causal_m
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

    def test_ipw_msm(self, data_causal_b):
        d = data_causal_b

        def psi(theta):
            return ee_ipw_msm(theta, y=d['Y'], A=d['A'],
                              W=d[['I', 'W']], V=d[['I', 'A']],
                              link='logit', distribution='binomial')

        mestimator = MEstimator(psi, init=[0., 0., 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(d['A'], d[['I', 'W']],
                     family=sm.families.Binomial()).fit()
        pi = glm.predict()
        ipw = np.where(d['A'] == 1, 1/pi, 1/(1-pi))
        msm = sm.GEE(d['Y'], d[['I', 'A']], family=sm.families.Binomial(), weights=ipw,
                     groups=d.index).fit()

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestimator.theta[2:],
                            np.asarray(glm.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestimator.theta[0:2],
                            msm.params,
                            atol=1e-6)

    def test_ipw_msm_truncate(self, data_causal_b):
        d = data_causal_b

        def psi(theta):
            return ee_ipw_msm(theta, y=d['Y'], A=d['A'], W=d[['I', 'W']], V=d[['I', 'A']],
                              link='logit', distribution='binomial', truncate=(0.1, 0.5))

        mestimator = MEstimator(psi, init=[0., 0., 0., 0.])
        mestimator.estimate(solver='lm')

        # By-hand IPW estimator with statsmodels
        glm = sm.GLM(d['A'], d[['I', 'W']], family=sm.families.Binomial()).fit()
        pi = glm.predict()
        pi = np.clip(pi, 0.1, 0.5)
        ipw = np.where(d['A'] == 1, 1/pi, 1/(1-pi))
        msm = sm.GEE(d['Y'], d[['I', 'A']], family=sm.families.Binomial(), weights=ipw,
                     groups=d.index).fit()

        # Checking logistic coefficients (nuisance model estimates)
        npt.assert_allclose(mestimator.theta[2:],
                            np.asarray(glm.params),
                            atol=1e-6)
        # Checking mean estimates
        npt.assert_allclose(mestimator.theta[0:2],
                            msm.params,
                            atol=1e-6)

    def test_ipw_msm_weights(self, data_causal_m):
        # Setting up data
        d = data_causal_m.copy()
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

    def test_aipw(self, data_causal_b):
        d = data_causal_b
        d1 = d.copy()
        d1['A'] = 1
        d0 = d.copy()
        d0['A'] = 0

        # M-estimation
        def psi_builtin_regression(theta):
            return ee_aipw(theta, y=d['Y'], A=d['A'],
                           W=d[['I', 'W']], X=d[['I', 'A', 'W']],
                           X1=d1[['I', 'A', 'W']], X0=d0[['I', 'A', 'W']])

        mestimator = MEstimator(psi_builtin_regression, init=[0., 0.5, 0.5,   # Parameters of interest
                                                              0., 0., 0.,     # Outcome nuisance model
                                                              0., 0.])        # Treatment nuisance model
        mestimator.estimate(solver='lm', tolerance=1e-12)

        # By-hand IPW estimator with statsmodels
        pi_m = sm.GLM(d['A'], d[['I', 'W']],
                      family=sm.families.Binomial()).fit()
        y_m = sm.GLM(d['Y'], d[['I', 'A', 'W']],
                     family=sm.families.Binomial()).fit()
        # Predicting coefficients
        pi = pi_m.predict()
        cd = d[['I', 'A', 'W']].copy()
        cd['A'] = 1
        ya1 = y_m.predict(cd)
        cd['A'] = 0
        ya0 = y_m.predict(cd)
        # AIPW estimator
        ya1_star = d['Y'] * d['A'] / pi - ya1 * (d['A'] - pi) / pi
        ya0_star = d['Y'] * (1-d['A']) / (1-pi) - ya0 * (pi - d['A']) / (1-pi)
        # AIPW variance estimator!
        var_ate = np.nanvar((ya1_star - ya0_star) - np.mean(ya1_star - ya0_star), ddof=1) / d.shape[0]
        var_r1 = np.nanvar(ya1_star - np.mean(ya1_star), ddof=1) / d.shape[0]
        var_r0 = np.nanvar(ya0_star - np.mean(ya0_star), ddof=1) / d.shape[0]

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

    def test_gestimaton_linear(self, data_causal_c):
        d = data_causal_c
        d['VW'] = d['V']*d['W']

        # M-estimator
        def psi(theta):
            return ee_gestimation_snmm(theta=theta, y=d['Y'], A=d['A'],
                                       W=d[['I', 'V', 'W']], V=d[['I', 'V']],
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

    def test_gestimation_linear_weighted(self, data_causal_m):
        # Setting up data for estimation equation
        d = data_causal_m
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
            ee_snm = ee_gestimation_snmm(theta=alpha, y=y, A=a,
                                         W=W, V=snm,
                                         model='linear', weights=ipmw)
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

    def test_gestimation_linear_efficient(self, data_causal_c):
        d = data_causal_c

        # M-estimator
        def psi(theta):
            return ee_gestimation_snmm(theta=theta, y=d['Y'], A=d['A'],
                                       W=d[['I', 'V', 'W']], V=d[['I', 'W']],
                                       X=d[['I', 'V', 'W']], model='linear')

        mestr = MEstimator(psi, init=[0., ] * 8)
        mestr.estimate(solver='lm')

        # Comparing nuisance models
        npt.assert_allclose(mestr.theta[2:5], [2.06553139, -0.65396442, -0.80506948],
                            atol=1e-7)
        npt.assert_allclose(mestr.theta[5:], [0.37758292, -0.03818214, 1.28057925],
                            atol=1e-7)

        smm_params = [2.70724664, -1.17655853]
        smm_var = [[1.10055729, -0.63263541],
                   [-0.63263541, 0.41548127]]

        # Checking SNM parameters
        npt.assert_allclose(mestr.theta[:2], smm_params,
                            atol=1e-7)

        # Checking variance
        npt.assert_allclose(mestr.variance[:2, :2], smm_var,
                            atol=1e-4)

    def test_gestimation_poisson(self, data_causal_c):
        d = data_causal_c
        ps_nuisance = [2.06553139, -0.65396442, -0.80506948]

        # Inefficient g-estimator
        def psi(theta):
            return ee_gestimation_snmm(theta=theta, y=d['Y'], A=d['A'],
                                       W=d[['I', 'V', 'W']], V=d[['I', 'W']],
                                       model='poisson')

        mestr = MEstimator(psi, init=[0., ] * 5)
        mestr.estimate(solver='lm')

        # Comparing nuisance models
        npt.assert_allclose(mestr.theta[2:], ps_nuisance, atol=1e-7)

        # Comparing SMM parameters
        smm_params = [0.72207801, -0.28975432]
        smm_var = [[0.15162332, -0.06732043],
                   [-0.06732043,  0.03567282]]
        npt.assert_allclose(mestr.theta[:2], smm_params,
                            atol=1e-7)
        npt.assert_allclose(mestr.variance[:2, :2], smm_var,
                            atol=1e-4)

        # Efficient g-estimator
        def psi(theta):
            return ee_gestimation_snmm(theta=theta, y=d['Y'], A=d['A'],
                                       W=d[['I', 'V', 'W']], V=d[['I', 'W']],
                                       X=d[['I', 'V', 'W']], model='poisson')

        mestr = MEstimator(psi, init=[0., ] * 8)
        mestr.estimate(solver='lm')

        # Comparing nuisance models
        npt.assert_allclose(mestr.theta[2:5], ps_nuisance, atol=1e-7)
        npt.assert_allclose(mestr.theta[5:], [1.60740877e-01, -1.80166331e-03, 4.31572318e-01], atol=1e-7)

        # Comparing SMM parameters
        smm_params = [1.04245099, -0.441452325]
        smm_var = [[0.10928566, -0.06168025],
                   [-0.06168025, 0.04047553]]
        npt.assert_allclose(mestr.theta[:2], smm_params,
                            atol=1e-7)
        npt.assert_allclose(mestr.variance[:2, :2], smm_var,
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


class TestEstimatingEquationsIV:

    @pytest.fixture
    def data_b(self):
        d = pd.DataFrame()
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['A'] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        d['Y'] = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        d['w'] = [1, 1, 2, 3, 1, 4, 1, 5, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 3]
        return d

    @pytest.fixture
    def data_2sls(self):
        # This data was generated using the following code:
        # np.random.seed(42)
        # n = 30
        # d = pd.DataFrame()
        # d['X'] = np.random.normal(size=n)
        # d['Z'] = np.random.normal(size=n)
        # d['U'] = np.random.normal(size=n)
        # d['A'] = 0.5*d['Z'] + 0.5*d['X'] + 0.4*d['U'] + np.random.normal(size=n)
        # d['Y'] = 2*d['X'] - 3*d['A'] + d['U'] + np.random.normal(size=n)
        # d['C'] = 1
        # d = np.round(d, decimals=2)
        d = pd.DataFrame()
        d['X'] = [0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91, -1.72,
                  -0.56, -1.01, 0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42, -0.54, 0.11, -1.15, 0.38, -0.6, -0.29]
        d['Z'] = [-0.6, 1.85, -0.01, -1.06, 0.82, -1.22, 0.21, -1.96, -1.33, 0.2, 0.74, 0.17, -0.12, -0.3, -1.48, -0.72,
                  -0.46, 1.06, 0.34, -1.76, 0.32, -0.39, -0.68, 0.61, 1.03, 0.93, -0.84, -0.31, 0.33, 0.98]
        d['A'] = [-0.15, 1.75, -0.83, -0.57, 0.23, -1.65, 1.16, 0.07, -0.75, -0.12, -1.13, 0.05, -0.29, -1.28, -2.81,
                  0.09, 1.18, 0.74, 0.01, -2.46, -1.11, -0.19, 0.35, 1.85, -0.27, 0.62, -0.66, -1., 0.8, 1.3]
        d['Y'] = [1.75, -6.63, 4.07, 2.17, 0.25, 8.02, -1.39, 1.77, 1.77, 0.31, 1.28, 0.53, 0.27, 2.06, 1.44, 0.97,
                  -6.27, -2.21, -0.95, 1.33, 6.27, 1.78, -1.03, -8.73, -0.82, -1.36, -0.63, 2.77, -3.6, -3.67]
        d['C'] = 1
        d['w'] = [1, ]*29 + [2, ]
        return d

    @pytest.fixture
    def data_multi_iv(self):
        d = pd.DataFrame()
        d['X'] = [0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47, 0.54, -0.46, -0.47, 0.24, -1.91, -1.72,
                  -0.56, -1.01, 0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42, -0.54, 0.11, -1.15, 0.38, -0.6, -0.29]
        d['Z1'] = [-0.6, 1.85, -0.01, -1.06, 0.82, -1.22, 0.21, -1.96, -1.33, 0.2, 0.74, 0.17, -0.12, -0.3, -1.48,
                   -0.72, -0.46, 1.06, 0.34, -1.76, 0.32, -0.39, -0.68, 0.61, 1.03, 0.93, -0.84, -0.31, 0.33, 0.98]
        d['Z2'] = [-0.48, -0.19, -1.11, -1.2, 0.81, 1.36, -0.07, 1., 0.36, -0.65, 0.36, 1.54, -0.04, 1.56, -2.62, 0.82,
                   0.09, -0.3, 0.09, -1.99, -0.22, 0.36, 1.48, -0.52, -0.81, -0.5, 0.92, 0.33, -0.53, 0.51]
        d['A'] = [0.92, 0.39, 1.77, -0.94, 0.48, 0.47, 0.04, -1.36, -0.91, -0.03, -2.09, -0.71, -1.13, -1.42, -1.8,
                  0.82, -0.79, 0.52, 0.61, -2.25, 0.42, 0.88, -2.33, 0.92, 0.67, 1.57, -2.52, -1.85, 1., 0.79]
        d['Y'] = [-1.42, -0.13, -5.4, 5.77, -2.01, -4.06, 5.19, 6.35, 0.59, 1.61, 2.95, 1.56, 4.68, -1.18, 2.75, -2.78,
                  3.06, 1.13, -3.62, 3.1, -1.14, -3.95, 7.11, -2.8, -3.01, -3.37, 5.24, 6.6, -3.33, 0.53]
        d['C'] = 1
        return d

    def test_usual_iv(self, data_b):
        d = data_b
        d['C'] = 1

        def psi(theta):
            return ee_iv_causal(theta, y=d['Y'], A=d['A'], Z=d['Z'])

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate()

        # By-hand
        d1 = d.loc[d['Z'] == 1].copy()
        d0 = d.loc[d['Z'] == 0].copy()
        beta = (np.mean(d1['Y']) - np.mean(d0['Y'])) / (np.mean(d1['A']) - np.mean(d0['A']))

        # 2SLS
        def psi(theta):
            return ee_2sls(theta, y=d['Y'], A=d['A'], Z=d[['Z', ]], W=d[['C', ]])

        estr_2sls = MEstimator(psi, init=[0., 0., 0., 0.])
        estr_2sls.estimate()

        # Checking point estimates versus by-hand
        npt.assert_allclose(estr.theta[0], beta, atol=1e-6)

        # Checking point estimates versus 2SLS
        npt.assert_allclose(estr.theta[0], estr_2sls.theta[0], atol=1e-6)

    def test_usual_iv_weights(self, data_b):
        d = data_b
        d['C'] = 1

        def psi(theta):
            return ee_iv_causal(theta, y=d['Y'], A=d['A'], Z=d['Z'], weights=d['w'])

        estr = GMMEstimator(psi, init=[0., 0.])
        estr.estimate()

        # By-hand
        d1 = d.loc[d['Z'] == 1].copy()
        d0 = d.loc[d['Z'] == 0].copy()
        beta = ((np.average(d1['Y'], weights=d1['w']) - np.average(d0['Y'], weights=d0['w']))
                / (np.average(d1['A'], weights=d1['w']) - np.average(d0['A'], weights=d0['w'])))

        # 2SLS
        def psi(theta):
            return ee_2sls(theta, y=d['Y'], A=d['A'], Z=d[['Z', ]], W=d[['C', ]], weights=d['w'])

        estr_2sls = MEstimator(psi, init=[0., 0., 0., 0.])
        estr_2sls.estimate()

        # Checking point estimates versus by-hand
        npt.assert_allclose(estr.theta[0], beta, atol=1e-6)

        # Checking point estimates versus 2SLS
        npt.assert_allclose(estr.theta[0], estr_2sls.theta[0], atol=1e-6)

    def test_2sls(self, data_2sls):
        d = data_2sls

        def psi(theta):
            return ee_2sls(theta=theta, y=d['Y'], A=d['A'], Z=d[['Z', ]])

        estr = GMMEstimator(psi, init=[0, 0, ])
        estr.estimate()

        # R code for external reference
        # library(ivmodel)
        # library(sandwich)
        # data = data.frame(x=c(0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47,
        #                       0.54, -0.46, -0.47, 0.24, -1.91, -1.72, -0.56, -1.01,
        #                       0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42, -0.54,
        #                       0.11, -1.15, 0.38, -0.6, -0.29),
        #                   z=c(-0.6, 1.85, -0.01, -1.06, 0.82, -1.22, 0.21, -1.96, -1.33,
        #                       0.2, 0.74, 0.17, -0.12, -0.3, -1.48, -0.72, -0.46, 1.06,
        #                       0.34, -1.76, 0.32, -0.39, -0.68, 0.61, 1.03, 0.93, -0.84,
        #                       -0.31, 0.33, 0.98),
        #                   a=c(-0.15, 1.75, -0.83, -0.57, 0.23, -1.65, 1.16, 0.07, -0.75,
        #                       -0.12, -1.13, 0.05, -0.29, -1.28, -2.81, 0.09, 1.18, 0.74,
        #                       0.01, -2.46, -1.11, -0.19, 0.35, 1.85, -0.27, 0.62, -0.66,
        #                       -1., 0.8, 1.3),
        #                   y=c(1.75, -6.63, 4.07, 2.17, 0.25, 8.02, -1.39, 1.77, 1.77,
        #                       0.31, 1.28, 0.53, 0.27, 2.06, 1.44, 0.97, -6.27, -2.21,
        #                       -0.95, 1.33, 6.27, 1.78, -1.03, -8.73, -0.82, -1.36,
        #                       -0.63, 2.77, -3.6, -3.67)
        #                   )
        # Y = data$y
        # D = data$a
        # Z = data$z
        # ivm = ivmodel(Y=Y,D=D,Z=Z,intercept=F)
        # coef(ivm)
        tsls_params = [-2.468675, ]

        # Checking point estimates are close
        npt.assert_allclose(estr.theta[:1], tsls_params,
                            atol=1e-7)

    def test_2sls_intercept(self, data_2sls):
        d = data_2sls

        def psi(theta):
            return ee_2sls(theta=theta, y=d['Y'], A=d['A'], Z=d[['Z', ]], W=d[['C', ]])

        estr = GMMEstimator(psi, init=[0, 0, 0, 0])
        estr.estimate()

        # R code for external reference
        # library(ivmodel)
        # library(sandwich)
        # data = data.frame(x=c(0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47,
        #                       0.54, -0.46, -0.47, 0.24, -1.91, -1.72, -0.56, -1.01,
        #                       0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42, -0.54,
        #                       0.11, -1.15, 0.38, -0.6, -0.29),
        #                   z=c(-0.6, 1.85, -0.01, -1.06, 0.82, -1.22, 0.21, -1.96, -1.33,
        #                       0.2, 0.74, 0.17, -0.12, -0.3, -1.48, -0.72, -0.46, 1.06,
        #                       0.34, -1.76, 0.32, -0.39, -0.68, 0.61, 1.03, 0.93, -0.84,
        #                       -0.31, 0.33, 0.98),
        #                   a=c(-0.15, 1.75, -0.83, -0.57, 0.23, -1.65, 1.16, 0.07, -0.75,
        #                       -0.12, -1.13, 0.05, -0.29, -1.28, -2.81, 0.09, 1.18, 0.74,
        #                       0.01, -2.46, -1.11, -0.19, 0.35, 1.85, -0.27, 0.62, -0.66,
        #                       -1., 0.8, 1.3),
        #                   y=c(1.75, -6.63, 4.07, 2.17, 0.25, 8.02, -1.39, 1.77, 1.77,
        #                       0.31, 1.28, 0.53, 0.27, 2.06, 1.44, 0.97, -6.27, -2.21,
        #                       -0.95, 1.33, 6.27, 1.78, -1.03, -8.73, -0.82, -1.36,
        #                       -0.63, 2.77, -3.6, -3.67)
        #                   )
        # Y = data$y
        # D = data$a
        # Z = data$z
        # ivm = ivmodel(Y=Y,D=D,Z=Z,intercept=T)
        # coef(ivm)
        # ivm$LIML$point.est.other
        tsls_params = [-2.542286, -0.3789797]

        # Checking point estimates are close
        npt.assert_allclose(estr.theta[:2], tsls_params,
                            atol=1e-6)

    def test_2sls_exog(self, data_2sls):
        d = data_2sls

        def psi(theta):
            return ee_2sls(theta=theta, y=d['Y'], A=d['A'], Z=d[['Z', ]], W=d[['C', 'X']])

        estr = GMMEstimator(psi, init=[0, 0, 0, 0, 0, 0])
        estr.estimate()

        # R code for external reference
        # library(ivmodel)
        # library(sandwich)
        # data = data.frame(x=c(0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47,
        #                       0.54, -0.46, -0.47, 0.24, -1.91, -1.72, -0.56, -1.01,
        #                       0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42, -0.54,
        #                       0.11, -1.15, 0.38, -0.6, -0.29),
        #                   z=c(-0.6, 1.85, -0.01, -1.06, 0.82, -1.22, 0.21, -1.96, -1.33,
        #                       0.2, 0.74, 0.17, -0.12, -0.3, -1.48, -0.72, -0.46, 1.06,
        #                       0.34, -1.76, 0.32, -0.39, -0.68, 0.61, 1.03, 0.93, -0.84,
        #                       -0.31, 0.33, 0.98),
        #                   a=c(-0.15, 1.75, -0.83, -0.57, 0.23, -1.65, 1.16, 0.07, -0.75,
        #                       -0.12, -1.13, 0.05, -0.29, -1.28, -2.81, 0.09, 1.18, 0.74,
        #                       0.01, -2.46, -1.11, -0.19, 0.35, 1.85, -0.27, 0.62, -0.66,
        #                       -1., 0.8, 1.3),
        #                   y=c(1.75, -6.63, 4.07, 2.17, 0.25, 8.02, -1.39, 1.77, 1.77,
        #                       0.31, 1.28, 0.53, 0.27, 2.06, 1.44, 0.97, -6.27, -2.21,
        #                       -0.95, 1.33, 6.27, 1.78, -1.03, -8.73, -0.82, -1.36,
        #                       -0.63, 2.77, -3.6, -3.67)
        #                   )
        # data$c = 1
        # Y = data$y
        # D = data$a
        # Z = data$z
        # X = data[,c("c", "x")]
        # ivm = ivmodel(Y=Y,D=D,Z=Z,X=X, intercept=F)
        # coef(ivm)
        # ivm$LIML$point.est.other
        tsls_params = [-2.784835, -0.07432767, 1.848357]

        # Checking point estimates are close
        npt.assert_allclose(estr.theta[:3], tsls_params,
                            atol=1e-6)

    def test_2sls_multi_iv(self, data_multi_iv):
        d = data_multi_iv

        def psi(theta):
            return ee_2sls(theta=theta, y=d['Y'], A=d['A'], Z=d[['Z1', 'Z2']], W=d[['C', 'X']])

        estr = GMMEstimator(psi, init=[0, 0, 0, 0, 0, 0, 0])
        estr.estimate()

        # R code for external reference
        # library(ivmodel)
        # library(sandwich)
        # data = data.frame(x=c(0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47,
        #                       0.54, -0.46, -0.47, 0.24, -1.91, -1.72, -0.56, -1.01,
        #                       0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42, -0.54,
        #                       0.11, -1.15, 0.38, -0.6, -0.29),
        #                   z1=c(-0.6, 1.85, -0.01, -1.06, 0.82, -1.22, 0.21, -1.96,
        #                        -1.33, 0.2, 0.74, 0.17, -0.12, -0.3, -1.48, -0.72, -0.46,
        #                        1.06, 0.34, -1.76, 0.32, -0.39, -0.68, 0.61, 1.03, 0.93,
        #                        -0.84, -0.31, 0.33, 0.98),
        #                   z2=c(-0.48, -0.19, -1.11, -1.2, 0.81, 1.36, -0.07, 1., 0.36,
        #                        -0.65, 0.36, 1.54, -0.04, 1.56, -2.62, 0.82, 0.09, -0.3,
        #                        0.09, -1.99, -0.22, 0.36, 1.48, -0.52, -0.81, -0.5, 0.92,
        #                        0.33, -0.53, 0.51),
        #                   a=c(0.92, 0.39, 1.77, -0.94, 0.48, 0.47, 0.04, -1.36, -0.91,
        #                       -0.03, -2.09, -0.71, -1.13, -1.42, -1.8, 0.82, -0.79,
        #                       0.52, 0.61, -2.25, 0.42, 0.88, -2.33, 0.92, 0.67, 1.57,
        #                       -2.52, -1.85, 1., 0.79),
        #                   y=c(-1.42, -0.13, -5.4, 5.77, -2.01, -4.06, 5.19, 6.35, 0.59,
        #                       1.61, 2.95, 1.56, 4.68, -1.18, 2.75, -2.78, 3.06, 1.13,
        #                       -3.62, 3.1, -1.14, -3.95, 7.11, -2.8, -3.01, -3.37, 5.24,
        #                       6.6, -3.33, 0.53)
        #                   )
        # data$c = 1
        # Y = data$y
        # D = data$a
        # Z = data[,c("z1", "z2")]
        # X = data[,c("c", "x")]
        # ivm = ivmodel(Y=Y,D=D,Z=Z,X=X, intercept=F)
        # coef(ivm)
        # ivm$kClass$point.est.other
        tsls_params = [-2.259155, 0.3856519, 1.658916]

        # Checking point estimates are close
        npt.assert_allclose(estr.theta[:3], tsls_params,
                            atol=1e-6)

    def test_2sls_weights(self, data_2sls):
        d = data_2sls

        def psi(theta):
            return ee_2sls(theta=theta, y=d['Y'], A=d['A'], Z=d[['Z', ]], W=d[['C', 'X']], weights=d['w'])

        estr = GMMEstimator(psi, init=[0, 0, 0, 0, 0, 0])
        estr.estimate()

        # R code for external reference
        # library(ivmodel)
        # library(sandwich)
        # data = data.frame(x=c(0.5, -0.14, 0.65, 1.52, -0.23, -0.23, 1.58, 0.77, -0.47,
        #                       0.54, -0.46, -0.47, 0.24, -1.91, -1.72, -0.56, -1.01,
        #                       0.31, -0.91, -1.41, 1.47, -0.23, 0.07, -1.42, -0.54,
        #                       0.11, -1.15, 0.38, -0.6, -0.29, -0.29),
        #                   z=c(-0.6, 1.85, -0.01, -1.06, 0.82, -1.22, 0.21, -1.96, -1.33,
        #                       0.2, 0.74, 0.17, -0.12, -0.3, -1.48, -0.72, -0.46, 1.06,
        #                       0.34, -1.76, 0.32, -0.39, -0.68, 0.61, 1.03, 0.93, -0.84,
        #                       -0.31, 0.33, 0.98, 0.98),
        #                   a=c(-0.15, 1.75, -0.83, -0.57, 0.23, -1.65, 1.16, 0.07, -0.75,
        #                       -0.12, -1.13, 0.05, -0.29, -1.28, -2.81, 0.09, 1.18, 0.74,
        #                       0.01, -2.46, -1.11, -0.19, 0.35, 1.85, -0.27, 0.62, -0.66,
        #                       -1., 0.8, 1.3, 1.3),
        #                   y=c(1.75, -6.63, 4.07, 2.17, 0.25, 8.02, -1.39, 1.77, 1.77,
        #                       0.31, 1.28, 0.53, 0.27, 2.06, 1.44, 0.97, -6.27, -2.21,
        #                       -0.95, 1.33, 6.27, 1.78, -1.03, -8.73, -0.82, -1.36,
        #                       -0.63, 2.77, -3.6, -3.67, -3.67)
        #                   )
        # data$c = 1
        # Y = data$y
        # D = data$a
        # Z = data$z
        # X = data[,c("c", "x")]
        # ivm = ivmodel(Y=Y,D=D,Z=Z,X=X, intercept=F)
        # coef(ivm)
        # ivm$kClass$point.est.other
        tsls_params = [-2.753454, -0.05424833, 1.838784]

        # Checking point estimates are close
        npt.assert_allclose(estr.theta[:3], tsls_params,
                            atol=1e-6)
