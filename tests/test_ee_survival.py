####################################################################################################################
# Tests for built-in estimating equations -- survival
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
from lifelines import ExponentialFitter, WeibullFitter, WeibullAFTFitter

from delicatessen import MEstimator
from delicatessen.estimating_equations import (ee_exponential_model, ee_exponential_measure, ee_weibull_model,
                                               ee_weibull_measure, ee_aft_weibull, ee_aft_weibull_measure)


class TestEstimatingEquationsSurvParam:

    @pytest.fixture
    def data_s(self):
        times = np.array([1, 2, 3, 4, 5, 1, 1, 2, 2.5, 3, 4, 5])
        event = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        return times, event

    def test_exponential_model(self, data_s):
        times, events = data_s

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

    def test_exponential_survival(self, data_s):
        times, events = data_s

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

    def test_exponential_risk(self, data_s):
        times, events = data_s

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

    def test_exponential_hazard(self, data_s):
        times, events = data_s

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

    def test_exponential_cumulative_hazard(self, data_s):
        times, events = data_s

        def psi(theta):
            ee_exp = ee_exponential_model(theta=theta[0],
                                          t=times, delta=events)
            ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                             times=[0.5, 1, 2, 3], n=times.shape[0],
                                             measure="cumulative_hazard")
            return np.vstack((ee_exp, ee_surv))

        mestimator = MEstimator(psi, init=[0.2, 0., 0., 0., 0.])
        mestimator.estimate(solver="lm")

        exf = ExponentialFitter()
        exf.fit(times, events)
        results = exf.cumulative_hazard_at_times(times=[0.5, 1, 2, 3])

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[1:],
                            results,
                            atol=1e-5)

    def test_exponential_density(self, data_s):
        times, events = data_s

        def psi(theta):
            ee_exp = ee_exponential_model(theta=theta[0],
                                          t=times, delta=events)
            ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                             times=[0.5, 1, 2, 3], n=times.shape[0],
                                             measure="density")
            return np.vstack((ee_exp, ee_surv))

        mestimator = MEstimator(psi, init=[0.2, 0., 0., 0., 0.])
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

    def test_weibull_model(self, data_s):
        times, events = data_s

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

    def test_weibull_survival(self, data_s):
        times, events = data_s

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

    def test_weibull_risk(self, data_s):
        times, events = data_s

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

    def test_weibull_hazard(self, data_s):
        times, events = data_s

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

    def test_weibull_cumulative_hazard(self, data_s):
        times, events = data_s

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

    def test_weibull_density(self, data_s):
        times, events = data_s

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


class TestEstimatingEquationsAFT:

    @pytest.fixture
    def data_sc(self):
        d = pd.DataFrame()
        d['X'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['W'] = [1, 2, 3, 1, -1, 1, -2, 1, 0, 1, -3, 1, 0, 1, -1, 3, 0, 2, 1, 1, -1, 2, -3, 2, -2, 0, 0, 1, 1, 2, 3, 0, 1, 0, -1]
        d['t'] = [1, 2, 3, 5, 1, 9, 10, 1, 3, 6, 8, 6, 9, 4, 3, 2, 1, 1, 5, 6, 7, 8, 3, 4, 4, 3, 2.1, 3, 5, 3, 2, 1, 3, 2, 10]
        d['delta'] = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        d['weight'] = [1, 2, 1, 0.1, 1, 1, 1, 3, 1, 1, 3, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 2, 1, 4, 1]
        return d

    def test_weibull_aft(self, data_sc):
        def psi(theta):
            return ee_aft_weibull(theta=theta,
                                  t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['X', 'W']])

        # # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[1., 0., 0., 0.6])
        mestimator.estimate(solver="lm")

        # Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data_sc[['X', 'W', 't', 'delta']], 't', 'delta',
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

    def test_weighted_weibull_aft(self, data_sc):
        def psi(theta):
            return ee_aft_weibull(theta=theta, weights=data_sc['weight'],
                                  t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['X', 'W']])

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[1., 0., 0., 0.6])
        mestimator.estimate(solver="lm")

        # Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data_sc[['X', 'W', 't', 'delta', 'weight']], 't', 'delta',
                 weights_col='weight', ancillary=False, robust=True)
        results = np.asarray(waft.summary[['coef', 'se(coef)', 'coef lower 95%', 'coef upper 95%']])
        results = results[[2, 1, 0, 3], :]

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta,
                            np.asarray(results[:, 0]),
                            atol=1e-5)

        # No variance check, since lifelines uses a different estimator

    def test_weibull_aft_survival(self, data_sc):
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data_sc.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='survival',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[1., 0., 0., 0.6] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data_sc[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_survival_function(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_risk(self, data_sc):
        times_to_eval = [1, 1.25, 3, 5]
        dta = data_sc.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='risk',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[1., 0., 0., 0.6] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data_sc[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = 1 - waft.predict_survival_function(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_density(self, data_sc):
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data_sc.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='density',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[1., 0., 0., 0.6] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data_sc[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = (waft.predict_survival_function(dta.iloc[0], times=times_to_eval)
                 * waft.predict_hazard(dta.iloc[0], times=times_to_eval))

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_hazard(self, data_sc):
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data_sc.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='hazard',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[1., 0., 0., 0.6] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data_sc[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_hazard(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-5)

    def test_weibull_aft_cumulative_hazard(self, data_sc):
        # Times to evaluate and covariate pattern to examine
        times_to_eval = [1, 1.25, 3, 5]
        dta = data_sc.copy()
        dta['X'] = 1
        dta['W'] = 1

        def psi(theta):
            aft = ee_aft_weibull(theta=theta[0:4], t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['X', 'W']])
            pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dta[['X', 'W']],
                                                 times=times_to_eval, measure='cumulative_hazard',
                                                 mu=theta[0], beta=theta[1:3], sigma=theta[3])
            return np.vstack((aft, pred_surv_t))

        # M-estimator with built-in Weibull AFT
        mestimator = MEstimator(psi, init=[1., 0., 0., 0.6] + [0.5, ]*len(times_to_eval))
        mestimator.estimate(solver="lm")

        # Predictions from Weibull AFT from lifelines for comparison
        waft = WeibullAFTFitter()
        waft.fit(data_sc[['X', 'W', 't', 'delta']], 't', 'delta', ancillary=False, robust=True)
        preds = waft.predict_cumulative_hazard(dta.iloc[0], times=times_to_eval)

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[4:],
                            np.asarray(preds).T[0],
                            atol=1e-4)
