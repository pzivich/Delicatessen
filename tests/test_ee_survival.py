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
                                               ee_weibull_measure, ee_aft_weibull, ee_aft_weibull_measure,
                                               ee_aft)


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
    def collett_bc(self):
        arr = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [23, 47, 69, 70, 71, 100, 101, 148, 181, 198, 208, 212, 224, 5, 8, 10, 13, 18, 24, 26, 26, 31, 35, 40,
                41, 48, 50, 59, 61, 68, 71, 76, 105, 107, 109, 113, 116, 118, 143, 154, 162, 188, 212, 217, 225],
               [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]]
        d = pd.DataFrame()
        d['X'] = arr[0]
        d['time'] = arr[1]
        d['delta'] = arr[2]
        d['C'] = 1
        return d

    @pytest.fixture
    def data_sc(self):
        d = pd.DataFrame()
        d['X'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
        d['W'] = [1, 2, 3, 1, -1, 1, -2, 1, 0, 1, -3, 1, 0, 1, -1, 3, 0, 2, 1, 1, -1, 2, -3, 2, -2, 0, 0,
                  1, 1, 2, 3, 0, 1, 0, -1]
        d['t'] = [1, 2, 3, 5, 1, 9, 10, 1, 3, 6, 8, 6, 9, 4, 3, 2, 1, 1, 5, 6, 7, 8, 3, 4, 4, 3, 2.1, 3,
                  5, 3, 2, 1, 3, 2, 10]
        d['delta'] = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                      0, 1, 0, 1, 0, 1, 0]
        d['weight'] = [1, 2, 1, 0.1, 1, 1, 1, 3, 1, 1, 3, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 6,
                       1, 1, 1, 1, 2, 1, 4, 1]
        d['C'] = 1
        return d

    def test_aft_exponential(self, collett_bc):
        # R code
        # library(survival)
        # library(collett)
        # library(sandwich)
        # bcancer$stain = bcancer$stain - 1
        # bcancer$delta = bcancer$status
        # sr = survreg(Surv(time,delta)~stain, bcancer, dist='exponential')
        # sr$coefficients
        # sandwich(sr)
        comparison_theta = np.asarray([5.8003040, -0.9516276])
        comparison_var = np.asarray([[0.1855775, -0.1855775],
                                     [-0.1855775, 0.2473048]])

        d = collett_bc

        def psi(theta):
            return ee_aft(theta=theta, t=d['time'], delta=d['delta'], X=d[['C', 'X']], distribution='exponential')

        # M-estimator with built-in Weibull AFT
        estr = MEstimator(psi, init=[2., 0.])
        estr.estimate(solver="lm")

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-7)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, comparison_var, atol=1e-6)

    def test_aft_weibull(self, collett_bc):
        # R code
        # library(survival)
        # library(collett)
        # library(sandwich)
        # bcancer$stain = bcancer$stain - 1
        # bcancer$delta = bcancer$status
        # sr = survreg(Surv(time,delta)~stain, bcancer, dist='weibull')
        # sr$coefficients
        # sr$scale
        # sandwich(sr)
        comparison_theta = np.asarray([5.8543638, -0.9966647, np.log(1 / 1.066777)])
        comparison_var = np.asarray([[0.23281646, -0.2231509, -0.02183579],
                                     [-0.22315087, 0.2827840, 0.01221040],
                                     [-0.02183579, 0.0122104, 0.01580507]])

        d = collett_bc

        def psi(theta):
            return ee_aft(theta=theta, t=d['time'], delta=d['delta'], X=d[['C', 'X']], distribution='weibull')

        # M-estimator with built-in Weibull AFT
        estr = MEstimator(psi, init=[5., -.5, 0.])
        estr.estimate(solver="lm")

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-6)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, comparison_var, atol=1e-5)

    def test_aft_loglogistic(self, collett_bc):
        # R code
        # library(survival)
        # library(collett)
        # library(sandwich)
        # bcancer$stain = bcancer$stain - 1
        # bcancer$delta = bcancer$status
        # sr = survreg(Surv(time,delta)~stain, bcancer, dist='loglogistic')
        # sr$coefficients
        # sr$scale
        # sandwich(sr)
        comparison_theta = np.asarray([5.461100, -1.149056, np.log(1 / 0.8047005)])
        comparison_var = np.asarray([[0.19819085, -0.188012820, -0.017487829],
                                     [-0.18801282, 0.255164051, 0.003017458],
                                     [-0.01748783, 0.003017458, 0.021047226]])

        d = collett_bc

        def psi(theta):
            return ee_aft(theta=theta, t=d['time'], delta=d['delta'], X=d[['C', 'X']], distribution='log-logistic')

        # M-estimator with built-in log-logistic AFT
        estr = MEstimator(psi, init=[5., -.5, 0.])
        estr.estimate(solver="lm")

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-6)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, comparison_var, atol=1e-5)

    def test_aft_lognormal(self, collett_bc):
        # R code
        # library(survival)
        # library(collett)
        # library(sandwich)
        # bcancer$stain = bcancer$stain - 1
        # bcancer$delta = bcancer$status
        # sr = survreg(Surv(time,delta)~stain, bcancer, dist='lognormal')
        # sr$coefficients
        # sr$scale
        # sandwich(sr)
        comparison_theta = np.asarray([5.491726, -1.151172, np.log(1 / 1.359451)])
        comparison_var = np.asarray([[0.2027042, -0.187832061, -0.020758604],
                                     [-0.1878321, 0.247731353, 0.006169943],
                                     [-0.0207586, 0.006169943, 0.018085983]])

        d = collett_bc

        def psi(theta):
            return ee_aft(theta=theta, t=d['time'], delta=d['delta'], X=d[['C', 'X']], distribution='log-normal')

        # M-estimator with built-in log-normal AFT
        estr = MEstimator(psi, init=[5., -.5, 0.])
        estr.estimate(solver="lm")

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-6)

        # Checking variance estimate
        npt.assert_allclose(estr.variance, comparison_var, atol=1e-5)

    def test_weighted_weibull_aft(self, data_sc):
        def psi(theta):
            return ee_aft(theta=theta, weights=data_sc['weight'],
                          t=data_sc['t'], delta=data_sc['delta'], X=data_sc[['C', 'X', 'W']], distribution='weibull')

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
