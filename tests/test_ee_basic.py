####################################################################################################################
# Tests for built-in estimating equations -- basic
####################################################################################################################

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import gmean

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_mean, ee_mean_variance, ee_mean_robust, ee_mean_geometric


class TestEstimatingEquationsBase:

    @pytest.fixture
    def y(self):
        return np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

    @pytest.fixture
    def yr(self):
        return np.array([-10, -1, 2, 3, -2, 0, 3, 5, 12])

    def test_mean(self, y):
        def psi1(theta):
            return y - theta

        mcee = MEstimator(psi1, init=[0, ])
        mcee.estimate()

        def psi2(theta):
            return ee_mean(theta, y=y)

        mpee = MEstimator(psi2, init=[0, ])
        mpee.estimate()

        # Checking mean estimate
        npt.assert_allclose(mcee.theta, mpee.theta, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mcee.asymptotic_variance, mpee.asymptotic_variance, atol=1e-6)

    def test_mean_weighted(self):
        y1 = [1, 2, 3, 4]
        w1 = [4, 3, 2, 1]
        y2 = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]

        def psi1(theta):
            return ee_mean(theta=theta, y=y1, weights=w1)

        def psi2(theta):
            return ee_mean(theta=theta, y=y2, weights=None)

        estr1 = MEstimator(psi1, init=[0, ])
        estr1.estimate(deriv_method='exact')

        estr2 = MEstimator(psi2, init=[0, ])
        estr2.estimate()

        # Checking mean estimate
        npt.assert_allclose(estr1.theta, estr2.theta, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(estr1.variance, [[0.24, ], ], atol=1e-6)

    def test_mean_robust(self, yr):
        def byhand(theta):
            k = 6
            ee_rm = np.array(yr) - theta
            ee_rm = np.where(ee_rm > k, k, ee_rm)
            ee_rm = np.where(ee_rm < -k, -k, ee_rm)
            return ee_rm

        ref = MEstimator(byhand, init=[0, ])
        ref.estimate()

        def psi(theta):
            return ee_mean_robust(theta=theta, y=yr, k=6)

        mestimator = MEstimator(psi, init=[0, ])
        mestimator.estimate()

        # Checking mean estimate
        npt.assert_allclose(mestimator.theta[0], ref.theta[0], atol=1e-9)

    def test_mean_variance(self, y):
        def psi1(theta):
            return y - theta[0], (y - theta[0]) ** 2 - theta[1]

        mcee = MEstimator(psi1, init=[0, 0, ])
        mcee.estimate()

        def psi2(theta):
            return ee_mean_variance(theta=theta, y=y)

        mpee = MEstimator(psi2, init=[0, 0, ])
        mpee.estimate()

        # Checking mean estimate
        npt.assert_allclose(mcee.theta, mpee.theta, atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mcee.asymptotic_variance, mpee.asymptotic_variance, atol=1e-6)

    def test_mean_geometric(self, y):
        def psi(theta):
            return ee_mean_geometric(theta, y=y, weights=None, log_theta=True)

        estr = MEstimator(psi, init=[2, ])
        estr.estimate()
        mu_deli = estr.theta

        mu_scipy = gmean(y)

        # Checking mean estimate
        npt.assert_allclose(mu_deli, mu_scipy)

    def test_mean_geometric_weighted(self, y):
        weights = np.array([1, 1, 2, 1, 2, 1, 5, 1, 1, 1, 1, 3, 1, 1])

        def psi(theta):
            return ee_mean_geometric(theta, y=y, weights=weights, log_theta=True)

        estr = MEstimator(psi, init=[2, ])
        estr.estimate()
        mu_deli = estr.theta

        mu_scipy = gmean(y, weights=weights)

        # Checking mean estimate
        npt.assert_allclose(mu_deli, mu_scipy)

    def test_mean_geometric_log(self, y):
        def psi(theta):
            return ee_mean_geometric(theta, y=y, weights=None, log_theta=False)

        estr = MEstimator(psi, init=[0, ])
        estr.estimate()
        mu_deli = np.exp(estr.theta)

        mu_scipy = gmean(y)

        # Checking mean estimate
        npt.assert_allclose(mu_deli, mu_scipy)
