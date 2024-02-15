####################################################################################################################
# Tests for built-in estimating equations -- basic
####################################################################################################################

import numpy as np
import numpy.testing as npt
import pytest

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_mean, ee_mean_variance, ee_mean_robust


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
