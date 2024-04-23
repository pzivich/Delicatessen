####################################################################################################################
# Tests for built-in estimating equations -- measurement error
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression, ee_rogan_gladen, ee_rogan_gladen_extended
from delicatessen.utilities import inverse_logit, logit


class TestEstimatingEquationsMeasurement:

    @pytest.fixture
    def cole2023_data(self):
        # Data from Cole et al. (2023)
        d = pd.DataFrame()
        d['Y_star'] = [0, 1] + [0, 1, 0, 1]
        d['Y'] = [np.nan, np.nan] + [0, 0, 1, 1]
        d['S'] = [1, 1] + [0, 0, 0, 0]
        d['n'] = [950-680, 680] + [71, 18, 38, 203]
        d = pd.DataFrame(np.repeat(d.values, d['n'], axis=0), columns=d.columns)
        d = d[['Y', 'Y_star', 'S']].copy()
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

    @pytest.fixture
    def data_covs(self):
        # Compact data set for X=1
        dc = pd.DataFrame()
        dc['Y_star'] = [0, 1] + [0, 1, 0, 1]
        dc['Y'] = [np.nan, np.nan] + [0, 0, 1, 1]
        dc['S'] = [1, 1] + [0, 0, 0, 0]
        dc['n'] = [400, 400] + [75, 25, 5, 95]
        d1 = pd.DataFrame(np.repeat(dc.values, dc['n'], axis=0), columns=dc.columns)
        d1 = d1[['Y', 'Y_star', 'S']].copy()
        d1['C'] = 1
        d1['X'] = 1
        d1['weights'] = 1

        # Compact data set for X=1
        dc = pd.DataFrame()
        dc['Y_star'] = [0, 1] + [0, 1, 0, 1]
        dc['Y'] = [np.nan, np.nan] + [0, 0, 1, 1]
        dc['S'] = [1, 1] + [0, 0, 0, 0]
        dc['n'] = [100, 100] + [85, 15, 20, 80]
        d0 = pd.DataFrame(np.repeat(dc.values, dc['n'], axis=0), columns=dc.columns)
        d0 = d0[['Y', 'Y_star', 'S']].copy()
        d0['C'] = 1
        d0['X'] = 0
        d0['weights'] = np.where(d0['Y'].isna(), 2, 1)

        return pd.concat([d1, d0])

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

    def test_extended_rogan_gladen(self, cole2023_data):
        # Replicate Cole et al. 2023 Rogan-Gladen as a comparison for the extended version

        def psi(theta):
            return ee_rogan_gladen(theta,
                                   y=cole2023_data['Y'],
                                   y_star=cole2023_data['Y_star'],
                                   r=cole2023_data['S'])

        estr0 = MEstimator(psi, init=[0.5, 0.5, .75, .75])
        estr0.estimate(solver='lm')

        def psi(theta):
            return ee_rogan_gladen_extended(theta=theta, y=cole2023_data['Y'],
                                            y_star=cole2023_data['Y_star'],
                                            X=cole2023_data[['C', ]],
                                            r=cole2023_data['S'])

        estr1 = MEstimator(psi, init=[0.5, logit(0.75), logit(0.75), ])
        estr1.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr0.theta[0], estr1.theta[0], atol=1e-6)
        npt.assert_allclose(estr0.theta[2], inverse_logit(estr1.theta[1]), atol=1e-6)
        npt.assert_allclose(estr0.theta[3], inverse_logit(estr1.theta[2]), atol=1e-6)

        # Checking variance estimate
        npt.assert_allclose(estr0.variance[0, 0], estr1.variance[0, 0], atol=1e-6)

    def test_rogan_gladen_extended_weights(self, cole_roses_data):
        # Replicate Cole et al. 202x Rejoinder example as a test
        dr = cole_roses_data
        y_no_nan = np.asarray(dr['Y'].fillna(-9))
        ystar_no_nan = np.asarray(dr['Y_star'].fillna(-9))
        w_no_nan = np.asarray(dr[['C', 'W']].fillna(-9))
        s1 = np.where(dr['S'] == 1, 1, 0)
        s2 = np.where(dr['S'] == 2, 1, 0)
        s3 = np.where(dr['S'] == 3, 1, 0)

        def psi(theta):
            param = theta[:3]
            beta = theta[3:]

            # Inverse odds weights model
            ee_sm = ee_regression(beta, X=w_no_nan, y=s2,
                                  model='logistic')
            ee_sm = ee_sm * (1 - s3)
            pi_s = inverse_logit(np.dot(w_no_nan, beta))
            iosw = s1 * pi_s / (1-pi_s) + s3

            # Rogan-Gladen
            ee_rg = ee_rogan_gladen_extended(param,
                                             y=y_no_nan,
                                             y_star=ystar_no_nan,
                                             r=s1, X=dr[['C', ]],
                                             weights=iosw)
            ee_rg = ee_rg * (1 - s2)
            return np.vstack([ee_rg, ee_sm])

        estr = MEstimator(psi, init=[0.5, .75, .75, 0., 0.])
        estr.estimate(solver='lm')

        reference_theta = [0.0967144999, logit(0.9), logit(0.8)]

        # Checking mean estimate
        npt.assert_allclose(estr.theta[0:3], reference_theta,
                            atol=1e-6)

    def test_rogan_gladen_extended_covs(self, data_covs):
        d = data_covs

        def psi(theta):
            return ee_rogan_gladen_extended(theta=theta, y=d['Y'], y_star=d['Y_star'],
                                            X=d[['C', 'X']], r=d['S'])

        estr = MEstimator(psi, init=[0.5, 1., 0., 1., 0., ])
        estr.estimate(solver='lm')

        # Checking sensitivity
        npt.assert_allclose(.8, inverse_logit(estr.theta[1]), atol=1e-7)
        npt.assert_allclose(.95, inverse_logit(estr.theta[1] + estr.theta[2]), atol=1e-7)

        # Checking specificity
        npt.assert_allclose(.85, inverse_logit(estr.theta[3]), atol=1e-7)
        npt.assert_allclose(.75, inverse_logit(estr.theta[3] + estr.theta[4]), atol=1e-7)

        # Checking corrected mean
        pr_x1 = 0.8
        corrected_mu = ((1-pr_x1) * ((0.5 + 0.85 - 1) / (0.85 + 0.8 - 1))
                        + pr_x1 * ((0.5 + 0.75 - 1) / (0.75 + 0.95 - 1)))
        npt.assert_allclose(corrected_mu, estr.theta[0], atol=1e-7)

    def test_rogan_gladen_extended_covs_weights(self, data_covs):
        d = data_covs

        def psi(theta):
            return ee_rogan_gladen_extended(theta=theta, y=d['Y'], y_star=d['Y_star'],
                                            X=d[['C', 'X']], r=d['S'], weights=d['weights'])

        estr = MEstimator(psi, init=[0.5, 1., 0., 1., 0., ])
        estr.estimate(solver='lm')

        # Checking sensitivity
        npt.assert_allclose(.8, inverse_logit(estr.theta[1]), atol=1e-7)
        npt.assert_allclose(.95, inverse_logit(estr.theta[1] + estr.theta[2]), atol=1e-7)

        # Checking specificity
        npt.assert_allclose(.85, inverse_logit(estr.theta[3]), atol=1e-7)
        npt.assert_allclose(.75, inverse_logit(estr.theta[3] + estr.theta[4]), atol=1e-7)

        # Checking corrected mean
        pr_x1 = 800 / 1200
        corrected_mu = ((1-pr_x1) * ((0.5 + 0.85 - 1) / (0.85 + 0.8 - 1))
                        + pr_x1 * ((0.5 + 0.75 - 1) / (0.75 + 0.95 - 1)))
        npt.assert_allclose(corrected_mu, estr.theta[0], atol=1e-7)
