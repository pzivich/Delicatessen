import pytest
import numpy as np
import pandas as pd
from delicatessen import MEstimator

epsilon = 1.0E-6


class TestMEstimationExamples:
    def test_sample_mean(self):
        # Define dataset
        y = np.array([-2, 1, 3, 4, 1, 4, -5, 3, 6])

        def psi(theta):
            mean = y - theta[0]
            vari = (y - theta[0]) ** 2 - theta[1]
            return mean, vari

        mestimate = MEstimator(psi, init=[0, 1])
        mestimate.estimate()

        assert mestimate.theta[0] - np.mean(y) < epsilon
        assert mestimate.theta[1] - np.var(y) < epsilon

    # Ratio 7.2.3 --> typo in equation?
    # Y_i - \theta_1 X_i
    def test_ratio(self):
        n = 10000
        y_loc = 10
        x_loc = 5
        data = pd.DataFrame()
        data['Y'] = np.random.normal(loc=y_loc, scale=2, size=n)
        data['X'] = np.random.normal(loc=x_loc, scale=1, size=n)

        def psi(theta):
            return data['Y'] - data['X'] * theta

        mestimate = MEstimator(psi, init=[1, ])
        mestimate.estimate()

        assert mestimate.theta[0] - data['Y'].mean() / data['X'].mean() < epsilon

        def psi(theta):
            mean_y = data['Y'] - theta[0]
            mean_x = data['X'] - theta[1]
            ratio = np.ones(data.shape[0]) * (theta[0] - theta[1] * theta[2])
            return mean_y, mean_x, ratio

        mestimate = MEstimator(psi, init=[0, 0, 1])
        mestimate.estimate()

        assert mestimate.theta[0] - data['Y'].mean() < epsilon
        assert mestimate.theta[1] - data['X'].mean() < epsilon
        assert mestimate.theta[2] - data['Y'].mean() / data['X'].mean() < epsilon

    def test_delta_method(self):
        n = 10000
        y_loc = 16
        data = pd.DataFrame(np.random.normal(loc=y_loc, scale=2, size=n),
                            columns=['Y'])

        def psi_delta(theta):
            # Get the mean and variance (nearly identical to sample mean)
            mean = data['Y'] - theta[0]
            vari = (data['Y'] - theta[0]) ** 2 - theta[1]
            # Get the standard error and log variance?
            root = np.ones(data.shape[0]) * np.sqrt(theta[1]) - theta[2]
            loga = np.ones(data.shape[0]) * np.log(theta[1]) - theta[3]
            # Return values
            return mean, vari, root, loga

        mestimate = MEstimator(psi_delta, init=[0, 0, 1, 1])
        mestimate.estimate()

        assert mestimate.theta[0] - data['Y'].mean() < epsilon
        assert mestimate.theta[1] - data['Y'].var() < epsilon
        assert mestimate.theta[2] - data['Y'].std() < epsilon
        assert mestimate.theta[3] - np.log(data['Y'].var()) < epsilon

    # Typo: Psi equations should have subscripts to index by i?
    def test_instrumental_variable(self):
        n = 10000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2 * data['X'] + np.random.normal(loc=0, size=n)
        data['W'] = data['X'] + np.random.normal(loc=0, size=n)
        data['T'] = -0.75 - 1 * data['X'] + np.random.normal(loc=0, size=n)

        def psi(theta):
            tinf = theta[0] - data['T']
            tvar = (data['Y'] - theta[1] * data['W']) * (theta[0] - data['T'])
            return tinf, tvar

        def psi_instrumental(theta):
            tinf = theta[0] - data['T']
            winf = theta[1] - data['W']
            wvar = (data['Y'] - theta[2] * data['W']) * (theta[1] - data['W'])
            tvar = (data['Y'] - theta[3] * data['W']) * (theta[0] - data['T'])
            return tinf, winf, wvar, tvar

        mestimate1 = MEstimator(psi, init=[0.1, 0.1, ])
        mestimate2 = MEstimator(psi_instrumental, init=[0.1, 0.1, 0.1, 0.1])
        mestimate1.estimate()
        mestimate2.estimate()

        # Psi1 of docs equivalent, as is Psi2 and Psi4
        assert mestimate1.theta[0] == mestimate2.theta[0]
        assert mestimate1.theta[1] == mestimate2.theta[3]

        # Test actual values

    def test_robust_location(self):
        pass

    def test_linear_regression(self):
        pass

    def test_gee(self):
        pass

    def test_combination(self):
        n = 10000
        y_loc = 10
        x_loc = 5
        data = pd.DataFrame()
        data['Y'] = np.random.normal(loc=y_loc, scale=2, size=n)
        data['X'] = np.random.normal(loc=x_loc, scale=1, size=n)

        def psi(theta):
            mean_y = data['Y'] - theta[0]
            mean_x = data['X'] - theta[1]
            ratio = np.ones(data.shape[0]) * (theta[0] - theta[1] * theta[2])
            vary = (data['Y'] - theta[0]) ** 2 - theta[3]
            varx = (data['X'] - theta[1]) ** 2 - theta[4]
            return mean_y, mean_x, ratio, vary, varx

        mestimate = MEstimator(psi, init=[0, 0, 1, 1, 1])
        mestimate.estimate()

        assert mestimate.theta[0] - data['Y'].mean() < epsilon
        assert mestimate.theta[1] - data['X'].mean() < epsilon
        assert mestimate.theta[2] - data['Y'].mean() / data['X'].mean() < epsilon
        assert mestimate.theta[3] - data['Y'].var() < epsilon
        assert mestimate.theta[4] - data['X'].var() < epsilon
