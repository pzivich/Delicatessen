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
        pass

    def test_instrumental_variable(self):
        pass

    def test_robust_location(self):
        pass

    def test_linear_regression(self):
        pass

    def test_gee(self):
        pass