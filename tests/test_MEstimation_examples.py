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
            # Get the standard error and log variance
            sqrt_var = np.ones(data.shape[0]) * np.sqrt(theta[1]) - theta[2]
            log_var = np.ones(data.shape[0]) * np.log(theta[1]) - theta[3]
            # Return values
            return mean, vari, sqrt_var, log_var

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

    # Estimating equation for 7.4.1 --> provided code in example takes only lower bound, not upper bound
    def test_robust_location(self):
        n = 1000
        k = 3
        y = np.random.normal(size=n)

        # Precalculate the robust mean
        yh = np.where(y > k, k, y)
        yhl = np.where(yh < -k, -k, yh)

        def psi_robust_mean(theta):
            k = 3
            yr = np.where(y > k, k, y)  # Applying upper bound
            yr = np.where(yr < -k, -k, yr)  # Applying lower bound
            return yr - theta

        estr = MEstimator(psi_robust_mean, init=[0., ])
        estr.estimate()

        print(estr.theta - yhl.mean())
        assert estr.theta[0] - yhl.mean() < epsilon

    def test_quantile_estimation(self):
        n = 1000
        y = np.random.normal(size=n)

        # Use numpy to find quantiles
        qs = np.quantile(y, q=[0.25, 0.50, 0.75])

        def psi_quantile(theta):
            return (
                0.25 - 1 * (y <= theta[0]),
                0.50 - 1 * (y <= theta[1]),
                0.75 - 1 * (y <= theta[2])
            )

        estr = MEstimator(psi_quantile, init=[0., 0., 0., ])
        estr.estimate(solver='hybr',
                      tolerance=1e-3,
                      dx=1,
                      order=9)

        assert estr.theta[0] - qs[0] < epsilon
        assert estr.theta[1] - qs[1] < epsilon
        assert estr.theta[2] - qs[2] < epsilon

    def test_linear_regression(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 5 * data['X'] - 2 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        def psi_regression_loop(theta):
            x = np.asarray(data[['C', 'X', 'Z']])
            y = np.asarray(data['Y'])
            beta = np.asarray(theta)[:, None]
            n = x.shape[0]

            # Place to store resulting estimate values
            est_vals = []

            for i in range(n):
                v_i = (y[i] - np.dot(x[i], beta)) * x[i]
                est_vals.append(v_i)

            return np.asarray(est_vals).T

        def psi_regression_vector(theta):
            x = np.asarray(data[['C', 'X', 'Z']])
            y = np.asarray(data['Y'])
            beta = np.asarray(theta)[:, None]

            dot = np.dot(x, beta)
            yval = y - dot
            xtim = yval.dot(x)
            return xtim.T

        mestimator1 = MEstimator(psi_regression_loop, init=[0.1, 0.1, 0.1])
        mestimator1.estimate()
        mestimator2 = MEstimator(psi_regression_vector, init=[0.1, 0.1, 0.1])
        mestimator2.estimate()

        assert mestimator1.theta[0] - mestimator2.theta[0] < epsilon
        assert mestimator1.theta[1] - mestimator2.theta[1] < epsilon
        assert mestimator1.theta[2] - mestimator2.theta[2] < epsilon

    def test_gee(self):
        pass

    def test_combination(self):
        n = 1000
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
