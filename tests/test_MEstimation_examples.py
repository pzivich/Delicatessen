import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
from delicatessen import MEstimator
from delicatessen.estimating_equations import (ee_mean, ee_mean_variance,
                                               ee_percentile, ee_positive_mean_deviation,
                                               ee_regression, ee_robust_linear_regression)

epsilon = 1.0E-6
np.random.seed(236461)


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

        # Test relative to built-in equation
        mest = MEstimator(lambda theta: ee_mean(theta=theta, y=y), init=[0, ])
        mest.estimate()

        me = MEstimator(lambda theta: ee_mean_variance(theta, y), init=[0, 1])
        me.estimate()

        assert mestimate.theta[0] - mest.theta[0] < epsilon
        assert mestimate.theta[0] - me.theta[0] < epsilon
        assert mestimate.theta[1] - me.theta[1] < epsilon

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

        mestimate = MEstimator(psi_delta, init=[16, 1, 1, 1])
        mestimate.estimate(solver='lm')

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
        n = 20000
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
                      tolerance=1e-4,
                      dx=1,
                      order=9)

        # Verify using built-in equations
        mest25 = MEstimator(lambda theta: ee_percentile(theta=theta, y=y, q=0.25), init=[-0.1, ])
        mest50 = MEstimator(lambda theta: ee_percentile(theta=theta, y=y, q=0.50), init=[0., ])
        mest75 = MEstimator(lambda theta: ee_percentile(theta=theta, y=y, q=0.75), init=[0.1, ])
        mest25.estimate(solver='hybr',
                        tolerance=1e-3,
                        dx=1,
                        order=15)
        mest50.estimate(solver='hybr',
                        tolerance=1e-3,
                        dx=1,
                        order=15)
        mest75.estimate(solver='hybr',
                        tolerance=1e-3,
                        dx=1,
                        order=15)

        assert estr.theta[0] - mest25.theta[0] < 1e-2
        assert estr.theta[1] - mest50.theta[0] < 1e-2
        assert estr.theta[2] - mest75.theta[0] < 1e-2

        assert estr.theta[0] - qs[0] < 1e-2
        assert estr.theta[1] - qs[1] < 1e-2
        assert estr.theta[2] - qs[2] < 1e-2

    def test_positive_mean_deviation(self):
        n = 10000
        y = np.random.normal(size=n)

        md = (abs(y - y.mean())).sum() / n

        def psi_deviation(theta):
            pmd = 2 * (y - theta[1]) * (y > theta[1]) - theta[0]
            med = 0.5 - (y <= theta[1])
            return pmd, med

        estr = MEstimator(psi_deviation, init=[0., 0., ])
        estr.estimate(solver='hybr',
                      tolerance=1e-3,
                      dx=1,
                      order=9)

        assert estr.theta[0] - md < 0.1
        assert estr.theta[1] - np.median(y) < 0.1

        # Test built-in equations
        mest = MEstimator(lambda theta: ee_positive_mean_deviation(theta, y), init=[0., 0., ])
        mest.estimate(solver='hybr',
                      tolerance=1e-3,
                      dx=1, order=9)

        assert estr.theta[0] - mest.theta[0] < 0.1
        assert estr.theta[1] - mest.theta[1] < 0.1

    def test_linear_regression(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 5 * data['X'] - 2 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        x = np.asarray(data[['C', 'X', 'Z']])
        y = np.asarray(data['Y'])[:, None]

        model = LinearRegression().fit(x, y)
        coef = model.coef_[0]

        def psi_regression(theta):
            beta = np.asarray(theta)[:, None]
            return ((y - np.dot(x, beta)) * x).T

        estr = MEstimator(psi_regression, init=[0., 0., 0., ])
        estr.estimate()

        assert estr.theta[0] - model.intercept_[0] < epsilon
        assert estr.theta[1] - coef[1] < epsilon
        assert estr.theta[2] - coef[2] < epsilon

        # Test built-in equations
        mest = MEstimator(lambda theta: ee_regression(theta=theta, X=data[['C', 'X', 'Z']], y=data['Y'], model='linear'),
                          init=[0., 0., 0., ])
        mest.estimate()

        assert estr.theta[0] - mest.theta[0] < epsilon
        assert estr.theta[1] - mest.theta[1] < epsilon
        assert estr.theta[2] - mest.theta[2] < epsilon

    # Using Huber regression currently - seems to be issue with extraneous results such as
    # extremely large estimated beta values
    # Seems to lead to results that are equal to LR, not RobR
    def test_robust_regression(self):
        n = 1000
        k = 2
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 5 * data['X'] - 2 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        x = np.asarray(data[['C', 'X', 'Z']])
        y = np.asarray(data['Y'])[:, None]

        def psi_robust_regression(theta):
            beta = np.asarray(theta)[:, None]
            preds = np.asarray(y - np.dot(x, beta))
            preds = np.clip(preds, a_min=-k, a_max=k)

            return (preds * x).T

        estr = MEstimator(psi_robust_regression, init=[0., 0., 0., ])
        estr.estimate(solver='hybr')

        model = HuberRegressor(epsilon=k).fit(x, np.asarray(data['Y']))
        coef = model.coef_

        assert estr.theta[0] - coef[0] < 0.3
        assert estr.theta[1] - coef[1] < 1e-2
        assert estr.theta[2] - coef[2] < 1e-2

        # Test built-in equations
        mest = MEstimator(lambda theta: ee_robust_linear_regression(theta, X=data[['C', 'X', 'Z']], y=data['Y'], k=2),
                          init=[0., 0., 0., ])
        mest.estimate(solver='hybr')

        assert estr.theta[0] - mest.theta[0] < epsilon
        assert estr.theta[1] - mest.theta[1] < epsilon
        assert estr.theta[2] - mest.theta[2] < epsilon

    # Specifically for testing whether one-lined lambda functions for built-in estimating equations throw errors.
    def test_lambda_oneliners(self):
        n = 10000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 5 * data['X'] - 2 * data['Z'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        x = np.asarray(data[['C', 'X', 'Z']])
        y = np.asarray(data['Y'])[:, None]

        # Use sklearn LinearRegression to confirm accuracy
        model = LinearRegression().fit(x, y)
        coef = model.coef_[0]

        # Initializing MEstimator should work, but actually estimating should throw error.
        # Guarantees that lambda function given must have variable "theta" not anything else.
        mest = MEstimator(lambda t: ee_regression(theta=t, X=data[['C', 'X', 'Z']], y=data['Y'], model='linear'),
                          init=[0., 0., 0., ])
        with pytest.raises(TypeError) as e_info:
            mest.estimate()

        # Initialize with valid lambda function
        mest = MEstimator(lambda theta: ee_regression(theta=theta, X=data[['C', 'X', 'Z']],
                                                      y=data['Y'], model='linear'),
                          init=[0., 0., 0., ])
        try:
            mest.estimate()
        except:
            pytest.fail("Linear regression should not throw error")

        assert mest.theta[0] - model.intercept_[0] < epsilon
        assert mest.theta[1] - coef[1] < epsilon
        assert mest.theta[2] - coef[2] < epsilon
