####################################################################################################################
# Tests for sandwich computations
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
from delicatessen import compute_sandwich, MEstimator
from delicatessen.estimating_equations import ee_mean_variance, ee_glm
from delicatessen.sandwich import compute_bread, compute_meat, build_sandwich


@pytest.fixture
def y():
    return np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])


@pytest.fixture
def x():
    return np.array([2, -1, 2, 6, 2, 4, -5, 7, -5, 1, 3, 1, 1, 0])


class TestBread:

    def test_error_method(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        with pytest.raises(ValueError, match="input for deriv_method"):
            compute_bread(psi, theta=[mean, ], deriv_method='wrong')

    def test_error_nan(self, y):
        def psi(theta):
            return y - theta

        with pytest.warns(UserWarning, match="contains at least one np.nan"):
            compute_bread(psi, theta=[np.nan, ], deriv_method='approx')

    def test_approx(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        bread = compute_bread(psi, theta=[mean, ], deriv_method='approx')
        npt.assert_allclose([[1*len(y), ], ], bread, atol=1e-7)

    def test_fapprox(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        bread = compute_bread(psi, theta=[mean, ], deriv_method='fapprox')
        npt.assert_allclose([[1*len(y), ], ], bread, atol=1e-7)

    def test_bapprox(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        bread = compute_bread(psi, theta=[mean, ], deriv_method='bapprox')
        npt.assert_allclose([[1*len(y), ], ], bread, atol=1e-7)

    def test_capprox(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        bread = compute_bread(psi, theta=[mean, ], deriv_method='capprox')
        npt.assert_allclose([[1*len(y), ], ], bread, atol=1e-7)

    def test_exact(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        bread = compute_bread(psi, theta=[mean, ], deriv_method='exact')
        npt.assert_allclose([[1*len(y), ], ], bread, atol=1e-7)

    def test_approx_2d(self, y, x):
        def psi(theta):
            return [y - theta[0], x - theta[1]]

        mean_y = np.mean(y)
        mean_x = np.mean(x)
        bread = compute_bread(psi, theta=[mean_y, mean_x], deriv_method='approx')
        npt.assert_allclose([[1*len(y), 0], [0, 1*len(x)]],
                            bread, atol=1e-7)

    def test_fapprox_2d(self, y, x):
        def psi(theta):
            return [y - theta[0], x - theta[1]]

        mean_y = np.mean(y)
        mean_x = np.mean(x)
        bread = compute_bread(psi, theta=[mean_y, mean_x], deriv_method='fapprox')
        npt.assert_allclose([[1*len(y), 0], [0, 1*len(x)]],
                            bread, atol=1e-7)

    def test_bapprox_2d(self, y, x):
        def psi(theta):
            return [y - theta[0], x - theta[1]]

        mean_y = np.mean(y)
        mean_x = np.mean(x)
        bread = compute_bread(psi, theta=[mean_y, mean_x], deriv_method='bapprox')
        npt.assert_allclose([[1*len(y), 0], [0, 1*len(x)]],
                            bread, atol=1e-7)

    def test_capprox_2d(self, y, x):
        def psi(theta):
            return [y - theta[0], x - theta[1]]

        mean_y = np.mean(y)
        mean_x = np.mean(x)
        bread = compute_bread(psi, theta=[mean_y, mean_x], deriv_method='capprox')
        npt.assert_allclose([[1*len(y), 0], [0, 1*len(x)]],
                            bread, atol=1e-7)

    def test_exact_2d(self, y, x):
        def psi(theta):
            return [y - theta[0], x - theta[1]]

        mean_y = np.mean(y)
        mean_x = np.mean(x)
        bread = compute_bread(psi, theta=[mean_y, mean_x], deriv_method='exact')
        npt.assert_allclose([[1*len(y), 0], [0, 1*len(x)]],
                            bread, atol=1e-7)


class TestMeat:

    def test_1d(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        meat = compute_meat(psi, theta=[mean, ]) / len(y)
        npt.assert_allclose(np.var(y, ddof=0), meat, atol=1e-7)

    def test_2d(self, y, x):
        def psi(theta):
            return [y - theta[0], x - theta[1]]

        mean_y = np.mean(y)
        mean_x = np.mean(x)
        meat = compute_meat(psi, theta=[mean_y, mean_x]) / len(y)
        npt.assert_allclose(np.cov(y, x, ddof=0), meat, atol=1e-7)


class TestBuildSandwich:

    def test_nan(self, y):
        def psi(theta):
            return y - theta

        mean = np.nan
        bread = compute_bread(psi, theta=[mean, ], deriv_method='approx')
        meat = compute_meat(psi, theta=[mean, ])
        sandwich = build_sandwich(bread=bread, meat=meat)
        assert np.isnan(sandwich)

    def test_solve_1d(self, y):
        def psi(theta):
            return y - theta

        mean = np.mean(y)
        bread = compute_bread(psi, theta=[mean, ], deriv_method='approx') / len(y)
        meat = compute_meat(psi, theta=[mean, ]) / len(y)
        sandwich = build_sandwich(bread=bread, meat=meat)
        npt.assert_allclose(np.var(y, ddof=0), sandwich, atol=1e-7)

    def test_solve_2d(self, y, x):
        def psi(theta):
            return [y - theta[0], x - theta[1]]

        mean_y = np.mean(y)
        mean_x = np.mean(x)
        bread = compute_bread(psi, theta=[mean_y, mean_x], deriv_method='approx') / len(y)
        meat = compute_meat(psi, theta=[mean_y, mean_x]) / len(y)
        sandwich = build_sandwich(bread=bread, meat=meat)
        npt.assert_allclose(np.cov(y, x, ddof=0), sandwich, atol=1e-7)


class TestComputeSandwich:

    def test_docs_example(self):
        def psi(theta):
            return ee_mean_variance(theta=theta, y=y_dat)

        y_dat = [1, 2, 4, 1, 2, 3, 1, 5, 2]
        mean = np.mean(y_dat)
        var = np.var(y_dat, ddof=0)
        sandwich = compute_sandwich(stacked_equations=psi, theta=[mean, var]) / len(y_dat)
        npt.assert_allclose([[np.var(y_dat, ddof=0) / len(y_dat), 0.20576132],
                             [0.20576132, 0.48834019]],
                            sandwich, rtol=1e-6)

    def test_compute_versus_build(self):
        d = pd.DataFrame()
        d['X'] = [1, -1, 0, 1, 2, 1, -2, -1, 0, 3, -3, 1, 1, -1, -1, -2, 2, 0, -1, 0]
        d['Z'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d['Y'] = [0, 0, 0, 0, 0, 15, 15, 25, 25, 45, 0, 0, 0, 0, 15, 15, 15, 25, 25, 35]
        d['I'] = 1

        # M-estimation negative binomial
        def psi(theta):
            ef_glm = ee_glm(theta[:-1], X=d[['I', 'X', 'Z']], y=d['Y'],
                            distribution='nb', link='log')
            ef_ta = np.ones(d.shape[0]) * (np.exp(theta[-2]) - theta[-1])
            return np.vstack([ef_glm, ef_ta])

        mestr = MEstimator(psi, init=[0., 0., 0., -2., 1.])
        mestr.estimate(solver='lm', maxiter=5000)
        var_build = mestr.variance

        # Compute sandwich calculations
        var_compute = compute_sandwich(psi, theta=mestr.theta, deriv_method='approx') / d.shape[0]

        # Checking variance estimates
        npt.assert_allclose(var_build, var_compute, atol=1e-7)
