import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.stats import logistic
from scipy.optimize import approx_fprime

from delicatessen.utilities import inverse_logit, identity
from delicatessen.derivative import auto_differentiation
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_mean_variance, ee_regression

np.random.seed(20230704)


class TestAutoDifferentiation:

    def test_compare_elementary_operators(self):
        # Defining the functions to check
        def f(x):
            return [10,
                    10 - 5 + 32*5 - 6**2,
                    5 + x[0] + x[1] - x[2] - x[3],
                    -32 + x[0]*x[2] + x[1]*x[3],
                    x[1]**2 + x[0] + x[3] - 30,
                    -32 + x[0]**x[2] + x[1]**x[3],
                    (x[0] + x[1])**(x[2] + x[3]) + 6,
                    5*x[1]**2 + (x[2]**2)*5,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_single_evaluation(self):
        def f(x):
            return -32 + 4*x - 10*x**2

        # Points to Evaluate at
        xinput = [2.754, ]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)


    def test_compare_equality1_operators(self):
        # Defining the functions to check
        def f(x):
            return [(x[0] <= 0.1)*x[0] + x[1]**x[2],
                    (x[0] < 0.1)*x[0] + x[1]**x[2],
                    (x[0] >= 0.1) * x[0] + x[1] ** x[2],
                    (x[0] > 0.1) * x[0] + x[1] ** x[2],
                    (x[0] >= 5.0)*x[0] + x[1]**x[2],
                    (x[0] > 5.0)*x[0] + x[1]**x[2],
                    (x[0] <= 5.0)*x[0] + x[1]**x[2],
                    (x[0] < 5.0)*x[0] + x[1]**x[2],
                    (x[0] <= 5.1)*(x[0] <= 7.0)*(x[0] ** 2.5)*(x[0] + 3)**0.5 + 27*x[0]**3,
                    (x[0] < 5.1) * (x[0] + x[1] ** 2) ** 3,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_equality2_operators(self):
        # Defining the functions to check
        def f(x):
            return [(x[0] == 0.5) * (x[0] + x[1]**2),
                    (x[0] != 0.5) * (x[0] + x[1]**2),
                    (x[0] == 5.5) * (x[0] + x[1]**2),
                    (x[0] != 5.5) * (x[0] + x[1]**2),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Truth by-hand (approx_fprime breaks here due to shifts)
        dx_true = np.array([[1., 3.8, 0., 0., ],
                            [0., 0., 0., 0., ],
                            [0., 0., 0., 0., ],
                            [1., 3.8, 0., 0., ]])

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_true, dx_exact, atol=1e-9)

    def test_compare_numpy_operators(self):
        # Defining the functions to check
        def f(x):
            return [np.sum(x),
                    np.prod(x),
                    np.dot(np.array([1, -2, 3]), x[0:3].T),
                    np.power(x[1], 0) + np.power(x[2], 2),
                    -np.power(x[0], 2) + 3*np.power(x[3], 2),
                    np.sqrt(x[1]),

                    np.abs(x[0] + x[2]),
                    np.abs(x[0] + (x[1] + x[0])**2),
                    np.sign(x[2]),
                    np.sign(x[2])*x[0] + x[1]**x[2],

                    5 + np.exp(1) + x[1] - x[2],
                    np.exp(x[0] - x[1]) + 1.5*x[2]**3,
                    5 + np.exp(x[0] + x[1]) + x[2] + 5 * x[3],

                    np.log(x[0] + x[1]) + x[3]**1.5,
                    np.log2(x[1]) + np.log10(x[3]) - x[2]**2,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_numpy_trig(self):
        # Defining the functions to check
        def f(x):
            return [np.sin(x[0]) + x[0]**2,
                    np.cos(x[1]**2) + x[2],
                    np.tan(x[2])**2 - 5*x[0]*x[1]**2,
                    np.arcsin(x[0]) - 2*np.arccos(x[1]) + np.arctan(0.1*x[2]) + x[3]**3,
                    np.arctan(x[0] + np.arcsin(0.5*x[1] + 0.1*x[2])),
                    np.sinh(1.1*x[0]) + np.cosh(0.5*x[1]) + np.tanh(0.9*x[2] + x[3]),
                    np.arcsinh(x[0]) - 2*np.arccosh(x[1]) + np.arctanh(0.1*x[2]) + x[3]**3,
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_numpy_comparisons(self):
        # Defining the functions to check
        def f(x):
            return [np.where(x[0] <= 0.1, 1, 0),
                    np.where(x[0] <= 5.1, 1, 0),
                    np.where(x[0] <= 5.1, 1, 0) * x[1],
                    np.where(x[0] <= 0.1, 1, 0) * x[0],
                    np.where(x[0] <= 5.1, 1, 0) * (x[0] ** 2.5),
                    np.where(x[0] <= 5.1, x[0] ** 2.5, 0),
                    np.where(x[0] < 5.1, 1, 0)*(x[3]**1.5),
                    np.where(x[0] < 5.1, x[3] ** 1.5, 0),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_compare_numpy_truncate(self):
        # Defining the functions to check
        def f(x):
            return [np.floor(x[0]),
                    1 + np.floor(x[0] + x[1]),
                    np.floor(x[3] - 0.1),
                    np.ceil(x[1]),
                    np.clip(x[0], -10, 10),
                    np.clip(x[0], 8, 10),
                    np.clip(x[0], -10, -8),
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-5)

    def test_numpy_hard_comparisons(self):
        # Defining the functions to check
        def f(x):
            return [x[1] ** 2 + 1 * np.where(x[2] == -2.3, 1, 0) * (x[0] + x[3]),  # Equality causes bad approx
                    x[1] ** 2 + 1 * np.where(x[2] == -2.3, x[0] + x[3], 0),        # Equality causes bad approx
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # True Derivatives
        dx_true = np.array([[1., 3.8, 0., 1.],
                            [1., 3.8, 0., 1.],
                            ])

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_true, dx_exact, atol=1e-5)

    def test_compare_deli_utilities(self):
        # Defining the functions to check
        def f(x):
            return [inverse_logit(-0.75 + 0.5*x[0] - 0.78*x[1] + 1.45*x[2] + 1.2*x[3]),
                    identity(-0.75 + 0.5*x[0] - 0.78*x[1] + 1.45*x[2]) + 1.2*x[3], ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 3]

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-7)

    def test_large_gradient(self):
        # Function to evaluate (with 500 parameters)
        def f(x):
            x_init = 1
            storage = []
            for xi in x:
                y = np.sin(x_init - xi**2 - 0.01*xi)
                storage.append(y)
                x_init = xi
            return storage

        # Points to Evaluate at
        xinput = np.linspace(-10, 10, 500)

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)
        dx_approx = approx_fprime(xinput, f, epsilon=1e-9)

        # Checking
        npt.assert_allclose(dx_approx, dx_exact, atol=1e-4)

    def test_undefined_derivs(self):
        # Defining the functions to check
        def f(x):
            return [np.floor(x[3]) + x[0],                 # Floor is not defined if no remain
                    np.ceil(x[3]) + x[1],                  # Ceil is not defined if no remain
                    np.abs(x[0] - 1/x[3]) + x[2],          # Absolute value not defined at 0
                    ]

        # Points to Evaluate at
        xinput = [0.5, 1.9, -2.3, 2]

        # True Derivatives
        dx_true = np.array([[np.nan, np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan, np.nan],
                            ])

        # Evaluating the derivatives at the points
        dx_exact = auto_differentiation(xinput, f)

        # Checking
        npt.assert_allclose(dx_true, dx_exact, atol=1e-5)


class TestSandwichAutoDiff:

    def test_exact_bread_mean(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return y - theta

        mestr = MEstimator(psi, init=[0, ])
        mestr.estimate(deriv_method='exact')

        # Checking bread estimates
        npt.assert_allclose(mestr.bread,
                            [[1]],
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(mestr.asymptotic_variance,
                            np.var(y, ddof=0),
                            atol=1e-6)

    def test_exact_bread_mean_var(self):
        # Data set
        y = np.array([5, 1, 2, 4, 2, 4, 5, 7, 11, 1, 6, 3, 4, 6])

        def psi(theta):
            return ee_mean_variance(theta=theta, y=y)

        mestr = MEstimator(psi, init=[0, 1, ])
        mestr.estimate(deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-5)

    def test_exact_bread_logit_reg(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Z'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
        data['C'] = 1
        data['w'] = np.random.uniform(1, 10, size=n)

        def psi_regression(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X', 'Z']], y=data['Y'],
                                 model='logistic', weights=data['w'])

        mestr = MEstimator(psi_regression, init=[0., 2., -1.])

        # Auto-differentation
        mestr.estimate(deriv_method='exact')
        bread_exact = mestr.bread
        var_exact = mestr.variance

        # Central difference method
        mestr.estimate(deriv_method='approx')
        bread_approx = mestr.bread
        var_approx = mestr.variance

        # Checking bread estimates
        npt.assert_allclose(bread_approx,
                            bread_exact,
                            atol=1e-6)

        # Checking variance estimates
        npt.assert_allclose(var_approx,
                            var_exact,
                            atol=1e-6)
