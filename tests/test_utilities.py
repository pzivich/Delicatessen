import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import (identity, inverse_logit, logit,
                                    robust_loss_functions,
                                    regression_predictions,
                                    spline,
                                    additive_design_matrix)


np.random.seed(80958151)

@pytest.fixture
def design_matrix():
    array = np.array([[1, 1, 1, 1, 1],
                      [1, 5, 10, 15, 20],
                      [1, 2, 3, 4, 5], ])
    return array.T


class TestFunctions:

    def test_identity_var(self):
        """Checks the logit transformation of a single observation
        """
        npt.assert_allclose(identity(0.5), 0.5)
        npt.assert_allclose(identity(0.25), 0.25)
        npt.assert_allclose(identity(0.75), 0.75)

    def test_logit_var(self):
        """Checks the logit transformation of a single observation
        """
        npt.assert_allclose(logit(0.5), 0.)
        npt.assert_allclose(logit(0.25), -1.098612288668)
        npt.assert_allclose(logit(0.75), 1.098612288668)

    def test_invlogit_var(self):
        """Checks the inverse logit transformation of a single observation
        """
        npt.assert_allclose(inverse_logit(0.), 0.5)
        npt.assert_allclose(inverse_logit(-1.098612288668), 0.25)
        npt.assert_allclose(inverse_logit(1.098612288668), 0.75)

    def test_logit_backtransform(self):
        """Checks the inverse logit transformation of a single observation
        """
        original = 0.6521
        original_logit = logit(original)

        # Transform 1st order
        npt.assert_allclose(inverse_logit(original_logit), original)
        npt.assert_allclose(logit(inverse_logit(original_logit)), original_logit)

        # Transform 2nd order
        npt.assert_allclose(inverse_logit(logit(inverse_logit(original_logit))), original)
        npt.assert_allclose(logit(inverse_logit(logit(inverse_logit(original_logit)))), original_logit)

        # Transform 3rd order
        npt.assert_allclose(inverse_logit(logit(inverse_logit(logit(inverse_logit(original_logit))))),
                            original)
        npt.assert_allclose(logit(inverse_logit(logit(inverse_logit(logit(inverse_logit(original_logit)))))),
                            original_logit)

    def test_identity_array(self):
        """Checks the inverse logit transformation of an array
        """
        vals = np.array([0.5, 0.25, 0.75, 0.5])
        npt.assert_allclose(identity(vals), vals)

    def test_logit_array(self):
        """Checks the inverse logit transformation of an array
        """
        prbs = np.array([0.5, 0.25, 0.75, 0.5])
        odds = np.array([0., -1.098612288668, 1.098612288668, 0.])

        npt.assert_allclose(logit(prbs), odds)

    def test_inverse_logit_array(self):
        """Checks the inverse inverse logit transformation of an array
        """
        prbs = np.array([0.5, 0.25, 0.75, 0.5])
        odds = np.array([0., -1.098612288668, 1.098612288668, 0.])

        npt.assert_allclose(inverse_logit(odds), prbs)

    def test_rloss_huber(self):
        """Checks the robust loss function: Huber's
        """
        residuals = np.array([-5, 1, 0, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='huber', k=4)
        byhand = np.array([-4, 1, 0, -2, 4, 3])

        npt.assert_allclose(func, byhand)

    def test_rloss_tukey(self):
        """Checks the robust loss function: Tukey's biweight
        """
        residuals = np.array([-5, 1, 0.1, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='tukey', k=4)

        byhand = np.array([-5 * 0,
                           1 * ((1-(1/4)**2)**2),
                           0.1 * ((1-(0.1/4)**2)**2),
                           -2 * ((1-(2/4)**2)**2),
                           8 * 0,
                           3 * ((1-(3/4)**2)**2)])

        npt.assert_allclose(func, byhand)

    def test_rloss_andrew(self):
        """Checks the robust loss function: Andrew's Sine
        """
        residuals = np.array([-5, 1, 0.1, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='andrew', k=1)

        byhand = np.array([-5 * 0,
                           np.sin(1 / 1),
                           np.sin(0.1 / 1),
                           np.sin(-2 / 1),
                           8 * 0,
                           np.sin(3 / 1)])

        npt.assert_allclose(func, byhand)

    def test_rloss_hampel(self):
        """Checks the robust loss function: Hampel
        """
        residuals = np.array([-5, 1, 1.5, -1.3, 0.1, -2, 8, 3])

        func = robust_loss_functions(residual=residuals, loss='hampel',
                                     k=4, a=1, b=2)

        byhand = np.array([-5 * 0,
                           1,
                           1,
                           -1,
                           0.1,
                           (-4 + 2)/(-4 + 2)*-1,
                           8 * 0,
                           (4 - 3)/(4 - 2)*1])

        npt.assert_allclose(func, byhand)

    def test_rloss_hampel_error(self):
        residuals = np.array([-5, 1, 1.5, -1.3, 0.1, -2, 8, 3])

        # All parameters are specified
        with pytest.raises(ValueError, match="requires the optional"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=4)

        # Ordering of parameters
        with pytest.raises(ValueError, match="requires that a < b < k"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=-4, a=1, b=2)
        with pytest.raises(ValueError, match="requires that a < b < k"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=4, a=1, b=-2)
        with pytest.raises(ValueError, match="requires that a < b < k"):
            robust_loss_functions(residual=residuals, loss='hampel',
                                  k=1.5, a=1, b=2)

    def test_regression_predictions_linear(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Y'] = 0.5 + 2*data['X'] + np.random.normal(loc=0, size=n)
        data['C'] = 1

        p = pd.DataFrame()
        p['X'] = np.linspace(-2.5, 2.5, 20)
        p['C'] = 1

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X']], y=data['Y'],
                                 model='linear')

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate(solver='lm')
        returned = regression_predictions(X=p[['C', 'X']], theta=estr.theta, covariance=estr.variance)

        # Statsmodels function equivalent
        glm = smf.glm("Y ~ X", data).fit(cov_type="HC1")
        expected = np.asarray(glm.get_prediction(p).summary_frame())
        expected[:, 1] = expected[:, 1]**2

        npt.assert_allclose(returned, expected, atol=1e-6)

    def test_regression_predictions_logit(self):
        n = 1000
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Y'] = np.random.binomial(n=1, p=logistic.cdf(-0.5 + 2*data['X']), size=n)
        data['C'] = 1

        p = pd.DataFrame()
        p['X'] = np.linspace(-2.5, 2.5, 20)
        p['C'] = 1

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X']], y=data['Y'],
                                 model='logistic')

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate(solver='lm')
        returned = regression_predictions(X=p[['C', 'X']], theta=estr.theta, covariance=estr.variance)

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X", data, family=sm.families.Binomial()).fit(cov_type="HC1")
        expected = np.asarray(glm.get_prediction(p).summary_frame())

        npt.assert_allclose(returned[:, 0], logit(expected[:, 0]), atol=1e-6)
        npt.assert_allclose(returned[:, 2], logit(expected[:, 2]), atol=1e-6)
        npt.assert_allclose(returned[:, 3], logit(expected[:, 3]), atol=1e-6)

    def test_regression_predictions_poisson(self):
        n = 500
        data = pd.DataFrame()
        data['X'] = np.random.normal(size=n)
        data['Y'] = np.random.poisson(lam=np.exp(-1 + 2*data['X']), size=n)
        data['C'] = 1

        p = pd.DataFrame()
        p['X'] = np.linspace(-2.5, 2.5, 20)
        p['C'] = 1

        def psi(theta):
            return ee_regression(theta,
                                 X=data[['C', 'X']], y=data['Y'],
                                 model='poisson')

        estr = MEstimator(psi, init=[0., 0.])
        estr.estimate(solver='lm')
        returned = regression_predictions(X=p[['C', 'X']], theta=estr.theta, covariance=estr.variance)

        # Comparing to statsmodels GLM (with robust covariance)
        glm = smf.glm("Y ~ X", data, family=sm.families.Poisson()).fit(cov_type="HC1")
        expected = np.asarray(glm.get_prediction(p).summary_frame())

        npt.assert_allclose(returned[:, 0], np.log(expected[:, 0]), atol=1e-6)
        npt.assert_allclose(returned[:, 2], np.log(expected[:, 2]), atol=1e-6)
        npt.assert_allclose(returned[:, 3], np.log(expected[:, 3]), atol=1e-6)

    def test_spline1(self):
        vars = [1, 5, 10, 15, 20]

        # Spline setup 1
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 4.0], ]).T
        returned = spline(variable=vars, knots=[16, ], power=1, restricted=False)
        npt.assert_allclose(returned, expected)

        # Spline setup 2
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 16.0], ]).T
        returned = spline(variable=vars, knots=[16, ], power=2, restricted=False)
        npt.assert_allclose(returned, expected)

        # Spline setup 3
        expected = np.array([[0.0, 0.0, 0.0, 5.0**1.5, 10.0**1.5],
                             [0.0, 0.0, 0.0, 0.0, 4.0**1.5]]).T
        returned = spline(variable=vars, knots=[10, 16], power=1.5, restricted=False)
        npt.assert_allclose(returned, expected)

        # Spline setup 4
        expected = np.array([[0.0, 0.0, 0.0, 5.0, 6.0], ]).T
        returned = spline(variable=vars, knots=[10, 16], power=1, restricted=True)
        npt.assert_allclose(returned, expected)

        # Spline setup 5
        expected = np.array([[0.0, 0.0, 5.0**2, 10.0**2, 15.0**2 - 4.0**2], ]).T
        returned = spline(variable=vars, knots=[5, 16], power=2, restricted=True)
        npt.assert_allclose(returned, expected)

    def test_adm_error_noknots(self, design_matrix):
        # Testing error when dictionary with no knots given
        specs = [None,
                 {"power": 2},
                 None]
        with pytest.raises(ValueError, match="`knots` must be"):
            additive_design_matrix(X=design_matrix,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_error_negknots(self, design_matrix):
        # Testing error when dictionary with negative number of knots given
        specs = [None,
                 {"knots": -2},
                 None]
        with pytest.raises(ValueError, match="`knots` must be a pos"):
            additive_design_matrix(X=design_matrix,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_error_misalign(self, design_matrix):
        # Testing error matrix is misaligned
        # Testing warning for extra spline arguments
        specs = [None,
                 {"knots": 2},
                 None]
        with pytest.raises(ValueError, match="number of input"):
            additive_design_matrix(X=design_matrix.T,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_warn_extras(self, design_matrix):
        # Testing warning for extra spline arguments
        specs = [None,
                 {"knots": 2, "extra": 4},
                 None]
        with pytest.warns(UserWarning, match="following keys"):
            additive_design_matrix(X=design_matrix,
                                   specifications=specs,
                                   return_penalty=False)

    def test_adm_defaults(self, design_matrix):
        # Testing default additive design matrix specifications
        specs = [None,
                 {"knots": [5, 16]},
                 {"knots": [2, 4]}]
        expected_matrix = np.array([[1, 1, 1, 1, 1],
                                    [1, 5, 10, 15, 20],
                                    [0.0, 0.0, 5.0**3, 10.0**3, 15.0**3 - 4.0**3],
                                    [1, 2, 3, 4, 5],
                                    [0, 0, 1, 2**3, 3**3 - 1]]).T
        expected_penalty = np.array([0, 0, 0, 0, 0])
        returned_matrix, returned_penalty = additive_design_matrix(X=design_matrix,
                                                                   specifications=specs,
                                                                   return_penalty=True)
        npt.assert_allclose(returned_matrix, expected_matrix)
        npt.assert_allclose(returned_penalty, expected_penalty)

    def test_adm_v1(self, design_matrix):
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False},
                 None]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4],
                             [1, 2, 3, 4, 5]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_v2(self, design_matrix):
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False},
                 {"knots": [2, 4], "power": 2, "natural": True}]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4],
                             [1, 2, 3, 4, 5],
                             [0, 0, 1, 4, 3**2 - 1]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_v3(self, design_matrix):
        # Testing non-ascending order correction for knots
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False},
                 {"knots": [4, 2], "power": 2, "natural": True}]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4],
                             [1, 2, 3, 4, 5],
                             [0, 0, 1, 4, 3**2 - 1]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_v4(self, design_matrix):
        # Testing number of generated rows
        specs = [None,
                 {"knots": [16, ], "power": 2, "natural": False},
                 {"knots": [2, 4], "power": 1, "natural": False}]
        expected = np.array([[1, 1, 1, 1, 1],
                             [1, 5, 10, 15, 20],
                             [0, 0, 0, 0, 4**2],
                             [1, 2, 3, 4, 5],
                             [0, 0, 1, 2, 3],
                             [0, 0, 0, 0, 1]]).T
        returned = additive_design_matrix(X=design_matrix,
                                          specifications=specs,
                                          return_penalty=False)
        npt.assert_allclose(returned, expected)

    def test_adm_penalty(self, design_matrix):
        specs = [None,
                 {"knots": [16, ], "power": 1, "natural": False, "penalty": 3},
                 {"knots": [2, 3, 4], "power": 2, "natural": True, "penalty": 5}]
        expected = np.array([0, 0, 3, 0, 5, 5])
        design, returned = additive_design_matrix(X=design_matrix,
                                                  specifications=specs,
                                                  return_penalty=True)
        npt.assert_allclose(returned, expected)
