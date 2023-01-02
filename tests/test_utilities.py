import pytest
import numpy as np
import numpy.testing as npt

from delicatessen.utilities import inverse_logit, logit, robust_loss_functions


class TestFunctions:

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
