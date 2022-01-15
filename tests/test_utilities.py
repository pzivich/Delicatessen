import pytest
import numpy as np
import numpy.testing as npt

from delicatessen.utilities import inverse_logit, logit, partial_derivative


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

    def test_partial_derivative(self):
        """Checks the partial derivative numerical approximations with known solution
        """
        def function(arg):
            x, y = arg
            return (x**2) * (y**3),

        def partial_x_derivative(x, y):
            return 2 * x * (y**3)

        def partial_y_derivative(x, y):
            return 3 * (x**2) * (y**2)

        loc_x = 3
        loc_y = -1

        # Checking partial derivative for X
        dx1 = partial_derivative(func=function, var=0, point=np.array([loc_x, loc_y]), output=0,
                                 dx=1, order=5)
        npt.assert_allclose(dx1, partial_x_derivative(loc_x, loc_y))

        # Checking partial derivative for Y
        dy1 = partial_derivative(func=function, var=1, point=np.array([loc_x, loc_y]), output=0,
                                 dx=1, order=5)
        npt.assert_allclose(dy1, partial_y_derivative(loc_x, loc_y))
