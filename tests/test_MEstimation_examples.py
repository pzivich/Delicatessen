import pytest
import numpy as np
import numpy.testing as npt
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
        pass
