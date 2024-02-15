####################################################################################################################
# Tests for built-in estimating equations -- dose-response
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_2p_logistic, ee_3p_logistic, ee_4p_logistic, ee_effective_dose_delta
from delicatessen.data import load_inderjit


np.random.seed(236461)


class TestEstimatingEquationsDoseResponse:

    def test_4pl(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # library(lmtest)
        # data(ryegrass)
        # rgll4 = drm(rootl ~ conc, data=ryegrass, fct=LL.4())
        # coeftest(rgll4, vcov=sandwich)
        # Results from Ritz et al. and R
        comparison_theta = np.asarray([0.48141, 3.05795, 2.98222, 7.79296])
        comparison_var = np.asarray([0.12779, 0.26741, 0.47438, 0.15311])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_4p_logistic(theta=theta, X=dose_data, y=resp_data)

        # Optimization procedure
        estr = MEstimator(psi, init=[0, 2, 1, 10])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)**0.5, comparison_var, atol=1e-4)

    def test_3pl(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # library(lmtest)
        # data(ryegrass)
        # rgll3 = drm(rootl ~ conc, data=ryegrass, fct=LL.3())
        # coeftest(rgll3, vcov=sandwich)
        # R optimization from Ritz et al.
        comparison_theta = np.asarray([3.26336, 2.47033, 7.85543])
        comparison_var = np.asarray([0.26572, 0.29238, 0.15397])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_3p_logistic(theta=theta, X=dose_data, y=resp_data,
                                  lower=0)

        # Optimization procedure
        estr = MEstimator(psi, init=[2, 1, 10])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)**0.5, comparison_var, atol=1e-5)

    def test_2pl(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # library(lmtest)
        #
        # data(ryegrass)
        # rgll2 = drm(rootl ~ conc, data=ryegrass, fct=LL.2(upper=8))
        # coeftest(rgll2, vcov=sandwich)
        # R optimization from Ritz et al.
        comparison_theta = np.asarray([3.19946, 2.38220])
        comparison_var = np.asarray([0.24290, 0.27937])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            return ee_2p_logistic(theta=theta, X=dose_data, y=resp_data,
                                  lower=0, upper=8)

        # Optimization procedure
        estr = MEstimator(psi, init=[2, 1])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta, comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)**0.5, comparison_var, atol=1e-5)

    def test_3pl_ed_delta(self):
        # Compares against R's drc library:
        #
        # library(drc)
        # library(sandwich)
        # data(ryegrass)
        # rgll3 = drm(rootl ~ conc, data=ryegrass, fct=LL.3())
        # ED(rgll3, c(5, 10, 50), interval='delta', vcov=sandwich)
        # R optimization from Ritz et al.
        comparison_theta = np.asarray([0.99088, 1.34086, 3.26336])
        comparison_var = np.asarray([0.12397, 0.13134, 0.26572])

        d = load_inderjit()
        dose_data = d[:, 1]
        resp_data = d[:, 0]

        def psi(theta):
            lower_limit = 0
            pl3 = ee_3p_logistic(theta=theta, X=dose_data, y=resp_data, lower=lower_limit)
            ed05 = ee_effective_dose_delta(theta[3], y=resp_data, delta=0.05,
                                           steepness=theta[0], ed50=theta[1],
                                           lower=lower_limit, upper=theta[2])
            ed10 = ee_effective_dose_delta(theta[4], y=resp_data, delta=0.10,
                                           steepness=theta[0], ed50=theta[1],
                                           lower=lower_limit, upper=theta[2])
            ed50 = ee_effective_dose_delta(theta[5], y=resp_data, delta=0.50,
                                           steepness=theta[0], ed50=theta[1],
                                           lower=lower_limit, upper=theta[2])
            return np.vstack((pl3,
                              ed05,
                              ed10,
                              ed50))

        # Optimization procedure
        estr = MEstimator(psi, init=[2, 1, 10, 1, 1, 2])
        estr.estimate(solver='lm')

        # Checking mean estimate
        npt.assert_allclose(estr.theta[-3:], comparison_theta, atol=1e-5)

        # Checking variance estimate
        npt.assert_allclose(np.diag(estr.variance)[-3:]**0.5, comparison_var, atol=1e-5)
