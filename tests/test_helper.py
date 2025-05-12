####################################################################################################################
# Tests for helper functions
####################################################################################################################

import pytest
import numpy as np
import numpy.testing as npt

from delicatessen.helper import convert_survival_measures


class TestHelperFuncs:

    def test_surv_convert_error_measure(self):
        with pytest.raises(ValueError, match="measure 'denfity' is"):
            convert_survival_measures(survival=0.5, hazard=0.005, measure='denfity')

    def test_surv_convert_error_shapes(self):
        survival = [0.90, 0.75, 0.5, 0.25]
        hazard = [0.001, 0.006, 0.005, 0.0015, 0.0023]
        with pytest.raises(ValueError, match="operands could"):
            convert_survival_measures(survival=survival, hazard=hazard, measure='density')

    def test_surv_convert_surv(self):
        measure = 'survival'

        # Floats
        survival = 0.5
        hazard = 0.005
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(survival,
                            output,
                            atol=1e-8)

        # Arrays
        survival = [0.90, 0.75, 0.5, 0.25, 0.10]
        hazard = [0.001, 0.006, 0.005, 0.0015, 0.0023]
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(survival,
                            output,
                            atol=1e-8)

    def test_surv_convert_haz(self):
        measure = 'hazard'

        # Floats
        survival = 0.5
        hazard = 0.005
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(hazard,
                            output,
                            atol=1e-8)

        # Arrays
        survival = [0.90, 0.75, 0.5, 0.25, 0.10]
        hazard = [0.001, 0.006, 0.005, 0.0015, 0.0023]
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(hazard,
                            output,
                            atol=1e-8)

    def test_surv_convert_risk(self):
        measure = 'risk'

        # Floats
        survival = 0.5
        hazard = 0.005
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(1 - survival,
                            output,
                            atol=1e-8)

        # Arrays
        survival = [0.90, 0.75, 0.5, 0.25, 0.10]
        hazard = [0.001, 0.006, 0.005, 0.0015, 0.0023]
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(1 - np.asarray(survival),
                            output,
                            atol=1e-8)

    def test_surv_convert_chaz(self):
        measure = 'cumulative_hazard'

        # Floats
        survival = 0.5
        hazard = 0.005
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(-np.log(survival),
                            output,
                            atol=1e-8)

        # Arrays
        survival = [0.90, 0.75, 0.5, 0.25, 0.10]
        hazard = [0.001, 0.006, 0.005, 0.0015, 0.0023]
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(-np.log(survival),
                            output,
                            atol=1e-8)

    def test_surv_convert_density(self):
        measure = 'density'

        # Floats
        survival = 0.5
        hazard = 0.005
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(survival * hazard,
                            output,
                            atol=1e-8)

        # Arrays
        survival = [0.90, 0.75, 0.5, 0.25, 0.10]
        hazard = [0.001, 0.006, 0.005, 0.0015, 0.0023]
        output = convert_survival_measures(survival=survival, hazard=hazard, measure=measure)
        npt.assert_allclose(np.asarray(survival) * np.asarray(hazard),
                            output,
                            atol=1e-8)
