import pytest
import numpy as np
import pandas as pd
import numpy.testing as npt

from delicatessen import load_shaq_free_throws


class TestDataSets:

    def test_shaq_ft_data(self):
        d = load_shaq_free_throws()  # Loading the data
        assert d.shape[0] == 23      # Checking number of rows
        assert d.shape[1] == 4       # Checking number of columns
        npt.assert_equal(d.columns,  # Checking column names
                         np.array(['game', 'ft_success', 'ft_attempt', 'ft_prop']))
