####################################################################################################################
# Tests for loading provided data
####################################################################################################################

import pytest

from delicatessen.data import load_shaq_free_throws, load_inderjit, load_breast_cancer


class TestDataSets:

    def test_shaq_ft_data(self):
        d = load_shaq_free_throws()  # Loading the data
        assert d.shape[0] == 23      # Checking number of rows
        assert d.shape[1] == 3       # Checking number of columns

    def test_inderjit(self):
        d = load_inderjit()          # Loading the data
        assert d.shape[0] == 24      # Checking number of rows
        assert d.shape[1] == 2       # Checking number of columns

    def test_bcancer(self):
        d = load_breast_cancer()     # Loading the data
        assert d.shape[0] == 45      # Checking number of rows
        assert d.shape[1] == 3       # Checking number of columns

