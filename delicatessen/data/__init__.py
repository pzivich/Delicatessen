import numpy as np
import pandas as pd
from pkg_resources import resource_filename


def load_shaq_free_throws():
    """Load example data from Boos and Stefanski (2013) on Shaquille O'Neal free throws in the 2000 NBA playoffs
    (Table 7.1 on pg 324).

    Notes
    -----
    * game - game number
    * ft_success - free throws made during game
    * ft_attempt - free throws attempted during game
    * ft_prop - free throw success proportion

    Returns
    -------
    data
    """
    d = pd.read_csv(resource_filename('delicatessen', 'data/shaq_free_throws.csv'))
    d['ft_prop'] = d['ft_success'] / d['ft_attempt']
    return d

