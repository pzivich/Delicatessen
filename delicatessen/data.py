import numpy as np


def load_shaq_free_throws():
    """Load example data from Boos and Stefanski (2013) on Shaquille O'Neal free throws in the 2000 NBA playoffs
    (Table 7.1 on pg 324).

    Notes
    -----
    From left to right, the columns in the array correspond to:
        * game - game number
        * ft_success - free throws made during game
        * ft_attempt - free throws attempted during game

    Returns
    -------
    array :
        Returns a 24-by-2 NumPy array.

    References
    ----------
    Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
    (pp. 297-337). Springer, New York, NY.
    """
    d = np.array([[ 1,  4,  5],
                  [ 2,  5, 11],
                  [ 3,  5, 14],
                  [ 4,  5, 12],
                  [ 5,  2,  7],
                  [ 6,  7, 10],
                  [ 7,  6, 14],
                  [ 8,  9, 15],
                  [ 9,  4, 12],
                  [10,  1,  4],
                  [11, 13, 27],
                  [12,  5, 17],
                  [13,  6, 12],
                  [14,  9,  9],
                  [15,  7, 12],
                  [16,  3, 10],
                  [17,  8, 12],
                  [18,  1,  6],
                  [19, 18, 39],
                  [20,  3, 13],
                  [21, 10, 17],
                  [22,  1,  6],
                  [23,  3, 12], ])
    return d


def load_inderjit():
    """Load example data from Inderjit et al. (2002) on the dose-response of herbicide on perennial ryegrass growth.

    Notes
    -----
    From left to right, the columns in the array correspond to:
        * response - ryegrass root length
        * dose - herbicide dose

    Returns
    -------
    array :
        Returns a 24-by-2 NumPy array.

    References
    ----------
    Inderjit, Streibig JC, & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
    allelopathy research. *Physiologia Plantarum*, 114(3), 422-428.
    """
    d = np.array([[7.5800000,  0.00],
                  [8.0000000,  0.00],
                  [8.3285714,  0.00],
                  [7.2500000,  0.00],
                  [7.3750000,  0.00],
                  [7.9625000,  0.00],
                  [8.3555556,  0.94],
                  [6.9142857,  0.94],
                  [7.7500000,  0.94],
                  [6.8714286,  1.88],
                  [6.4500000,  1.88],
                  [5.9222222,  1.88],
                  [1.9250000,  3.75],
                  [2.8857143,  3.75],
                  [4.2333333,  3.75],
                  [1.1875000,  7.50],
                  [0.8571429,  7.50],
                  [1.0571429,  7.50],
                  [0.6875000, 15.00],
                  [0.5250000, 15.00],
                  [0.8250000, 15.00],
                  [0.2500000, 30.00],
                  [0.2200000, 30.00],
                  [0.4400000, 30.00], ])
    return d


def load_robust_regress(outlier=True):
    """Load illustrative example of robust linear regression published in Zivich et al. (2022).

    Parameters
    ----------
    outlier : bool, optional
        Whether to induce the outlier (``True``) or not (``False``).

    Returns
    -------
    array :
        Returns a 15-by-2 NumPy array.

    References
    ----------
    Zivich PN, Klose M, Cole SR, Edwards JK, & Shook-Sa BE. (2022). Delicatessen: M-estimation in Python.
    *arXiv:2203.11300*.
    """
    height = [168.519, 166.944, 164.327, 164.058, 166.212, 167.358,
              165.244, 169.352, 159.386, 166.953, 163.876,
              164.245, 165.061, 162.876, 164.185]
    weight = [67.634, 67.418, 63.394, 66.18, 65.573, 67.66, 64.592,
              68.4, 62.043, 67.093, 65.202, 64.328, 64.754,
              62.179, 64.716]
    if outlier:
        weight[8] += 3

    return np.array([height, weight]).T
