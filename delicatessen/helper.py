import numpy as np


def convert_survival_measures(survival, hazard, measure):
    r"""Function to convert between different survival analysis metrics. This function takes input survival and hazard
    values for a specific time (or for a vector of times) and converts them to the desired metric. Let :math:`S(t)` be
    survival and :math:`h(t)` be the hazard at time :math:`t`. The different metrics are then obtained via

    .. math::

        S(t) = S(t)              \\
        F(t) = 1 - S(t)          \\
        H(t) = - \log(S(t))      \\
        h(t) = h(t)              \\
        f(t) = h(t) \times S(t)

    where :math:`F(t)` is the risk or cumulative distribution function, :math:`H(t)` is the cumulative hazard, and
    :math:`f(t)` is the density function.

    This functionality is principally intended to be used internally to manage conversions between

    Parameters
    ----------
    survival : float, ndarray, list, vector
        Survival at a designated time or times.
    hazard : float, ndarray, list, vector
        Hazard at a designated time or times.
    measure : str
        Survival analysis metric to return.

    Returns
    -------
    array :
        Returns a float or NumPy array of the transformed metric(s) depending on the input types.

    Examples
    --------
    The following illustrates use of ``convert_survival_measures`` to convert between measures

    >>> from delicatessen.helper import convert_survival_measures

    Different measures can be computed via

    >>> convert_survival_measures(survival=0.75, hazard=0.006, measure='survival')
    >>> convert_survival_measures(survival=0.75, hazard=0.006, measure='risk')
    >>> convert_survival_measures(survival=0.75, hazard=0.006, measure='density')
    >>> convert_survival_measures(survival=0.75, hazard=0.006, measure='hazard')
    >>> convert_survival_measures(survival=0.75, hazard=0.006, measure='cumulative_hazard')

    Inputs can also be vectors of matching lengths

    >>> survival_t = [0.75, 0.50, 0.40]
    >>> hazard_t = [0.006, 0.004, 0.001]
    >>> convert_survival_measures(survival=survival_t, hazard=hazard_t, measure='survival')

    References
    ----------
    Collett D. (2015). Survival analysis. In: Modelling Survival Data in Medical Research. CRC press. pg 10-14
    """
    survival = np.asarray(survival)
    hazard = np.asarray(hazard)

    # Converting survival or hazard to desired measure
    if measure == "survival":
        metric = survival  # S(t) = S(t)
    elif measure in ["risk", "cdf", "cumulative_distribution_function"]:
        metric = 1 - survival  # F(t) = 1 - S(t)
    elif measure in ["cumulative_hazard", "chazard"]:
        metric = -1 * np.log(survival)  # H(t) = -log(S(t))
    elif measure == "hazard":
        metric = hazard  # h(t) = h(t)
    elif measure in ["density", "pdf"]:
        metric = hazard * survival  # f(t) = h(t) * S(t)
    else:
        raise ValueError("The measure '" + str(measure) + "' is not supported. "
                         "Please select one of the following: survival, density, risk, hazard, cumulative_hazard.")
    return metric
