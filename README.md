# Delicatessen

[![PyPI version](https://badge.fury.io/py/delicatessen.svg)](https://badge.fury.io/py/delicatessen)
[![Documentation Status](https://readthedocs.org/projects/deli/badge/?version=latest)](https://deli.readthedocs.io/en/latest/?badge=latest)

The one-stop sandwich (variance) shop in Python. `delicatessen` is a Python 3.6+ library for the generalized calculus 
of M-estimation. 

**Notice:** `delicatessen` is in _beta_ test currently. So please feel free to try out the functionalities
but the releases may not be currently stable until the true release. I would not recommend on integrating
`delicatessen` into your full workflow at this time.

Here, we provide a brief overview of M-estimation theory. For a more detailed and formal introduction to M-estimation,
I highly recommend chapter 7 of Boos & Stefanski (2013). M-estimation is a generalization of robust inference (here
robust refers to allowing for misspecification of secondary assumptions does not invalidate inference) for
likelihood-based methods to a general context. *M-estimators* are solutions to estimating equations. A large number of 
consistent and asymptotically normal statistics can be put into the M-Estimation framework. Some examples include: 
mean, regression, delta method, and many others.

To apply the M-Estimator, we solve the stacked estimating equations using observed data. This is similar to other 
approaches, but the key advantage of M-Estimators is the straightforward estimation of the variance following from this 
framework, under suitable regularity conditions. Specifically, the sandwich variance.

While M-Estimation is a powerful tool, the derivatives and matrix algebra can quickly become unwieldy. This is where 
`delicatessen` comes in. `delicatessen` takes stacked estimating equations and data and works through all the necessary 
calculations. Therefore, M-Estimation can be more widely adopted without needing to solve every derivative for your 
particular problem. We can let the computer do all that hard work of finding the roots, numerically approximating 
the derivatives, and doing the matrix calculations for us.

In addition to implementing a general M-estimator, `delicatessen` also comes with a variety of built-in estimating 
equations. See the [delicatessen website](https://deli.readthedocs.io/en/latest/) for the full set of available
estimating equations and how to use them.

# Installation

## Installing:

You can install via `python -m pip install delicateseen`

## Dependencies:

There are only two dependencies: `numpy`, `scipy`

To replicate the tests located in `tests/`, you will additionally need to install: `panda`, `statsmodels`, and `pytest`

While `delicatessen` will likely work with older versions of NumPy and SciPy (this has not been formally tested),
it is recommended to use `numpy >= 1.18.0` and `scipy >= 1.4.0`. There has been no reported testing on versions
earlier than these.

# Getting started

To demonstrate `delicatessen`, below is a simple demonstration of calculating the mean for the following data

```python
import numpy as np
y = np.array([1, 2, 3, 1, 4, 1, 3, -2, 0, 2])
```

Loading the M-estimator functionality from deli, building the estimating equation, and 

```python
from delicatessen import MEstimator

def psi(theta):
    return y - theta

mestimate = MEstimator(psi, init=[0, ])
mestimate.estimate()

print(mestimate.theta)     # Estimate of the mean
print(mestimate.variance)  # Variance estimator for the mean
```

For recommendations and general advice on using `delicatessen`, see the full documentation and worked examples 
available at [delicatessen website](https://deli.readthedocs.io/en/latest/).
