![delicatessen](docs/images/delicatessen_header.png)

# Delicatessen

![tests](https://github.com/pzivich/Delicatessen/actions/workflows/python-package.yml/badge.svg)
[![version](https://badge.fury.io/py/delicatessen.svg)](https://badge.fury.io/py/delicatessen)
[![docs](https://readthedocs.org/projects/deli/badge/?version=latest)](https://deli.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/delicatessen/month)](https://pepy.tech/project/delicatessen)

The one-stop sandwich (variance) shop in Python. `delicatessen` is a Python 3.8+ library for the generalized calculus
of M-estimation.

**Citation**: Zivich PN, Klose M, Cole SR, Edwards JK, & Shook-Sa BE. (2022). Delicatessen: M-Estimation in Python.
*arXiv:2203.11300* [stat.ME]

## M-Estimation and Estimating Equations

Here, we provide a brief overview of M-estimation theory. For a detailed introduction to M-estimation, see Chapter 7 of
Boos & Stefanski (2013). M-estimation is a generalization of likelihood-based methods. *M-estimators* are solutions to
estimating equations. To apply the M-estimator, we solve the estimating equations using observed data. This is similar
to other approaches, but the key advantage of M-Estimators is estimation of the variance via the sandwich variance.

While M-Estimation is a powerful tool, the derivatives and matrix algebra can quickly become unwieldy. This is where 
`delicatessen` comes in. `delicatessen` takes an array estimating equations and data, and solves for the parameter
estimates, numerically approximates the derivatives, and does the matrix calculations. Therefore, M-estimators can
be more widely adopted without by-hand calculations. We can let the computer do all the math for us.

`delicatessen` also comes with a variety of built-in estimating equations. See
the [delicatessen website](https://deli.readthedocs.io/en/latest/) for the full set of available estimating equations
and how to use them.

## Installation

### Installing:

You can install via `python -m pip install delicatessen`

### Dependencies:

The dependencies are: `numpy`, `scipy`

To replicate the tests located in `tests/`, you will additionally need to install: `panda`, `statsmodels`, and `pytest`

While versions of `delicatessen` prior to v1.0 were compatible with older versions of Python 3 and NumPy and SciPy, the
v1.0+ releases are only available for Python 3.8+ with NumPy v1.18.5+ and SciPy v1.9.0. This change was made to use
a better numerical approximation procedure for the derivative. If you want to use with older versions of those packages
or older versions of Python, install v0.6 instead.

## Getting started

Below is a simple demonstration of calculating the mean with `delicatessen`

```python
import numpy as np
y = np.array([1, 2, 3, 1, 4, 1, 3, -2, 0, 2])
```

Loading the M-estimator functionality, building the estimating equation, and printing the results to the console

```python
from delicatessen import MEstimator

def psi(theta):
    return y - theta[0]

estr = MEstimator(psi, init=[0, ])
estr.estimate()

print(estr.theta)     # Estimate of the mean
print(estr.variance)  # Variance estimator for the mean
```

For further details on using `delicatessen`, see the full documentation and worked examples available
at [delicatessen website](https://deli.readthedocs.io/en/latest/) or in the examples folder.


## References

Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.

Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. *The American Statistician*, 56(1), 29-38.

Zivich PN, Klose M, Cole SR, Edwards JK, & Shook-Sa BE. (2022). Delicatessen: M-Estimation in Python.
*arXiv preprint arXiv:2203.11300*.
