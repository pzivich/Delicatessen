.. image:: images/delicatessen_header.png

Delicatessen
=====================================

`delicatessen` is a one-stop shop for all your sandwich (variance) needs. This Python 3.6+ package supports a
generalization of M-Estimation theory.

Here, we provide a brief overview of M-estimation theory. For a more detailed and formal introduction to M-estimation,
I highly recommend chapter 7 of Boos & Stefanski (2013). M-estimation is a generalization of robust inference (here
robust refers to allowing for misspecification of secondary assumptions does not invalidate inference) for
likelihood-based methods to a general context. *M-estimators* are solutions to estimating equations, where a generic
estimating equation takes the form of

.. math::

     \sum_i^n \psi(Z_i, \theta) = 0

where `Z` is independent observations (these need not be identically distributed). The parameter(s) of interest are
the *b*-dimensional vector :math:`\theta`, and :math:`\psi` is a *b*-by-1 function that is known. A large number of
consistent and asymptotically normal statistics can be put into the M-Estimation framework. Some examples include:
mean, regression, delta method, and many others.

To apply the M-Estimator, we solve for :math:`\theta` given the data and stacked estimating equations. This is similar
to other approaches, but M-estimation requires the equations take the form provided above. The key advantage of
M-Estimators is the straightforward estimation of the variance following from this framework, under suitable regularity
conditions (and whatever else is needed). Specifically, M-Estimation provides the following sandwich variance estimator:

.. math::

    V_n(Y,\hat{\theta}) = A_n(Y,\hat{\theta})^{-1} B_n(Y,\hat{\theta}) \left(A_n(Y,\hat{\theta})^{-1}\right)^T

where

.. math::

    A_n(Y,\hat{\theta}) = n^{-1} \sum_i^n - \psi'(Y_i, \hat{\theta})

where the prime indicates the first derivative, and

.. math::

    B_n(Y,\hat{\theta}) = n^{-1} \sum_i^n \psi(Y_i, \hat{\theta}) \psi(Y_i, \hat{\theta})^T

Therefore, we have a relatively simple way to calculate the variance for a large class of different statistics.
Additionally, this variance estimator is robust.

While M-Estimation is a powerful tool, the derivatives and matrix algebra can quickly become unwieldy. This is where
`delicatessen` comes in. `delicatessen` takes stacked estimating equations and data and works through all the necessary
calculations. Therefore, M-Estimation can be more widely adopted without needing to solve every derivative for your
particular problem. We can let the computer do all that hard work of finding the roots and numerically approximating the
derivatives for us.

Contents:
-------------------------------------

.. toctree::
  :maxdepth: 3

  Examples.rst
  Built-in Equations <Built-in Equations.rst>
  Custom Equations <Custom Equations.rst>
  Optimization Advice <Optimization Advice.rst>
  Reference/index
  Create a GitHub Issue <https://github.com/pzivich/Deli/issues>

Installation:
-------------

To install `delicatessen`, use the following command in terminal or command prompt

``python -m pip install delicatessen``

Only two dependencies are necessary for `delicatessen` (both of which you will likely have already installed): NumPy
and SciPy.

To replicate the tests in `tests/` you will need to install `statsmodels` and `pytest` (but this is not necessary for
general use of the package).


Code and Issue Tracker
-----------------------------

Available on Github `pzivich/Deli <https://github.com/pzivich/Delicatessen/>`_
Please report bugs, issues, and feature requests there.

Also feel free to contact me via email (gmail: zivich.5) or on Twitter (@PausalZ)

References
-----------------------------
Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. *The American Statistician*, 56(1), 29-38.

Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.
