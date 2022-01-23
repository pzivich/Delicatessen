.. image:: images/delicatessen_header.png

Delicatessen
=====================================

``delicatessen`` is a one-stop shop for all your sandwich (variance) needs. This Python 3.6+ library supports
M-Estimation, which is a general statistical framework for estimating unknown parameters. If you are an R user, I
highly recommend the R library ``geex`` (`Saul & Hudgens (2020) <https://bsaul.github.io/geex/>`_).
``delicatessen`` supports a variety of pre-built estimating equations as well as custom, user-specified estimating
equations.

Here, we provide a brief overview of M-Estimation. For a more detailed and precise introduction, please refer to
Stefanski & Boos (2002) or Boos & Stefanski (2013). M-Estimation was developed to study the large sample properties of
robust statistics. However, many common large-sample statistics can be expressed with estimating equations, so
M-Estimation provides a unified structure and a streamlined approach to estimation. Let the parameter of interest be
the vector :math:`\theta = (\theta_1, \theta_2, ..., \theta_v)` and data is observed for :math:`n` independent units
:math"`Z_1, Z_2, …, Z_n`. Then :math:`\theta` can often be expressed as the solution to the vector equation
:math:`\sum_{i=1}^{n} \psi(Z_i,\theta) = 0` where :math:`\psi(\dot)` is a known :math:`v \times 1`-function that does
not depend on observation :math:`i` or :math:`n`. To compute point estimates, the vector equation is solved using the
:math:`n` units. M-Estimators further provides a convenient and automatic method of calculating large-sample variance
estimators via the . The sandwich variance estimator is:

.. math::

    V_n(Y,\hat{\theta}) = A_n(Y,\hat{\theta})^{-1} B_n(Y,\hat{\theta}) \left(A_n(Y,\hat{\theta})^{-1}\right)^T

where

.. math::

    A_n(Y,\hat{\theta}) = n^{-1} \sum_i^n - \psi'(Y_i, \hat{\theta})

where the prime indicates the first derivative, and

.. math::

    B_n(Y,\hat{\theta}) = n^{-1} \sum_i^n \psi(Y_i, \hat{\theta}) \psi(Y_i, \hat{\theta})^T

A key advantage of the sandwich variance estimator is that it is less computationally demanding compared to other
procedures, like bootstrapping.

While M-Estimation is a general approach, widespread application is hindered by the corresponding derivative and matrix
calculations. For complex estimating equations, these calculations can be especially tedious. To circumvent these
barriers, ``delicatessen`` automates the M-Estimator. We can let the computer do all that hard work of finding the
roots and numerically approximating the derivatives for us.

The following description is a high-level description of the process. The user provides their estimating equation(s) to
the ``MEstimator`` class object. Next, the ``MEstimator`` object solves for :math:`\theta` using a root-finding
algorithm. Root-finding algorithms implemented in SciPy, as well as user-provided root-finding algorithms, are
supported. After successful completion of the root-finding step, the bread is computed by numerically approximating the
partial derivatives and the filling is calculated via the requisite matrix multiplication using NumPy. Finally, the
sandwich variance is computed.


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

To install ``delicatessen``, use the following command in terminal or command prompt

``python -m pip install delicatessen``

Only two dependencies are necessary for ``delicatessen`` (both of which you will likely have already installed): NumPy
and SciPy.

To replicate the tests in ``tests/`` you will need to install ``statsmodels`` and ``pytest`` (but this is not necessary
for general use of the package).


Code and Issue Tracker
-----------------------------

Please report bugs, issues, or feature requests on GitHub
at `pzivich/Delicatessen <https://github.com/pzivich/Delicatessen/>`_.

Otherwise, you may contact us via email (gmail: zivich.5) or on Twitter (@PausalZ)

References
-----------------------------
Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. *The American Statistician*, 56(1), 29-38.

Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.

Saul BC, & Hudgens MG. (2020). The Calculus of M-Estimation in R with geex. *Journal of Statistical Software*,
92(2).