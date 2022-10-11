.. image:: images/delicatessen_header.png

Delicatessen
=====================================

``delicatessen`` is a one-stop shop for all your sandwich (variance) needs. This Python 3.6+ library supports
M-estimation, which is a general statistical framework for estimating unknown parameters. If you are an R user, I
recommend looking into ``geex`` (`Saul & Hudgens (2020) <https://bsaul.github.io/geex/>`_).
``delicatessen`` supports a variety of pre-built estimating equations as well as custom, user built estimating
equations.

Here, we provide a brief overview of M-Estimation. For a more detailed, please refer to
Stefanski & Boos (2002) or Boos & Stefanski (2013). M-estimation was developed to study the large sample properties of
robust statistics. However, many common large-sample statistics can be expressed with estimating equations, so
M-estimation provides a unified structure and a streamlined approach to estimation. Let the parameter of interest be
the vector :math:`\theta = (\theta_1, \theta_2, ..., \theta_v)` and data is observed for :math:`n` independent units
:math:`Z_1, Z_2, …, Z_n`. An M-estimator, :math:`\theta` is the solution to the vector equation
:math:`\sum_{i=1}^{n} \psi(Z_i, \hat{\theta}) = 0` where :math:`\psi` is a known :math:`v \times 1`-function that does
not depend on observation :math:`i` or :math:`n`. M-estimators further provides a convenient and automatic method of
calculating large-sample variance estimators via the sandwich variance

.. math::

    V_n(Z,\hat{\theta}) = B_n(Z,\hat{\theta})^{-1} F_n(Z,\hat{\theta}) \left(B_n(Z,\hat{\theta})^{-1}\right)^T

where the 'bread' is

.. math::

    B_n(Z,\hat{\theta}) = n^{-1} \sum_{i=1}^n - \psi'(Z_i, \hat{\theta})

where the :math:`\psi'` indicates the partial derivative, and the 'filling' is

.. math::

    F_n(Z,\hat{\theta}) = n^{-1} \sum_{i=1}^n \psi(Z_i, \hat{\theta}) \psi(Z_i, \hat{\theta})^T

While M-Estimation is a general approach, widespread application is hindered by the corresponding derivative and matrix
calculations. To circumvent these barriers, ``delicatessen`` automates M-estimators using numerical approximation
methods.

The following description is a high-level overview. The user provides their estimating equation(s) to
the ``MEstimator`` class object. Next, the ``MEstimator`` object solves for :math:`\hat{\theta}` using a root-finding
algorithm. After successful completion of the root-finding step, the bread is computed by numerically approximating the
partial derivatives and the filling is calculated. Finally, the sandwich variance is computed.

Installation:
-------------

To install ``delicatessen``, use the following command in terminal or command prompt

``python -m pip install delicatessen``

Only two dependencies for ``delicatessen`` are: NumPy, SciPy.

While pandas is not necessary, several examples are demonstrated with pandas for ease of data management. To replicate
the tests in ``tests/`` you will also need to install ``statsmodels`` and ``pytest`` (but this is not necessary for
general use of the package).

Citation:
-------------
Please use the following citation for ``delicatessen``:
Zivich PN, Klose M, Cole SR, Edwards JK, & Shook-Sa BE. (2022). Delicatessen: M-Estimation in Python.
*arXiv preprint arXiv:2203.11300*.

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


Code and Issue Tracker
-----------------------------

Please report bugs, issues, or feature requests on GitHub
at `pzivich/Delicatessen <https://github.com/pzivich/Delicatessen/>`_.

Otherwise, you may contact us via email (gmail: zivich.5) or on Twitter (@PausalZ)

References
-----------------------------

Zivich PN, Klose M, Cole SR, Edwards JK, & Shook-Sa BE. (2022). Delicatessen: M-Estimation in Python.
*arXiv preprint arXiv:2203.11300*.

Stefanski LA, & Boos DD. (2002). The calculus of M-estimation. *The American Statistician*, 56(1), 29-38.

Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.

Saul BC, & Hudgens MG. (2020). The Calculus of M-Estimation in R with geex. *Journal of Statistical Software*,
92(2).