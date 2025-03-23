Basics
=====================================

Here, the basics of the provided estimation approaches are described.

M-estimator
-------------------------------

An M-estimator, :math:`\hat{\theta}`, is defined as the solution to the estimating equation

.. math::

    \sum_{i=1}^{n} \psi(O_i, \hat{\theta}) = 0


where :math:`\psi` is a known :math:`v \times 1`-dimension estimating function, :math:`O_i` indicates the observed unit
:math:`i \in \{1,...,n\}`, and the parameters are the vector :math:`\theta = (\theta_1, \theta_2, ..., \theta_v)`. Note
that :math:`v` is finite-dimensional and the number of parameters matches the dimension of the estimating functions.

In this equation, we use a *root-finding* algorithm to solve for :math:`\theta`. Root-finding algorithms are
procedures for finding the zeroes (i.e., roots) of an equation. This is accomplished in ``delicatessen`` by using
SciPy's root-finding algorithms.

GMM-estimator
-------------------------------

The generalized method of moments (GMM) estimator is instead defined as the solution to

.. math::

    \text{argmin}_{\theta} \left[ \sum_{i=1}^n \psi(O_i, \hat{\theta}) \right]
        \text{Q}
        \left[ \sum_{i=1}^n \psi(O_i, \hat{\theta}) \right]


where :math:`\text{\Q}` is a weight matrix. In general, the weight matrix begins as the identity matrix as implemented
in ``delicatessen``.

For this equation, we use a *minimization* algorithm to solve for :math:`\theta`. This is accomplished in
``delicatessen`` by using SciPy's minimization routines.

Note that solving this equation is equivalent to the M-estimator when the dimension of the parameters and estimating
equations match. However, the GMM estimator can also be used when there is more estimating equations than parameters.
This is referred to as *over-identification*. In these settings ``GMMEstimator`` can be used, but ``MEstimator``
cannot.

Variance Estimation
-------------------------------

Regardless of the chosen point-estimation strategy, the empirical sandwich variance estimator is used to estimate the
variance for :math:`\theta`:

.. math::

    V_n(O,\hat{\theta}) = B_n(O,\hat{\theta})^{-1} F_n(O,\hat{\theta}) \left(B_n(O,\hat{\theta})^{-1}\right)^T

where the 'bread' is

.. math::

    B_n(O,\hat{\theta}) = n^{-1} \sum_{i=1}^n - \nabla \psi(O_i, \hat{\theta})

where the :math:`\nabla` indicates the partial derivatives, and the 'filling' is

.. math::

    F_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^n \psi(O_i, \hat{\theta}) \psi(O_i, \hat{\theta})^T

The sandwich variance requires finding the derivative of the estimating functions and some matrix algebra. Again, we
can get the computer to complete all these calculations for us. For the derivative, ``delicatessen`` offers two
options: numerical approximation or forward-mode automatic differentiation.

After computing the derivatives, the filling is computed via a dot product. The bread is then inverted using NumPy.
If the pseudo-inverse is allowed, the Moore-Penrose inverse is used. Finally, the bread and filling matrices are
combined via dot products.

This introduction has all been a little abstract. In the Applied Examples, you can see how these estimators can be used
to address a variety of different computational problems.

Automatic Differentiation Caveats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are some caveats to the use of automatic differentiation. First, some NumPy functionalities are not fully
supported. For example, ``np.log(x, where=0<x)`` will result in an error since there is an attempt to evaluate a
log at zero internally. When using these specialty functions are necessary, it is better to use numerical approximation
for differentiation. The second is regarding discontinuities. Consider the following function :math:`f(x) = x**2` if
:math:`x \ge 1` and :math:`f(x) = 0` otherwise. Because of how automatic differentiation operates, the derivative at
:math:`x=1` will result in :math:`2x` (this is the same behavior as other automatic differentiation software, like
``autograd``).

Code and Issue Tracker
-----------------------------

Please report any bugs, issues, or feature requests on GitHub
at `pzivich/Delicatessen <https://github.com/pzivich/Delicatessen/>`_.

Otherwise, you may contact me via email (gmail: zivich.5).
