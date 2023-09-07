Basics
=====================================

Here, the basics of M-estimator will be reviewed. An M-estimator, :math:`\hat{\theta}`, is defined as the solution to
the estimating equation

.. math::

    \sum_{i=1}^{n} \psi(O_i, \hat{\theta}) = 0


where :math:`\psi` is a known :math:`v \times 1`-dimension estimating function, :math:`O_i` indicates the observed unit
:math:`i \in \{1,...,n\}`, and the parameters are the vector :math:`\theta = (\theta_1, \theta_2, ..., \theta_v)`. Note
that :math:`v` is finite-dimensional and the number of parameters matches the dimension of the estimating functions.

Point Estimation
-------------------------------
To implement the point estimation of :math:`\theta`, we use a *root-finding* algorithm. Root-finding algorithms are
procedures for finding the zeroes (i.e., roots) of an equation. This is accomplished in ``delicatessen`` by using
SciPy's root-finding algorithms.

Variance Estimation
-------------------------------
To estimate the variance for :math:`\theta`, the M-estimator uses the empirical sandwich variance estimator:

.. math::

    V_n(O,\hat{\theta}) = B_n(O,\hat{\theta})^{-1} F_n(O,\hat{\theta}) \left(B_n(O,\hat{\theta})^{-1}\right)^T

where the 'bread' is

.. math::

    B_n(O,\hat{\theta}) = n^{-1} \sum_{i=1}^n - \psi'(O_i, \hat{\theta})

where the :math:`\psi'` indicates the partial derivative, and the 'filling' is

.. math::

    F_n(O, \hat{\theta}) = n^{-1} \sum_{i=1}^n \psi(O_i, \hat{\theta}) \psi(O_i, \hat{\theta})^T

The sandwich variance requires finding the derivative of the estimating functions and some matrix algebra. Again, we
can get the computer to complete all these calculations for us. For the derivative, ``delicatessen`` offers two
options. First, the derivatives can be numerically approximated using the central difference method. This is done using
SciPy's ``approx_fprime`` functionality. As of ``v2.0``, the derivatives can also be computed using forward-mode
automatic differentiation. This approach provides the exact derivative (as opposed to an approximation). This is
implemented by-hand in ``delicatessen`` via operator overloading. Finally, we use forward-mode because there is no
time advantage of backward-mode because the Jacobian is square.

After computing the derivatives, the filling is computed via a dot product. The bread is then inverted using NumPy.
Finally, the bread and filling matrices are combined via dot products.

This introduction has all been a little abstract. In the following Examples section, we will see how M-estimators can
be used to address specific estimation problems.
