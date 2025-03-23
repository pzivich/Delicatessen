Optimization Advice
====================

This section is meant to provide some guidance if you have trouble with root-finding or minimization for a given set of
estimating functions.

A weakness of ``delicatessen`` is that it does not have the fastest or most robust routines for estimating statistical
parameters. This is the cost of the flexibility of the generic estimating equation implementation used (a cost that
I have found to be worthwhile).

Below are a few recommendations for finding :math:`\theta` with ``MEstimator`` or ``GMMEstimator``.

Center initial values
---------------------

Optimization will be improved when the starting values (those provided in ``init``) are 'close' to the :math:`\theta`
values. However, we don't know what those values may be. Since the optimization procedure needs to look for the best
values, it is best to start it in the 'middle' of the possible space. If initial values are placed outside of the
bounds of a particular :math:`\theta`, this can also break the optimization procedure. Returning to the proportion,
providing a starting value of -10 may cause trouble, since proportions are actually bound to [0,1]. So make sure your
initial values are (1) reasonable, and (2) within the bounds of the possible values of :math:`\theta`.

As an example, consider the mean estimating equation for a binary variable. The starting value could be specified as
0, 1, or any other float. However, we can be nice to the optimizer by providing 0.5 as the starting value, since
0.5 is the middle of the possible space for a proportion.

For regression, starting values of 0 are likely to be preferred in absence of additional information about a problem.

Pre-wash initials
--------------------

In the case of stacked estimating equations composed of multiple estimating functions (e.g., g-computation, IPW, AIPW),
some parameters can be estimated independent of the others. Then the pre-optimized values can be passed as the initial
values for the overall estimator. This 'pre-washing' of values allows the ``delicatessen`` optimization to focus on
values of :math:`\theta` that can't be optimized outside.

This pre-washing approach is particularly useful for regression models, since more stable optimization strategies exist
for most regression implementations. Pre-washing the initial values allows ``delicatessen`` to 'borrow' the strength of
more stable methods.

Finally, ``delicatessen`` offers the option to run the optimization procedure for a subset of the estimating functions.
Therefore, some parameters can be solved outside of the procedure and only the remaining subset can be searched for.
This option is particularly valuable when an estimator consists of hundreds of parameters.

Increase iterations
--------------------

If neither of those works, increasing the number of iterations is a good next place to start.

Different optimization
----------------------

By default, ``MEstimator`` uses the Levenberg-Marquardt for root-finding and ``GMMEstimator`` uses the BFGS
for minimization. However, ``delicatessen`` also supports other algorithms in ``scipy.optimize``. Additionally,
manually-specified root-finding algorithms can also be used. Other algorithms may have better operating
characteristics for some problems.

Non-smooth equations
--------------------
As mentioned elsewhere, non-smooth estimating equations (e.g., percentiles, positive mean deviation, etc.) can be
difficult to optimize. In general, it is best to avoid using ``delicatessen`` with non-smooth estimating equations.

If one must use ``delicatessen`` with non-smooth estimating equations, some tricks we have found helpful are to:
increasing the tolerance (to the same order as :math:`n`) or modify the optimization algorithm.

A warning
-------------------

Before ending this section, I want to emphasize that simply increasing ``tolerance`` is not really advised. While it may
allow the optimization routine to succeed, it only allows the 'error' of the optimization to be greater. Therefore,
the optimization will stop 'further' away from the zero of the esitmating equation. Do **not** use this approach to
getting ``delicatessen`` to succeed in the optimization unless you are absolutely sure that the new ``tolerance`` is
within acceptable computational error tolerance for your problem. The default is ``1e-9``.
