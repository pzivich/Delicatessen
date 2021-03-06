Optimization Advice
====================

This section is meant to provide some guidance if you have trouble optimizing the estimating equations. Particularly,
this will hopefully help ``RuntimeErrors`` due to failed optimizations.

A weakness of ``delicatessen`` is that it does not have the fastest or most robust optimization routines. This is the
cost of the flexibility of the general M-Estimator (and a cost that I believe to be warranted). When :math:`\theta` only
consists of a few parameters, the optimizer will generally be robust. However, cases where :math:`\theta` consists of
many parameters is likely to occur (see survival curves). In these cases, ``delicatessen`` may need some help.

Below are a few recommendations on getting ``MEstimator`` to optimize without ``RuntimeErrors``.

Center initial values
---------------------

Optimization will be improved when the starting values (those provided in ``init``) are 'close' to the :math:`\theta`
values. However, we don't know what those values may be. Since the optimizer needs to look for the best values, it is
best to start it in the 'middle' of the possible space.

As an example, consider the mean estimating equation for a binary variable. The starting value could be specified as
0, 1, or any other number. However, we can be nice to the optimizer by providing 0.5 as the starting value, since 0.5 is
the middle of the possible space for a proportion.

For regression, starting values of 0 are likely to be preferred (without outside information).

If initial values are placed outside of the bounds of a particular :math:`\theta`, this can also break the optimization
procedure. Returning to the proportion, providing a starting value of -10 may cause the optimizer trouble, since
proportions are actually bound [0,1]. So make sure your initial values are (1) reasonable, and (2) within the bounds
of the measure :math:`\theta`.

Pre-wash initials
--------------------

In the case of stacked estimating equations made up of multiple regression models (e.g., g-computation, IPW, AIPW),
the regression models can be estimated outside of ``delicatessen``. Then the pre-optimized values can be passed as the
initial values. This 'pre-washing' of values allows the ``delicatessen`` optimizer to focus on values of :math:`\theta`
that can't be optimized outside.

This pre-washing approach is particularly useful for regression models, since more stable optimization strategies exist
for most regression implementations (i.e., most include the first derivative in the optimization, increasing speed and
stability). Pre-washing the initial values allows ``delicatessen`` to 'borrow' the strength of those optimizers.

Increase iterations
--------------------

If neither of those works, increasing the number of iterations is a good next place. By default, ``MEstimator``
goes to 1000 iterations (far beyond SciPy's default value). More iterations will increase the run-time of the program
but that extra time may be worth it if it means a successful optimization versus not.

Different optimizer
--------------------

By default, ``delicatessen`` uses the secant method available in ``scipy.optimize.newton``. However, ``delicatessen``
also supports algorithms in ``scipy.optimize.root``. I personally have had the most success with the 'secant' in, but
I have also seen the 'lm' optimization method perform well (some internal tests check with lm as well).

Non-smooth equations
--------------------
As mentioned in the examples, non-smooth estimating equations (e.g., percentiles, positive mean deviation, etc.) can be
difficult to optimize. In these cases, I have found ``solver='hybr'`` and increasing the tolerance (to the same order
as :math:`n`) help. Also be sure to update the numerical approximation values for the derivative (try something like
``dx=10, order=15``).

A warning
-------------------

Before ending this section, I want to emphasize that reducing the ``tolerance`` is not really advised (unless you are
in a specific scenario of a non-smooth function!). While it may allow the optimization routine to succeed, it only
allows the 'error' of the optimization to be greater. Therefore, the optimization will stop 'further' from the best
values. Do **not** use this approach to getting ``delicatessen`` to succeed in the optimization unless you are
absolutely sure that the new ``tolerance`` is within acceptable limits for your problem. The default is `1e-9`.
