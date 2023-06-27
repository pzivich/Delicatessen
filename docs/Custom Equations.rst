Custom Estimating Equations
=====================================

A key advantages of ``delicatessen`` is the flexibility in the estimating equations that can be specified. Here, I
provide an overview and tips for how to build your own estimating equations using ``delicatessen``.

In general, it will be best if you find an paper or book that directly provides the estimating equation(s) to
you. Alternatively, if you can find the score function or gradient for a regression model, that is the corresponding
estimating equation. This section does *not* address how to derive your own  estimating equation(s). Rather, this
section provides information on how to translate an estimating equation into code that is compatible with
``delicatessen``, as ``delicatessen`` assumes you are giving it a valid estimating equation.

Building from scratch
-------------------------------------

First, we will go through the case of building an estimating equation completely from scratch. To do this, we will
go through an example with linear regression.

First, we have the estimating equation (which is the score function) provided in Boos & Stefanski (2013)

.. math::

    \sum_i^n (Y_i - X_i^T \beta) X_i = 0

We will demonstrate using the following simulated data set

.. code::

    np.random.seed(8091421)
    n = 200
    d = pd.DataFrame()
    d['X'] = np.random.normal(size=n)
    d['W'] = np.random.normal(size=n)
    d['C'] = 1
    d['Y'] = 5 + d['X'] + np.random.normal(size=n)


First, we can build the estimating equation using a for-loop where each ``i`` piece will be stacked together. While this
for-loop approach will be 'slow', it is often a good strategy to implement a for-loop version that is easier to debug
first.

Below calculates the estimating equation for each ``i`` in the for-loop. This function returns a stacked array of each
``i`` observation as a 3-by-n array. That array can then be passed to the ``MEstimator``

.. code::

    def psi(theta):
        # Transforming to arrays
        X = np.asarray(d[['C', 'X', 'W']])   # Design matrix
        y = np.asarray(d['Y'])               # Dependent variable
        beta = np.asarray(theta)[:, None]    # Parameters
        n = X.shape[0]                       # Number of observations

        # Where to store each of the resulting estimating functions
        est_vals = []

        # Looping through each observation from 1 to n
        for i in range(n):
            v_i = (y[i] - np.dot(X[i], beta)) * X[i]
            est_vals.append(v_i)

        # returning 3-by-n NumPy array
        return np.asarray(est_vals).T


We can then apply this estimating equation via

.. code::

    mest = MEstimator(psi, init=[0., 0., 0.])
    mest.estimate()

for which the coefficients match the coefficients from a ordinary least squares model (variance estimates may differ,
since most OLS software use the inverse of the information matrix to estimate the variance, which is equivalent to the
inverse of the bread matrix).

Here, we can vectorize the operations. The advantage of the vectorized-form is that it will run much faster. With some
careful experimentation, the following is a vectorized version. Remember that ``delicatessen`` is expecting a
3-by-n array to be given by the ``psi`` function in this example. Failure to provide this is a common mistake when
building custom estimating equations.

.. code::

    def psi(theta):
        X = np.asarray(d[['C', 'X', 'W']])    # Design matrix
        y = np.asarray(d['Y'])[:, None]       # Dependent variable
        beta = np.asarray(theta)[:, None]     # Parameters
        return ((y - np.dot(X, beta)) * X).T  # Computes all estimating functions


As before, we can run this chunk of code. Vectorizing (even parts of an estimating equation) can help to improve
run-times if you find a M-estimator taking too long to solve.

Building with basics
-------------------------------------

Instead of building everything from scratch, you can also piece together built-in estimating equations with your
custom estimating equations code. To demonstrate this, we will go through inverse probability weighting.

The inverse probability weighting estimator consists of four estimating equations: the difference between the weighted
means, the weighted mean under :math:`A=1`, the weighted mean under :math:`A=0`, and the propensity score model. We
can express this mathematically as

.. math::

    \sum_{i=1}^n
    \begin{bmatrix}
        (\theta_1 - \theta_2) - \theta_0 \\
        \frac{A_i \times Y_i}{\pi_i} - \theta_1 \\
        \frac{(1-A_i) \times Y_i}{1-\pi_i} - \theta_2 \\
        (A_i - \text{expit}(W_i^T \alpha)) W_i
    \end{bmatrix}
    = 0

where :math:`A` is the action of interest, :math:`Y` is the outcome of interest, and :math:`W` is the set of confounding
variables.

Rather than re-code the logistic regression model (to estimate the propensity scores), we will use the built-in
logistic regression functionality. Below is a stacked estimating equation for the inverse probability weighting
estimator above

.. code::

    def psi(theta):
        # Ensuring correct typing
        W = np.asarray(d['C', 'W'])     # Design matrix of confounders
        A = np.asarray(d['A'])          # Action
        y = np.asarray(y)               # Outocome
        beta = theta[3:]                # Regression parameters

        # Estimating propensity score
        preds_reg = ee_regression(theta=beta,        # Built-in regression
                                  X=W,               # Plug-in covariates for X
                                  y=A,               # Plug-in treatment for Y
                                  model='logistic')  # Specify logistic
        # Estimating weights
        pi = inverse_logit(np.dot(W, beta))          # Pr(A|W) using delicatessen.utilities

        # Calculating Y(a=1)
        ya1 = (A * y) / pi - theta[1]                # i's contribution is (AY) / \pi

        # Calculating Y(a=0)
        ya0 = ((1-A) * y) / (1-pi) - theta[2]        # i's contribution is ((1-A)Y) / (1-\pi)

        # Calculating Y(a=1) - Y(a=0) (using np.ones to ensure a 1-by-n array)
        ate = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

        # Output (3+b)-by-n stacked array
        return np.vstack((ate,             # theta[0] is for the ATE
                          ya1[None, :],    # theta[1] is for R1
                          ya0[None, :],    # theta[2] is for R0
                          preds_reg))      # theta[3:] is for the regression coefficients


This example demonstrates how estimating equations can easily be stacked together using ``delicatessen``. Specifically,
both built-in and user-specified functions can be specified together seamlessly. All it requires is specifying both in
the estimating equation and returning a stacked array of the estimates.

One important piece to note here is that the returned array needs to be in the *same* order as the theta's are input. As
done here, all the ``theta`` values are the 3rd are for the propensity score model. Therefore, the propensity score
model values are last in the returned stack. Returning the values in a different order than input is a common mistake.

Handling ``np.nan``
-------------------------------------

Sometimes, ``np.nan`` will be necessary to include in your data set. However, ``delicatessen`` does not naturally
handle ``np.nan``. In fact, ``delicatessen`` will return an error when there are ``np.nan``'s present (this is by
design). The following discusses how ``np.nan`` can be handled appropriately in the estimating equations.

In the first case, we will consider handling ``np.nan`` with a built-in estimating equation. When trying to fit a
regression model where there are ``np.nan``'s present, the estimating equation missing values must be manually set to
zero. This can be done via the ``numpy.nan_to_num`` function. Below is an example using the built-in logistic
regression estimating equations:

.. code::

    import numpy as np
    import pandas as pd
    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_logistic_regression

    d = pd.DataFrame()
    d['X'] = np.random.normal(size=100)
    y = np.random.binomial(n=1, p=0.5 + 0.01 * d['X'], size=100)
    d['y'] = np.where(np.random.binomial(n=1, p=0.9, size=100), y, np.nan)
    d['C'] = 1

    X = np.asarray(d[['C', 'X']])
    y = np.asarray(d['y'])


    def psi(theta):
        # Estimating logistic model values
        a_model = ee_logistic_regression(theta,
                                         X=X, y=y)
        # Setting
        a_model = np.nan_to_num(a_model, copy=False, nan=0.)
        return a_model


    mest = MEstimator(psi, init=[0, 0, ])
    mest.estimate()

If the ``numpy.nan_to_num`` function had not been included, the optimized points would have been ``nan``.

As a second example, we will consider estimating the mean with missing data and correcting for informative missing
by inverse probability weighting. To reduce random error, this example uses 10,000 observations. Here, we must set
nan's to be zero's prior to subtracting off the mean. This is shown below:

.. code::

    import numpy as np
    import pandas as pd
    from scipy.stats import logistic
    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_logistic_regression
    from delicatessen.utilities import inverse_logit

    # Generating data
    d = pd.DataFrame()
    d['X'] = np.random.normal(size=100000)
    y = 5 + d['X'] + np.random.normal(size=100000)
    d['y'] = np.where(np.random.binomial(n=1, p=logistic.cdf(1 + d['X']), size=100000), y, np.nan)
    d['C'] = 1

    X = np.asarray(d[['C', 'X']])
    y = np.asarray(d['y'])
    r = np.asarray(np.where(d['y'].isna(), 0, 1))


    def psi(theta):
        # Estimating logistic model values
        a_model = ee_logistic_regression(theta[1:],
                                         X=X, y=r)
        pi = inverse_logit(np.dot(X, theta[1:]))

        y_w = np.where(r, y / pi, 0) - theta[0]  # nan-to-zero then subtract off
        return np.vstack((y_w[None, :],
                          a_model))

    mest = MEstimator(psi, init=[0, 0, 0])
    mest.estimate()

This will result in an estimate close to the truth (5). If we were to instead use ``np.where(r, y/pi - theta[0], 0)``,
then the wrong answer will be returned. When in doubt about the form to use (and where the subtraction should go), go
back to the formula for the estimating function or estimator. Here, the IPW mean is

.. math::

    \sum_{i=1}^{n} \left( \frac{I(R_i=1) Y_i}{\Pr(R_i=1 | X_i)} - \theta \right) = 0

As seen with the indicator function, observations where :math:`Y` is missing should contribute a zero *minus*
:math:`\theta`. If we had instead used, the Hajek estimator

.. math::

    \sum_{i=1}^{n} \left((Y_i - \theta) \frac{I(R_i=1)}{\Pr(R_i=1 | X_i)} \right) = 0

The subtraction would have been on the inside of the ``np.where`` step.

Common Mistakes
-------------------------------------

Here is a list of common mistakes, most of which I have done myself.

1. The ``psi`` function doesn't return a NumPy array.
2. The ``psi`` function returns the wrong shape. Remember, it should be a b-by-n NumPy array!
3. The ``psi`` function is summing over n. ``delicatessen`` needs to do the sum internally (in order to compute the
   bread and filling), so do not sum over n in ``psi``!
4. The ``theta`` values and ``b`` *must* be in the same order. If ``theta[0]`` is the mean, the 1st row of the returned
   array better be the estimating function for that mean!

If you still have trouble, please open an issue at
`pzivich/Delicatessen <https://github.com/pzivich/Delicatessen/issues>`_. This will help me to add other common
mistakes here and improve the documentation for custom estimating equations.

Additional Examples
-------------------------------
Additional examples are provided `here <https://github.com/pzivich/Delicatessen/tree/main/tutorials>`_ .
