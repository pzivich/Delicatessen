Custom Estimating Equations
=====================================

One of the key advantages of ``delicatessen`` is that it has a lot of flexibility in the estimating equations that can
be specified. Basically, it will allow for any estimating equation to be passed to it (but not that the estimating
equation(s) *must* be unbiased for the theory behind M-estimation to hold). Here, I provide an overview and tips for
how to build your own estimating equation.

In general, it will be best if you find an explicit paper or book (most likely written by a statistician) that directly
provides the estimating equation(s) to you. Deriving your own *unbiased* estimating equation may be a lot of effort
and will require a statistical proof. This section does *not* address this part of M-estimation. Rather, this section
provides information on how to construct an estimating equation within ``delicatessen``. ``delicatessen`` assumes you
are giving it a valid estimating equation.

Building from scratch
-------------------------------------

First, we will go through the case of building an estimating equation completely from scratch. To do this, I will
go through an example with linear regression. This is how I went about building the ``ee_linear_regression``
functionality.

First, we have the estimating equation (which is the score function in this case) provided in Boos & Stefanski (2013)

.. math::

    \sum_i^n \psi(Y_i, X_i, \theta) = (Y_i - X_i^T \beta) X_i = 0

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
for-loop approach will be slow, it is often a good strategy to implement this version first.

Below calculates the estimating equation for each ``i`` in the for-loop. This function returns a stacked array of each
``i`` observation as a 3-by-n array. That array can be validly passed to the ``MEstimator`` for optimization and
calculations.

.. code::

    def psi(theta):
        # Transforming to arrays
        X = np.asarray(d[['C', 'X', 'W']])
        y = np.asarray(d['Y'])
        beta = np.asarray(theta)[:, None]
        n = X.shape[0]

        # Where to store each of the resulting estimates
        est_vals = []

        # Looping through each observation
        for i in range(n):
            v_i = (y[i] - np.dot(X[i], beta)) * X[i]
            est_vals.append(v_i)

        # returning 3-by-n object
        return np.asarray(est_vals).T


We can then run this estimating equation with

.. code::

    mest = MEstimator(psi, init=[0., 0., 0.])
    mest.estimate()

for which the coefficients match the coefficients from a ordinary least squares model (variance estimates will differ).
Here, we can further vectorize the estimating equation. In the vector-form, this code will run much faster and this
is often the best approach to boosting speed in terms of run-time.

With some careful experimentation, the following is a vectorized version. Remember that ``delicatessen`` is expecting a
b-by-n array to be given by the ``psi`` function. Failure to provide this is a common mistake when building custom
estimating equations.

.. code::

    def psi(theta):
        X = np.asarray(d[['C', 'X', 'W']])
        y = np.asarray(d['Y'])[:, None]
        beta = np.asarray(theta)[:, None]
        return ((y - np.dot(X, beta)) * X).T


As before, we can run this chunk of code. However, this is substantially faster. If we run both implementations on the
same data set of 10,000 observations, the for-loop version took approximately 1.60 seconds and the vectorized version
took 0.05 seconds (on my fairly new laptop). That is a large difference in run time! Vectorizing (even parts of an
estimating equation) can help to improve run-times if you find the M-Estimation procedure taking too long.


Building with basics
-------------------------------------

Instead of building everything from scratch, you can also piece together the built-in estimating equations with your
own code. To demonstrate this, I will go through how I developed the code for inverse probability weighting.

The inverse probability weighting estimator consists of four estimating equations: the propensity score model, the
weighted mean for treatment, the weighted mean for no-treatment, and the difference between the weighted means. We
can write this as

.. math::

    \sum_i^n \psi_1(A_i, W_i, \theta) = \sum_i^n (A_i - expit(W_i^T \theta)) W_i = 0

    \sum_i^n \psi_2(Y_i, A_i, \pi_i, \theta_1) = \sum_i^n \frac{A_i \times Y_i}{\pi_i} - \theta_1 = 0

    \sum_i^n \psi_3(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n \frac{(1-A_i) \times Y_i}{1-\pi_i} - \theta_2 = 0

    \sum_i^n \psi_4(Y_i, A_i, \pi_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0


Rather than re-code the logistic regression model (to estimate the propensity scores), we will use the built-in
logistic regression functionality. Below is a stacked estimating equation for the inverse probability weighting
estimator

.. code::

    def psi(theta):
        # Ensuring correct typing
        W = np.asarray(d['W'])
        A = np.asarray(d['A'])
        y = np.asarray(y)
        beta = theta[3:]   # Extracting out theta's for the regression model

        # Estimating propensity score
        preds_reg = ee_logistic_regression(theta=beta,    # Using logistic regression
                                           X=W,           # Plug-in covariates for X
                                           y=A)           # Plug-in treatment for Y

        # Estimating weights
        pi = inverse_logit(np.dot(W, beta))          # Getting Pr(A|W) from model (using delicatessen.utilities)

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

One important piece to note here is that the returned array should be in the *same* order as the theta's are input. As
done here, all the ``theta`` values are the 3rd are for the propensity score model. Therefore, the propensity score
model values are last in the returned stack. Returning the values in a different order than expected by theta is a
common mistake and will lead to failed optimizations.


Common Mistakes
-------------------------------------

Here is a list of common mistakes, most of which I have done myself.

1. The ``psi`` function doesn't return a NumPy array.
2. The ``psi`` function returns the wrong shape. Remember, it should be a b-by-n NumPy array!
3. The ``psi`` function is summing over n. ``delicatessen`` needs to do the sum internally (for the bread), so do not
   sum over n!
4. The ``theta`` values and ``b`` *must* be in the same order. If ``theta[0]`` is the mean, the 1st row of the returned
   array better be the mean!

If you still have trouble, please open an issue on `GitHub<https://github.com/pzivich/Delicatessen/issues>`_. This will
help me to add other common mistakes here and improve the documentation for custom estimating equations.
