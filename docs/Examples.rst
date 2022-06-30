Examples
=====================================

Here, we will implement some of the examples described in Chapter 7 of Boos & Stefanski (2013). If you have the book
(or access to it), then reading along with each section may be helpful. To make it easier, I also provide the section
heading from Chapter 7. Finally, we will code each of the estimating equations by-hand (rather than using the built-in
options).


Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.


Sample Mean (7.2.2)
-------------------------------

First, we demonstrate a simple estimating equation for the mean and variance. The estimating equations are

.. image:: images/ee_example_mean.PNG

To demonstrate the example, we will use some generic data for :\math:`Y`. Below is an example data set that will be
used up to Section 7.2.6:

.. code::

    import numpy as np

    np.random.seed(80950841)

    n = 200
    data = pd.DataFrame()
    data['Y'] = np.random.normal(loc=10, scale=2, size=n)
    data['X'] = np.random.normal(loc=5, size=n)
    data['C'] = 1

    # Subsetting the data for 7.2.2
    y = np.asarray(data['Y'])


For use with the M-estimator in ``delicatessen``, we need to write the Python analog of the estimating equation math.
Below is one way we can do this. Notice that the shape of the returned array in 2-by-n.

.. code::

    def psi(theta):
        mean = y - theta[0]                 # \psi_1 from above
        vari = (y-theta[0])**2 - theta[1]   # \psi_2 from above
        return mean, vari

Once we have written up our stacked estimating equations, ``MEstimator`` can be called to solve for :math:`\theta` and
the sandwich variance. We can do that via

.. code::

    from delicatessen import MEstimator

    estr = MEstimator(psi, init=[0, 1])
    estr.estimate()

    print(estr.theta)                # [10.163  4.112]
    print(estr.asymptotic_variance)  # [[ 4.112, -1.674], [-1.674, 36.164]]


The M-Estimator solves for :math:`\theta` via a root finding procedure given the initial values in ``init``. Since the
variance must be >0, we provide a positive initial value. For the sandwich variance, ``delicatessen`` uses a numerical
approximation procedure for the derivative. This is different from the closed-form variance estimator provided in
Chapter 7, but both should return the same answer (within computational error). The advantage of the numerical
derivatives are that they can be done for arbitrary estimating equations.


Ratio (7.2.3)
-------------------------------

Now consider if we wanted to estimate the ratio between two means. For estimation of a ratio, we can consider the
following estimating equation

.. image:: images/ee_example_ratio1.PNG

We can translate the estimating equation from math into python as

.. code::

    def psi(theta):
        return data['Y'] - data['X']*theta  # \psi_1 from above


Now, we can pass this estimating equation and data to `MEstimator`

.. code::

    estr = MEstimator(psi, init=[1, ])
    estr.estimate()

    print(estr.theta)                # [2.082]
    print(estr.asymptotic_variance)  # 0.338

As you may notice, only a single initial value is provided (since only a single array is being returned). Furthermore,
we provide an initial value >0 since we are estimating a ratio.

There is another set of stacked estimating equations we can consider for the ratio. Specifically, we can estimate each
of the means and then take the ratio of those means (rather than doing everything simultaneously). Below is this
alternative set of estimating equations

.. image:: images/ee_example_ratio2.png

Translating this to an estimating equation in Python

.. code::

    def psi(theta):
        mean_y = data['Y'] - theta[0]        # \psi_1 from above
        mean_x = data['X'] - theta[1]        # \psi_2 from above
        ratio = (np.ones(data.shape[0]) *    # \psi_3 from above
                 (theta[0] - theta[1]*theta[2]))
        return mean_y, mean_x, ratio

    estr = MEstimator(psi, init=[0, 0, 1])
    estr.estimate()

    print(estr.theta)  # [10.163,  4.880,  2.082]

Here, we used a trick to make sure the dimension of ``ratio`` stays as :math:`n`, we use ``np.ones``. Without
multiplying by the array of ones, ``ratio`` would be a single value. However, ``MEstimator`` expects a
:math:`3 \times n` array. Multiplying the 3rd equation by an array of 1's keeps the correct dimension.

Also notice this form requires the use of 3 ``init`` values, unlike the other ratio estimator. As before, the ratio
initial value is set >0 to be nice to the root-finder.


Delta Method (7.2.4)
-------------------------------

The delta method has been used in a variety of contexts, including estimating the variance for transformations of
parameters. Instead of separately estimating the parameters, transforming the parameters, and then using the delta
method to estiamte the variance of the transformed parameters; we can apply the transformation in an estimating
equation and automatically estimate the variance for the transformed parameter(s) via the sandwich variance. To do this,
we stack the estimating equation for the transformation into our set of estimating equations. Below is the
mean-variance estimating equations stacked with two transformations of the variance

.. image:: images/ee_example_delta.png

These equations can be expressed programmatically as

.. code::

    def psi_delta(theta):
        mean = data['Y'] - theta[0]                           # \psi_1 from above
        variance = (data['Y'] - theta[0])**2 - theta[1]       # \psi_2 from above
        sqrt_var = (np.ones(data.shape[0])*np.sqrt(theta[1])  # \psi_3 from above
                    - theta[2])
        log_var = (np.ones(data.shape[0])*np.log(theta[1])    # \psi_4 from above
                   - theta[3])
        return (mean, variance, sqrt_var, log_var)

Notice the use of the ``np.ones`` trick again to ensure that the final equations are the correct shapes.

.. code::

    estr = MEstimator(psi, init=[0, 1, 1, 1])
    estr.estimate()

    print(estr.theta)  # [10.163, 4.112, 2.028, 1.414]

Here, there are 4 stacked equations, so ``init`` must be provided 4 values.


Instrumental Variable (7.2.6)
-------------------------------

As a further example, consider the following instrumental variable approach to correcting for measurement error of a
variable. Here, :math:`Y` is the outcome of interest, :math:`X` is the true, unmeasured variable, :math:`W` is the
possibly mismeasured variables, and :math:`T` is the instrument for :math:`X`.

The first set of estimating equations consider in Chapter 7 are

.. image:: images/ee_example_instru1.png

To demonstrate the example, below is some generic simulated data in the described instrumental variable context

.. code::

    np.random.seed(809421)
    n = 500

    data = pd.DataFrame()
    data['X'] = np.random.normal(size=n)
    data['Y'] = 0.5 + 2*data['X'] + np.random.normal(loc=0, size=n)
    data['W'] = data['X'] + np.random.normal(loc=0, size=n)
    data['T'] = -0.75 - 1*data['X'] + np.random.normal(loc=0, size=n)

The previous estimating equations can be translated as

.. code::

    def psi(theta):
        return (theta[0] - data['T'],
                (data['Y'] - data['W']*theta[1])*(theta[0] - data['T']))

    estr = MEstimator(psi, init=[0.1, 0.1, ])
    estr.estimate()

    print(estr.theta)  # [-0.777,  1.769,]

As mentioned in the chapter, certain joint distributions may be of interest. To capture these additional distributions,
the estimating equations were updated to

.. image:: images/ee_example_instru2.png

Again, we can easily translate these equations for ``delicatessen``,

.. code::

    def psi(theta):
        return (theta[0] - data['T'],
                theta[1] - data['W'],
                (data['Y'] - data['W']*theta[2])*(theta[1] - data['W']),
                (data['Y'] - data['W']*theta[3])*(theta[0] - data['T'])
                )

    estr = MEstimator(psi, init=[0.1, 0.1, 0.1, 0.1])
    estr.estimate()

    print(estr.theta)  # [-0.777, 0.005, 0.964, 1.769]

This example demonstrates the flexbility of M-Estimation through the ability to stack estimating equations together.


Robust Location (7.4.1)
-------------------------------

To begin, we generate some generic data used for this example and several of the following

.. code::

    np.random.seed(7841)
    y = np.random.normal(size=250)
    n = y.shape[0]

For the robust mean, the estimating equation proposed by Huber (1964) is

.. image:: images/ee_example_rmean.png

where :math:`k` indicates the bound, such that if :math:`Y_i>k` then :math:`k`, or :math:`Y_i<-k` then :math:`-k`,
otherwise :math:`Y_i`. Below is the estimating equation translated into code

.. code::

    def psi_robust_mean(theta):
        k = 3                          # Bound value
        yr = np.where(y > k, k, y)     # Applying upper bound
        yr = np.where(y < -k, -k, y)   # Applying lower bound
        return yr - theta


    estr = MEstimator(psi_robust_mean, init=[0.])
    estr.estimate()

    print(estr.theta)  # [-0.0126]

Notice that the estimating equation here is not smooth. Specifically, there is a jump at :math:`k`. Therefore, this
estimator only behaves correctly when the values of :math:`\theta` are differentiable (i.e., the true mean can't be
at :math:`k` or :math:`-k`).


Quantile Estimation (7.4.2)
-------------------------------

Despite the sandwich variance needing the function to be smooth at :math:`\theta` (so it is differentiable),
M-Estimation can also be used with non-smooth function. For example, the estimating equations for the sample quantile
is

.. image:: images/ee_example_quantile.png

It is this section, that we need to talk about different root-finding methods, and numerically approximating
derivatives. In the previous examples, we had smooth function that were both easy to find the roots of and had smooth
functions for derivatives. However, that is not the case for quantile estimation. So, we need to use some 'tricks' to
help the procedure along.

First, we are going to use the ``'hybr'`` method. we have found this method to be more reliable when attempting to find
the roots. Often the ``'lm'`` and ``'newton'`` methods appear worse at exploring the space. Next, our estimating
equations 'jump' in terms of their returned value (i.e., they are not smooth). This comes in to the ``tolerance``
parameter. The tolerance determines whether the root-finding has converged. For many quantiles were aren't going to
reach the strict tolerance values. So, we are going to weaken them (the algorithm will be considered as converged under
a weaker condition). If this is not changed, then a non-convergence error will come back.

Now we can talk about numerically approximating the derivatives. Numerical approximations roughly work by calculating
the slope of a line from two points on either side of value (akin to the definition of a derivative you may remember
from math class). For smooth functions, we can choose these points 'close' to the true value. However, this is not the
case for non-smooth functions. For non-smooth functions the derivative can be poorly approximated when relying on points
'too close' to the value. We can address this issue by increasing the ``dx`` parameter. However, large ``dx`` parameters
can also lead to poor approximations. Therefore, we will also increase the ``order`` parameter, which controls the
number of points to use (note: it must be odd).

Now, that we have these tricks, we are ready to find the 25th, 50th, and 75th percentiles using M-Estimation. The
estimating equations are

.. code::

    def psi_quantile(theta):
        return (0.25 - 1*(y <= theta[0]),
                0.50 - 1*(y <= theta[1]),
                0.75 - 1*(y <= theta[2]),)


    estr = MEstimator(psi_quantile, init=[0., 0., 0.])
    estr.estimate(solver='hybr',   # Selecting the hybr method
                  tolerance=1e-3,  # Increasing the tolerance
                  dx=1,            # Increasing distance for numerical approx
                  order=9)         # Increasing the number of points for numerical approx

    print(estr.theta)  # [-0.597  0.048  0.740]

We can compare these values to

.. code::

    np.quantile(y, q=[0.25, 0.50, 0.75])  # [-0.592, 0.047, 0.740]

You'll notice that there is a slight difference. This difference is a result of the non-smooth function. Values 'close'
to these points will not improve the zero finding in the estimating equations. That was why we decreased the tolerance
originally. So, there may be a slight discrepancy between the closed-form solution and M-Estimation.

For non-smooth functions, it is good practice to check against some closed form for the estimating equations.


Positive Mean Deviation (7.4.3)
-------------------------------

For another non-smooth estimating equation(s), we can talk about the positive mean deviation. The estimating equations
are

.. image:: images/ee_example_pmd.png

where :math:`\theta_1` is the positive mean deviation and :math:`\theta_2` is the median.

The estimating equations can be translated into code by

.. code::

    def psi_deviation(theta):
        return ((2 * (y - theta[1]) * (y > theta[1])) - theta[0],
                1/2 - (y <= theta[1]), )

As before, we will use the ``'hybr'`` method along with the updated parameters

.. code::

    estr = MEstimator(psi_deviation, init=[0., 0., ])
    estr.estimate(solver='hybr',   # Selecting the hybr method
                  tolerance=1e-3,  # Increasing the tolerance
                  dx=1,            # Increasing distance for numerical approx
                  order=9)         # Increasing the number of points for numerical approx

    print(estr.theta)  # [0.803 0.042]

If we had used the closed-form definition, we would have ended up with (0.798, 0.047). These values are close, and
again due to the non-smooth nature of the estimating equations.


Linear Regression (7.5.1)
-------------------------------

For linear regression, the estimating equation is

.. image:: images/ee_example_reg.png

For the following examples, the following generic simulated data is used

.. code::

    np.random.seed(5555)
    n = 500
    data = pd.DataFrame()
    data['X'] = np.random.normal(size=n)
    data['Z'] = np.random.normal(size=n)
    data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    data['C'] = 1

As with all the preceding estimating equations, there are multiple ways to code these. Since linear regression involes
some careful matrix manipulations for the programmed estimating equations to return the correct format for
``delicatessen``, we highlight two variations here.

First, we present a vectorized version first.

.. code::

    def psi_regression(theta):
        x = np.asarray(data[['C', 'X', 'Z']])
        y = np.asarray(data['Y'])[:, None]
        beta = np.asarray(theta)[:, None]
        return ((y - np.dot(x, beta)) * x).T

    estr = MEstimator(psi_regression, init=[0., 0., 0.])
    estr.estimate()

    print(estr.theta)  # [0.477, 2.123, -0.852]

For the second approach, a for-loop variation is used instead. Below is the for-loop equivalent for the estimating
equations

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


While these two approaches give the same answer, vectorized versions will generally be faster than for-loop variations
(but may be less 'human readable'). For example, the vectorized version has a runtime of 0.037 seconds and the for-loop
version has a runtime of 0.166 seconds (4.5x slower) on my laptop. Having said that, it is easy to make a mistake with
a vectorized version. I would generally recommend creating a for-loop version first (and then creating a vectorized
version if that for-loop is too slow for your purposes).


Robust Regression (7.5.4)
-------------------------------

The next example is robust regression, where the standard linear regression model is made robust to outliers.
Essentially, we use the robust mean formula from before but now apply it to the error terms of the regression model.
The estimating equations are

.. image:: images/ee_example_robustreg.png

where :math:`k` indicates the bound, such that if :math:`Y_i>k` then :math:`k`, or :math:`Y_i<-k` then :math:`-k`,
otherwise :math:`Y_i`.

Taking the previous vectorized version of the linear regression model and building in the :math:`g_k()` function,

.. code::

    def psi_regression(theta):
        X = np.asarray(data[['C', 'X', 'Z']])
        y = np.asarray(data['Y'])[:, None]
        beta = np.asarray(theta)[:, None]
        k = 2

        # Generating predictions and applying Huber function for robust
        preds = np.asarray(y - np.dot(X, beta))
        preds = np.where(preds > k, k, preds)       # Apply the upper bound
        preds = np.where(preds < -k, -k, preds)     # Apply the lower bound

        # Output b-by-n matrix
        return (preds * X).T


    estr = MEstimator(psi_regression, init=[0., 0., 0.])
    estr.estimate()

    print(estr.theta)  # [0.491, 2.05, -0.795]

You'll notice that the coefficients have changed slightly here. That is because we have reduced the extent of outliers
on the estimation of the linear regression parameters (however, our simulated data mechanism doesn't really result in
major outliers, so the change is small here).

Additional Examples
-------------------------------
Additional examples are provided `here<https://github.com/pzivich/Delicatessen/tree/main/tutorials>`_.
