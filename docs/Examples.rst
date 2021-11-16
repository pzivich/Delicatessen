Examples
=====================================

Here, I will go through a variety of examples (using custom estimating equations, rather than the built-in options).
These examples follow Chapter 7 of Boos & Stefanski. If you have the book (or access to it), then reading along with
each section may be helpful. To make it easier, I also provide the section heading from Chapter 7.

Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.


Sample Mean (7.2.2)
-------------------------------

First, we demonstrate a simple estimating equation for the mean and variance (stacked together). The estimating
equations are

.. math::

    \psi_1(Y_i, \theta) &= Y_i - \theta_1 \\
    \psi_2(Y_i, \theta) &= (Y_i - \theta_1)^2 - \theta_2

To demonstrate the example, we will use some generic data for Y

.. code::

    import numpy as np
    y = np.array([-2, 1, 3, 4, 1, 4, -5, 3, 6])

For use with the M-estimator in `delicatessen`, we can program these estimating equations as

.. code::

    def psi(theta):
        mean = y - theta[0]                 # \psi_1 from above
        vari = (y-theta[0])**2 - theta[1]   # \psi_2 from above
        return mean, vari

After defining the stacked estimating equations, the M-estimator can be called

.. code::

    mestimate = MEstimator(psi, init=[0, 1])
    mestimate.estimate()

The M-estimator will solve for :math:`\theta` via a root finding procedure. For the sandwich variance, `delicatessen`
uses a numerical approximation procedure for the derivative. This is different from the closed-form variance estimator
provided in Chapter 7, but both should return the same answer (within computational error). The advantage of the
numerical derivatives are that they can be done for arbitrary estimating equations.


Ratio (7.2.3)
-------------------------------

For the ratio estimator, the following estimating equation is provided

.. math::

    \psi_1(Y_i, X_i, \theta) = Y_i - \theta_1 \X_i

For use with the M-estimator in `delicatessen`, we can program this estimating equation as

.. code::

    import numpy as np
    import pandas as pd

    n = 200
    data = pd.DataFrame()
    data['Y'] = np.random.normal(loc=10, scale=2, size=n)
    data['X'] = np.random.normal(loc=5, size=n)

    def psi(theta):
        return data['Y'] - data['X']*theta

As before, we can solve the estimating equation by

.. code::

    mestimate = MEstimator(psi, init=[1, ])
    mestimate.estimate()

As you may notice, only a single initial value is provided (since only a single array is being returned).

The chapter also provides an alternative series of estimating equations for the ratio.

.. math::

    \psi_1(Y_i, X_i, \theta) &= Y_i - \theta_1 \\
    \psi_2(Y_i, X_i, \theta) &= X_i - \theta_2 \\
    \psi_3(Y_i, X_i, \theta) &= \theta_1 - \theta_2 \times \theta_3


Similarly, this can also be programmed via

.. code::

    def psi(theta):
        mean_y = data['Y'] - theta[0]
        mean_x = data['X'] - theta[1]
        ratio = np.ones(data.shape[0]) * (theta[0] - theta[1]*theta[2])
        return mean_y, mean_x, ratio

    mestimate = MEstimator(psi, init=[0, 0, 1])
    mestimate.estimate()

Here, there is a series of estimating equations. It is also important to note the use of `np.ones` in the third step.
This ensures that `ratio` consists on *n* observations. Without multiplying by the array of ones, `ratio` would be a
single value. However, `MEstimator` expects a 3-by-*n* array here. Multiplying the 3rd equation by an array of 1's
keeps the correct dimension and keeps the values.

Also notice that these estimating equations require the use of 3 `init` values, unlike the other ratio estimator.


Delta Method (7.2.4)
-------------------------------

The delta method can also be cast as an M-estimation problem. The chapter demonstrates two transformations of Y and
their corresponding mean. The stacked estimating equations are

.. math::

    \psi_1(Y_i, \theta) &= Y_i - \theta_1 \\
    \psi_2(Y_i, \theta) &= (Y_i - \theta_1)^2 - \theta_2 \\
    \psi_3(Y_i, \theta) &= \sqrt{\theta_2} - \theta_3 \\
    \psi_4(Y_i, \theta) &= \log(\theta_2) - \theta_4 \\


These equations can be expressed programmatically for `delicatessen` as

.. code::

    def psi_delta(theta):
        return (data['Y'] - theta[0],
                (data['Y'] - theta[0])**2 - theta[1],
                np.ones(data.shape[0])*np.sqrt(theta[1]) - theta[2],
                np.ones(data.shape[0])*np.log(theta[1]) - theta[3])

Notice the use of the `np.ones` trick as done with the ratio estimating equations to ensure that the final equations are
the correct shapes.

.. code::

    mestimate = MEstimator(psi, init=[0, 0, 1, 1])
    mestimate.estimate()

Here, there are 4 stacked equations, so `init` must be provided 4 values.


Instrumental Variable (7.2.6)
-------------------------------

The first set of estimating equations for the instrumental variable analysis are

.. math::

    \psi_1(Y_i, W_i, T_i, \theta) &= \theta_1 - T \\
    \psi_2(Y_i, W_i, T_i, \theta) &= (Y - \theta_2 W)(\theta_1 - T) \\

To demonstrate the example, below is some generic simulated data

.. code::

    n = 500
    data = pd.DataFrame()
    data['X'] = np.random.normal(size=n)
    data['Y'] = 0.5 + 2*data['X'] + np.random.normal(loc=0, size=n)
    data['W'] = data['X'] + np.random.normal(loc=0, size=n)
    data['T'] = -0.75 - 1*data['X'] + np.random.normal(loc=0, size=n)

These estimating equations can be programmed for `delicatessen` as

.. code::

    def psi(theta):
        return (theta[0] - data['T'],
                (data['Y'] - data['W']*theta[1])*(theta[0] - data['T']))

    mestimate = MEstimator(psi, init=[0.1, 0.1, ])
    mestimate.estimate()

As mentioned in the chapter, certain joint distributions may be of interest. To capture those distributions, the
estimating equations from before were further updated to

.. math::

    \psi_1(Y_i, W_i, T_i, \theta) &= \theta_1 - T \\
    \psi_2(Y_i, W_i, T_i, \theta) &= \theta_2 - W \\
    \psi_3(Y_i, W_i, T_i, \theta) &= (Y - \theta_3 W)(\theta_2 - W) \\
    \psi_4(Y_i, W_i, T_i, \theta) &= (Y - \theta_4 W)(\theta_1 - T) \\

Again, we can easily write these equations for `delicatessen`,

.. code::

    def psi(theta):
        return (theta[0] - data['T'],
                theta[1] - data['W'],
                (data['Y'] - data['W']*theta[2])*(theta[1] - data['W']),
                (data['Y'] - data['W']*theta[3])*(theta[0] - data['T'])
                )

    mestimator = MEstimator(psi, init=[0.1, 0.1, 0.1, 0.1])
    mestimator.estimate()

This example further demonstrates the flexibility of M-estimation by stacking together estimating equations.


Robust Location (7.4.1)
-------------------------------

For robust location estimation, the estimating equation is

.. math::

    \psi_k(Y_i, \theta) = Y^k_i - theta_1

where *k* indicates the upper and lower bound, and Y superscript *k* is the bounded values of Y.

Below is the estimating equation in Python

.. code::

    import numpy as np
    var = np.array([1, -10, 2, 1, 4, 1, 4, 2, 4, 2, 3, 12])

    def psi(theta):
        var = np.where(var > k, k, var)       # Apply the upper bound
        var = np.where(var < -k, -k, var)     # Apply the lower bound
        return var - theta                    # Estimating equation for robust mean

    mestimator = MEstimator(psi, init=[0., ])
    mestimator.estimate()

Notice that the estimating equation here is not smooth. Specifically, there is a jump at *k*. Therefore, this will only
work for values of theta that are differentiable (i.e., the true mean can't be at *k*).


Linear Regression (7.5.1)
-------------------------------

For linear regression, the estimating equation is

.. math::

    \psi(X_i, Y_i \beta) = (Y_i - X_i^T \beta) X_i

Here, we present the vectorized version first. Notice that the vectorized version will generally be faster than a
for-loop implementation. However, it is easy to make a mistake with a vectorized version, so I generally recommend
creating a for-loop version first (and then creating a vectorized version if that for-loop is too slow).

With some generic data,

.. code::

    n = 500
    data = pd.DataFrame()
    data['X'] = np.random.normal(size=n)
    data['Z'] = np.random.normal(size=n)
    data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    data['C'] = 1

The estimating equation and M-estimation procedure is then called by

.. code::

    def psi_regression(theta):
        x = np.asarray(data[['C', 'X', 'Z']])
        y = np.asarray(data['Y'])[:, None]
        beta = np.asarray(theta)[:, None]
        return ((y - np.dot(x, beta)) * x).T

    mestimator = MEstimator(psi_regression, init=[0.1, 0.1, 0.1])
    mestimator.estimate()

As mentioned, a for-loop version can also be used. Below is an example of the for-loop version for regression

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


GEE (7.5.6)
-------------------------------

... to be added ...

