Custom Estimating Equations
=====================================

One of the key advantages of `deli` is that it has a lot of flexibility in the estimating equations that can be
specified. Basically, it will allow for any estimating equation to be passed to it (but not that the estimating
equation(s) *must* be unbiased for the theory behind M-estimation to hold). Here, I provide an overview and tips for
how to build your own estimating equation.

In general, it will be best if you find an explicit paper or book (most likely written by a statistician) that directly
provides the estimating equation(s) to you. Deriving your own *unbiased* estimating equation may be a lot of effort
and will require a statistical proof. This section does *not* address this part of M-estimation. Rather, this section
provides information on how to construct an estimating equation within `deli`. `deli` assumes you are giving it a
valid estimating equation.

Building from scratch
-------------------------------------

First, we will go through the case of building an estimating equation completely from scratch. To do this, I will
go through an example with linear regression. This is how I went about building the `ee_linear_regression`
functionality.

First, we have the estimating equation (which is the score function in this case) provided in Boos & Stefanski (2013)

.. math::

    \sum_i^n \psi(Y_i, X_i, \theta) = ... = 0



Building with basics
-------------------------------------

Instead of building everything from scratch, you can also piece together the built-in estimating equations with your
own code. To demonstrate this, I will go through how I developed the code for the g-formula.


Common Mistakes
-------------------------------------

Here is a list of common mistakes, most of which I have done myself.

1. The `psi` function doesn't return a NumPy array.
2. The `psi` function returns the wrong shape. Remember, it should be a b-by-n NumPy array!
3. The `psi` function is summing over n. `deli` needs to do the sum internally (for the bread), so do not sum over n!
4. The `theta` values and `b` *must* be in the same order. If `theta[0]` is the mean, the 1st row of the returned
   array better be the mean!

If you still have trouble, please open an issue on GitHub. This will help me to add other common mistakes here and
improve the documentation for custom estimating equations.
