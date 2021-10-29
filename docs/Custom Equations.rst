Custom Estimating Equations
=====================================

One of the key advantages of `deli` is that it has a lot of flexibility in the estimating equations that can be
specified. Basically, it will allow for any estimating equation to be passed to it (but not that the estimating
equation(s) *must* be unbiased for the theory behind M-estimation to hold). Here, I provide an overview and tips for
how to build your own estimating equation.



Building your own custom-regression model is easy if you have access to the score function (and can feasibly program
the score function).

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
