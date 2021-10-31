Built-in Estimating Equations
'''''''''''''''''''''''''''''''''''''

Here, we provide an overview of some of the built-in estimating equations with `delicatessen`. This documentation is
split into four sections, corresponding to basic, regression, survival, and causal estimating equations.

All built-in estimating equations need to be 'wrapped' inside a outer function. Below is an example of an outer
function, `psi` for the generic estimating equation `ee` (`ee` does not exist but it a placeholder here).

.. code::python

    def psi(theta):
        return ee(theta=theta, data=dat)

Here, the generic `ee` takes two inputs `theta` and `data`. `theta` is the general theta vector that is present in
all stacked estimating equations expected by `delicatessen`. `data` is an argument that takes an input source of data.
The `dat` provided should be **in the local scope** of the .py file this function lives in.

After wrapped in an outer function, the function can be passed to `MEstimator`. See the examples below for further
details and examples.

Basic Equations
=============================

Some basic estimating equations for the mean and variance are provided.

Mean
----------------------------

The most basic is the estimating equation for the mean: `ee_mean`. To illustrate, consider we wanted to estimated the
mean for the following data

.. code::

    obs_vals = [1, 2, 1, 4, 1, 4, 2, 4, 2, 3]

To use `ee_mean` with `MEstimator`, this function will be wrapped in an outer function. Below is an example

.. code::

    from delicatessen.estimating_equations import ee_mean

    def psi(theta):
        return ee_mean(theta=theta, y=obs_vals)

Note that `obs_vals` must be available in the scope of this defined function.

After creating the wrapper function, the M-Estimator can be called like the following

.. code::

    from delicatessen import MEstimator

    mestimation = MEstimator(stacked_equations=psi, init=[0, ])
    mestimation.estimate()

Since `ee_mean` consists of a single parameter, only a single `init` value is provided.

Robust Mean
----------------------------

Sometimes extreme outliers are observed. The mean will be sensitive to these extreme outliers, but excluding them also
seems like 'cheating'. Instead, a robust version of the mean could be considered. Consider the following generic data,
where there are two extreme outliers

.. code::

    obs_vals = [1, -10, 2, 1, 4, 1, 4, 2, 4, 2, 3, 12]

Rather than excluding the -10 and 12, we can use the robust mean proposed by Huber. This trims the outliers to a
pre-specified level. Therefore, they still contribute information but only values up to the bound. In this example, a
bound of -6,6 will be applied.

The robust mean estimating equation is available in `ee_mean_robust`. Below is an example (including the wrapper
function and call).

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_mean_robust

    def psi(theta):
        return ee_mean_robust(theta=theta, y=obs_vals, k=6)

    mestimation = MEstimator(stacked_equations=psi, init=[0, ])
    mestimation.estimate()

    print("Mean:    ", mestimation.theta)
    print("Variance:", mestimation.variance)

Therefore, the robust mean point and variance estimate are displayed.

Mean and Variance
----------------------------

A stacked estimating equation, where the first value is the mean and the second is the variance, is also provided.
Returning to the previous data,

.. code::

    obs_vals = [1, 2, 1, 4, 1, 4, 2, 4, 2, 3]

The mean-variance estimating equation can be implemented as follows (remember the wrapper function!)

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_mean_variance

    def psi(theta):
        return ee_mean_variance(theta=theta, y=obs_vals)

    mestimation = MEstimator(stacked_equations=psi, init=[0, 1, ])
    mestimation.estimate()

    print("theta:     ", mestimation.theta)
    print("Var(theta):", mestimation.variance)

*Note* the `init` here takes two values because the stacked estimating equations has a length of 2 (theta is b-by-1
where b=2). The first value of theta is the mean and the second is the variance. Now, the variance output provides
a 2-by-2 covariance matrix. The leading diagonal of that matrix are the variances (where the first is the estimated
variance of the mean and the second is the estimated variance of the variance).

Regression
=============================

Several basic regression model estimating equations are provided.

Linear Regression
----------------------------

The estimating equations for linear regression predict a continuous outcome as a function of provided covariates. The
implementation of linear regression here is similar to ordinary least squares, but the variance here is robust.
Specifically, the sandwich variance estimator of M-Estimation is robust.

To demonstrate application, consider the following simulated data set

.. code::

    import numpy as np
    import pandas as pd

    n = 500
    data = pd.DataFrame()
    data['X'] = np.random.normal(size=n)
    data['Z'] = np.random.normal(size=n)
    data['Y'] = 0.5 + 2*data['X'] - 1*data['Z'] + np.random.normal(loc=0, size=n)
    data['C'] = 1

In this case, X and Z are the independent variables and Y is the dependent variable. Here C is necessary as a column
since we need to manually provide the intercept (this may be different from other formula-based packages that
automatically add the intercept to the regression).

For this data, we can now create the wrapper function for the `ee_linear_regression` estimating equations

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_linear_regression

    def psi(theta):
        return ee_linear_regression(theta=theta,
                                    X=data[['C', 'X', 'Z']],
                                    y=data['Y'])

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the regression coefficients
and their variance

.. code::

    mestimation = MEstimator(stacked_equations=psi, init=[0., 0., 0.])
    mestimation.estimate()

    print("theta:     ", mestimation.theta)
    print("Var(theta):", mestimation.variance)

Note that `X` is 3 covariates, meaning `init` needs 3 starting values. The linear regression done here should match
the `statsmodels` generalized linear model with a robust variance estimate. Below is code demonstrating how to
estimate the same quantities with `statsmodels.glm`.

.. code::

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    glm = smf.glm("Y ~ X + Z", data).fit(cov_type="HC1")
    print(np.asarray(glm.params))         # Point estimates
    print(np.asarray(glm.cov_params()))   # Covariance matrix

While `statsmodels` likely runs faster, the benefit of M-estimation and `delicatessen` is that multiple estimating
equations can be stacked together (including multiple regression models). This advantage will become clearer in the
causal section.

Logistic Regression
----------------------------

In the case of a binary dependent variable, logistic regression can instead be performed (no linear probability models
here!).

To demonstrate application, consider the following simulated data set

.. code::

    import numpy as np
    import pandas as pd
    from scipy.stats import logistic

    n = 500
    data = pd.DataFrame()
    data['X'] = np.random.normal(size=n)
    data['Z'] = np.random.normal(size=n)
    data['Y'] = np.random.binomial(n=1, p=logistic.cdf(0.5 + 2*data['X'] - 1*data['Z']), size=n)
    data['C'] = 1

In this case, X and Z are the independent variables and Y is the dependent variable. Here C is necessary as a column
since we need to manually provide the intercept (this may be different from other formula-based packages that
automatically add the intercept to the regression).

For this data, we can now create the wrapper function for the `ee_logistic_regression` estimating equations

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_logistic_regression

    def psi(theta):
        return ee_logistic_regression(theta=theta,
                                      X=data[['C', 'X', 'Z']],
                                      y=data['Y'])

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the regression coefficients
and their variance

.. code::

    mestimation = MEstimator(stacked_equations=psi, init=[0., 0., 0.])
    mestimation.estimate()

    print("theta:     ", mestimation.theta)
    print("Var(theta):", mestimation.variance)

Note that `X` is 3 covariates, meaning `init` needs 3 starting values. The logistic regression done here should match
the `statsmodels` generalized linear model with a robust variance estimate. Below is code demonstrating how to
estimate the same quantities with `statsmodels.glm`.

.. code::

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    glm = smf.glm("Y ~ X + Z", data,
                  family=sm.families.Binomial()).fit(cov_type="HC1")
    print(np.asarray(glm.params))         # Point estimates
    print(np.asarray(glm.cov_params()))   # Covariance matrix

While `statsmodels` likely runs faster, the benefit of M-estimation and `delicatessen` is that multiple estimating
equations can be stacked together (including multiple regression models). This advantage will become clearer in the
causal section.


Causal Inference
=============================

To demonstrate the utility of M-estimation, particularly how estimating equations can be 'stacked' together, then
still have an appropriate variance estimator, several causal inference estimators are provided here.

It is recommended that you are familiar with causal inference (particularly the identification conditions of these
estimators) before using this utility widely. Causal inference is a difficult endeavour, my dear user!

In the following examples, we will use the generic data example here, where W is a confounder of the A-Y relationship

.. code::

    n = 200
    d = pd.DataFrame()
    d['W'] = np.random.binomial(1, p=0.5, size=n)
    d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    d['C'] = 1

Now to the examples

G-computation
----------------------------

First, is g-computation. The built-in estimating equations for g-computation calculate the average treatment effect,
risk / mean under all-treated, and the risk / mean under none-treated.

*A limitation*: the g-computation, as implemented in the built-in estimating equation only uses a single outcome model
and that outcome model does *not* support interaction terms. Here the g-computation is meant as a basic example. For
more general use, the provided estimating equation should be adapted. But the built-in estimating equation will provide
a basic structure for user's to build off of.

To load the estimating equations, we call

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_gformula

Again, we will wrap the built-in estimating equations inside a function.

.. code::

    def psi(theta):
        return ee_gformula(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

The arguments for `ee_gformula` are the theta values, the covariates (including an intercept (C) and the treatment (A)),
the outcome values (Y), and the column index for the treatment in X. Here, 1 designates the second column (python uses
zero-indexing), which corresponds to 'A' in how the X data is formatted.

Now we can call the M-estimator to solve for the values and the variance. Here, the initial values provided must be
3+*b* (where *b* is the number of columns in X). This is because the g-computation estimating equations output the
average treatment effect, risk under all-treated, risk under none-treated, and the regression model coefficients.

As for starting values, it will likely be best practice to have the initial values set as  [0., 0.5, 0.5, ...] in
general. The regression initial values can also be pre-washed to speed up optimization.

.. code::

    mestimation = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0., 0.])
    mestimation.estimate(solver='lm')

Now the average treatment effect, as well as the variance, can be output. Here, a key advantage of M-estimation can be
seen. The form of an M-estimator allows us to estimate the variance directly, while appropriately allowing for the
uncertainty in the regression model parameters to be carried forward. M-estimation does this automatically for us.
Essentially, we do not need to bootstrap to estimate the variance!

.. code::

    mestimation.theta[0]
    mestimation.variance[0, 0]

Besides the average treatment effect, the risk / mean under all-treated can be extracted by

.. code::

    mestimation.theta[1]
    mestimation.variance[1, 1]

and the risk / mean under none-treated can be extracted by

.. code::

    mestimation.theta[2]
    mestimation.variance[2, 2]

The `ee_gformula` supports both binary and continuous outcomes. Inside the function, it automatically detects whether
the outcome data is binary. If the outcome data is not binary, then it defaults to using a linear regression model
(but you can also force the use of a linear regression model for binary data by setting `force_continuous=True`

To summarize, the key advantage of M-estimation here is that it *appropriately* estimates the variance. We do *not*
need to bootstrap in this case (and more generally if the sample size is sufficiently large).

Inverse probability weighting
-------------------------------------

Rather than modeling the outcome, we can choose the inverse probability weighting (IPW) estimator, which models the
probability of treatment. The estimating equations for the IPW estimator are also built-in to `delicatessen`.

To load the estimating equations, we call

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_ipw

As with every built-in estimating equation, we will wrap it inside a function.

.. code::

    def psi(theta):
        return ee_ipw(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

The arguments for `ee_ipw` are the theta values, the covariates (including an intercept (C) and the treatment (A)),
the outcome values (Y), and the column index for the treatment in X. Here, 1 designates the second column (python uses
zero-indexing), which corresponds to 'A' in how the X data is formatted.

Now we can call the M-estimator to solve for the values and the variance. Here, the initial values provided must be
3+*b* (where *b* is the number of columns in X *minus 1*). This is because the IPW estimating equations output the
average treatment effect, risk under all-treated, risk under none-treated, and the logistic regression model
coefficients. Since we are modeling the conditional probability of A, one column in X is excluded from the covariates.

As for starting values, it will likely be best practice to have the initial values set as  [0., 0.5, 0.5, ...] in
general. The regression initial values can also be pre-washed to speed up optimization.

.. code::

    mestimation = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0., 0.])
    mestimation.estimate(solver='lm')

Now the average treatment effect, as well as the variance, can be output. Here, a key advantage of M-estimation can be
seen. The form of an M-estimator allows us to estimate the variance directly, while appropriately allowing for the
uncertainty in the regression model parameters to be carried forward. M-estimation does this automatically for us.
Essentially, we do not need to bootstrap or use the GEE-trick for IPW to estimate the variance!

.. code::

    mestimation.theta[0]
    mestimation.variance[0, 0]

Besides the average treatment effect, the risk / mean under all-treated can be extracted by

.. code::

    mestimation.theta[1]
    mestimation.variance[1, 1]

and the risk / mean under none-treated can be extracted by

.. code::

    mestimation.theta[2]
    mestimation.variance[2, 2]

The `ee_ipw` supports both binary and continuous outcomes automatically. Both of these variable types are handled in
the same way due to the form of the Horwitz-Thompson estimator.

Unlike the GEE-trick for IPW (which provides a conservative estimator of the variance), the variance estimator here
is correct. This means it will be narrower than the GEE-trick. Therefore, this approach is generally preferred over
the GEE-trick to calculating the variance for the IPW estimator. It is also much more computationally efficient than
the bootstrap.

Augmented inverse probability weighting
----------------------------------------------

Before, we model the outcome and treatment models separately. Now, we will consider the augmented inverse probability
weighting (AIPW) model, which incorporates both the treatment and outcome models. AIPW is a semi-parametric
doubly-robust estimator for the average treatment effect. For a basic overview, see Funk et al. (2011).

*A limitation*: as with g-computation, the built-in AIPW estimating equation only uses a single outcome model
and that outcome model does *not* support interaction terms. Here the AIPW is meant as a basic example. For
more general use, the provided estimating equation should be adapted. But the built-in estimating equation will provide
a basic structure for user's to build off of.

The estimating equations for the AIPW estimator are also provided in `delicatessen`. To load the estimating equations,
we call

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_aipw

As always, we will wrap the built-in estimating equation inside a function.

.. code::

    def psi(theta):
        return ee_aipw(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

The arguments for `ee_aipw` are the theta values, the covariates (including an intercept (C) and the treatment (A)),
the outcome values (Y), and the column index for the treatment in X. Here, 1 designates the second column (python uses
zero-indexing), which corresponds to 'A' in how the X data is formatted.

Now we can call the M-estimator to solve for the values and the variance. Here, the initial values provided must be
3+*b*+*b-1* (where *b* is the number of columns in X). This is because the AIPW estimating equations output the
average treatment effect, risk under all-treated, risk under none-treated, and the outcome model coefficients, and
the treatment model coefficients.

As for starting values, it will likely be best practice to have the initial values set as  [0., 0.5, 0.5, ...] in
general. The regression initial values can also be pre-washed to speed up optimization.

.. code::

    mestimation = MEstimator(stacked_equations=psi, init=[0., 0.5, 0.5, 0., 0., 0.])
    mestimation.estimate(solver='lm')

Now the average treatment effect, as well as the variance, can be output. Here, a key advantage of M-estimation can be
seen. The form of an M-estimator allows us to estimate the variance directly, while appropriately allowing for the
uncertainty in the regression model parameters to be carried forward. M-estimation does this automatically for us.
Essentially, we do not need to bootstrap or use the GEE-trick for IPW to estimate the variance!

.. code::

    mestimation.theta[0]
    mestimation.variance[0, 0]

Besides the average treatment effect, the risk / mean under all-treated can be extracted by

.. code::

    mestimation.theta[1]
    mestimation.variance[1, 1]

and the risk / mean under none-treated can be extracted by

.. code::

    mestimation.theta[2]
    mestimation.variance[2, 2]

The variance estimator in this case will match the influence function estimator of the variance that is commonly used
for AIPW. See Boos & Stefanski (2013) for more detailed discussion on the relation between M-estimation and influence
curves.

Further Readings
=============================
Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.

Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust estimation of causal
effects. *American Journal of Epidemiology*, 173(7), 761-767.

Huber PJ. (1992). Robust estimation of a location parameter. In Breakthroughs in statistics (pp. 492-518).
Springer, New York, NY.

Satten GA, & Datta S. (2001). The Kaplan–Meier estimator as an inverse-probability-of-censoring weighted average.
*The American Statistician*, 55(3), 207-210.
