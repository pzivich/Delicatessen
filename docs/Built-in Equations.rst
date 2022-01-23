Built-in Estimating Equations
'''''''''''''''''''''''''''''''''''''

Here, we provide an overview of some of the built-in estimating equations with ``delicatessen``. This documentation is
split into three sections, corresponding to basic, regression, and causal estimating equations.

All built-in estimating equations need to be 'wrapped' inside an outer function. Below is a generic example of an outer
function, where ``psi`` is the wrapper function and ``ee`` is the generic estimating equation example (``ee`` does not
exist but it a placeholder here).

.. code::

    def psi(theta):
        return ee(theta=theta, data=data)

Here, the generic ``ee`` takes two inputs ``theta`` and ``data``. ``theta`` is the general theta vector that is present
in all stacked estimating equations expected by ``delicatessen``. ``data`` is an argument that takes an input source
of data. The ``data`` provided should be **in the local scope** of the ``.py`` file this function lives in.

After wrapped in an outer function, the function can be passed to ``MEstimator``. See the examples below for further
details and examples.

Basic Equations
=============================

Some basic estimating equations are provided.

Mean
----------------------------

The most basic available estimating equation is for the mean: ``ee_mean``. To illustrate, consider we wanted to
estimated the mean for the following data

.. code::

    obs_vals = [1, 2, 1, 4, 1, 4, 2, 4, 2, 3]

To use ``ee_mean`` with ``MEstimator``, this function will be wrapped in an outer function. Below is an illustration of
this wrapper function

.. code::

    from delicatessen.estimating_equations import ee_mean

    def psi(theta):
        return ee_mean(theta=theta, y=obs_vals)

Note that ``obs_vals`` must be available in the scope of this defined function.

After creating the wrapper function, the M-Estimator can be called like the following

.. code::

    from delicatessen import MEstimator

    estr = MEstimator(stacked_equations=psi, init=[0, ])
    estr.estimate()

    print(estr.theta)   # [2.4, ])

Since ``ee_mean`` consists of a single parameter, only a single ``init`` value is provided.

Robust Mean
----------------------------

Sometimes extreme observations, termed outliers, occur. The mean is generally sensitive to these outliers. A common
approach to handling outliers is to exclude them. However, exclusion ignores all information contributed by outliers,
and should only be done when outliers are a result of experimental error. Robust statistics have been proposed as
middle ground, whereby outliers contribute to estimation but their influence is constrained.

.. code::

    obs_vals = [1, -10, 2, 1, 4, 1, 4, 2, 4, 2, 3, 12]

Rather than excluding the -10 and 12, we can use the robust mean proposed by Huber. This trims the outliers to a
pre-specified level. Therefore, they still contribute information but only values up to the bound. In this example, a
bound of -6,6 will be applied.

The robust mean estimating equation is available in ``ee_mean_robust``.

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_mean_robust

    def psi(theta):
        return ee_mean_robust(theta=theta, y=obs_vals, k=6)

    estr = MEstimator(stacked_equations=psi, init=[0, ])
    estr.estimate()

    print(estr.theta)  # [2.0, ]


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

    estr = MEstimator(stacked_equations=psi, init=[0, 1, ])
    estr.estimate()

    print(estr.theta)  # [2.4, 1.44]

*Note* the ``init`` here takes two values because the stacked estimating equations has a length of 2 (``theta`` is
b-by-1 where b=2). The first value of ``theta`` is the mean and the second is the variance. Now, the variance output
provides a 2-by-2 covariance matrix. The leading diagonal of that matrix are the variances (where the first is the
estimated variance of the mean and the second is the estimated variance of the variance).

Regression
=============================

Several common regression models are provided as built-in estimating equations.

Linear Regression
----------------------------

The estimating equations for linear regression predict a continuous outcome as a function of provided covariates.

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

In this case, ``X`` and ``Z`` are the independent variables and ``Y`` is the dependent variable. Here the column ``C``
is created to be the intercept column, since the intercept needs to be manually provided (this may be different from
other formula-based packages that automatically add the intercept to the regression).

For this data, we can now create the wrapper function for the ``ee_linear_regression`` estimating equations

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

    estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.])
    estr.estimate()

    print(estr.theta)
    print(estr.variance)

Note that there are 3 independent variables, meaning ``init`` needs 3 starting values. The linear regression done here
should match the ``statsmodels`` generalized linear model with a robust variance estimate. Below is code on how to
compare to ``statsmodels.glm``.

.. code::

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    glm = smf.glm("Y ~ X + Z", data).fit(cov_type="HC1")
    print(np.asarray(glm.params))         # Point estimates
    print(np.asarray(glm.cov_params()))   # Covariance matrix

While ``statsmodels`` likely runs faster, the benefit of M-estimation and ``delicatessen`` is that multiple estimating
equations can be stacked together (including multiple regression models). This advantage will become clearer in the
causal section.

Logistic Regression
----------------------------

In the case of a binary dependent variable, logistic regression can instead be performed. Consider the following
simulated data set

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

In this case, ``X`` and ``Z`` are the independent variables and ``Y`` is the dependent variable. Here the column ``C``
is created to be the intercept column, since the intercept needs to be manually provided (this may be different from
other formula-based packages that automatically add the intercept to the regression).

For this data, we can now create the wrapper function for the ``ee_logistic_regression`` estimating equations

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

    estr = MEstimator(stacked_equations=psi, init=[0., 0., 0.])
    estr.estimate()

    print(estr.theta)
    print(estr.variance)

Note that there are 3 independent variables, meaning ``init`` needs 3 starting values. The logistic regression done here
should match the ``statsmodels`` generalized linear model with a robust variance estimate. Below is code on how to
compare to ``statsmodels.glm``.

.. code::

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    glm = smf.glm("Y ~ X + Z", data,
                  family=sm.families.Binomial()).fit(cov_type="HC1")
    print(np.asarray(glm.params))         # Point estimates
    print(np.asarray(glm.cov_params()))   # Covariance matrix

While ``statsmodels`` likely runs faster, the benefit of M-estimation and ``delicatessen`` is that multiple estimating
equations can be stacked together (including multiple regression models). This advantage will become clearer in the
causal section.


Dose-Response
=============================

Estimating equations for dose-response relationships are also included. The following examples use the data from
Inderjit et al. (2002). This data can be loaded via

.. code::

    d = load_inderjit()   # Loading array of data
    dose_data = d[:, 1]   # Dose data
    resp_data = d[:, 0]   # Response data


4-parameter Logistic
----------------------------

The 4-parameter logistic model (4PL) consists of parameters for the lower-limit of the response, the effective dose,
steepness of the curve, and the upper-limit of the response.

The wrapper function for the 4PL model should look like

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_4p_logistic

    def psi(theta):
        # Estimating equations for the 4PL model
        return ee_4p_logistic(theta=theta, X=dose_data, y=resp_data)

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the coefficients for the
4PL model and their variance

.. code::

    estr = MEstimator(psi, init=[np.min(resp_data),
                                 (np.max(resp_data)+np.min(resp_data)) / 2,
                                 (np.max(resp_data)+np.min(resp_data)) / 2,
                                 np.max(resp_data)])
    estr.estimate()

    print(estr.theta)
    print(estr.variance)

When you use 4PL, you may notice convergence errors. This estimating equation can be hard to optimize since it has
implicit bounds the root-finder isn't aware of. To avoid these issues, we can give the root-finder good starting values.

First, the upper limit should *always* be greater than the lower limit. Second, the ED50 should be between the lower
and upper limits. Third, the sign for the steepness depends on whether the response declines (positive) or the response
increases (negative). Finally, some solvers may be better suited to the problem, so try a few different options. With
decent initial values, I have found ``lm`` to be fairly reliable.

For the 4PL, good general starting values I have found are the following. For the lower-bound, give the minimum response
value as the initial. For ED50, give the mid-point between the maximum response and the minimum response. The initial
value for steepness is more difficult. Ideally, we would give a starting value of zero, but that will fail in this
4PL. Giving a larger starting value (between 2 to 8) works in this example. For the upper-bound, give the maximum
response value as the initial.

To summarize, be sure to examine your data (e.g., scatterplot). This will help to determine the initial starting values
for the root-finding procedure. Otherwise, you may come across a convergence error.


3-parameter Logistic
----------------------------

The 3-parameter logistic model (3PL) consists of parameters for the effective dose, steepness of the curve, and the
upper-limit of the response. Here, the lower-limit is pre-specified and is no longer being estimated.

The wrapper function for the 3PL model should look like

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_3p_logistic

    def psi(theta):
        # Estimating equations for the 3PL model
        return ee_3p_logistic(theta=theta, X=dose_data, y=resp_data,
                              lower=0)

Since the shortest a root of a plant could be zero, a lower limit of zero makes sense here.

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the coefficients for the
3PL model and their variance

.. code::

    estr = MEstimator(psi, init=[(np.max(resp_data)+np.min(resp_data)) / 2,
                                 (np.max(resp_data)+np.min(resp_data)) / 2,
                                 np.max(resp_data)])
    estr.estimate(solver='lm')

    print(estr.theta)
    print(estr.variance)

As before, you may notice convergence errors. This estimating equation can be hard to optimize since it has implicit
bounds the root-finder isn't aware of. To avoid these issues, we can give the root-finder good starting values.

For the 3PL, good general starting values I have found are the following. For ED50, give the mid-point between the
maximum response and the minimum response. The initial value for steepness is more difficult. Ideally, we would give a
starting value of zero, but that will fail in this 3PL. Giving a larger starting value (between 2 to 8) works in this
example. For the upper-bound, give the maximum response value as the initial.

To summarize, be sure to examine your data (e.g., scatterplot). This will help to determine the initial starting values
for the root-finding procedure. Otherwise, you may come across a convergence error.

2-parameter Logistic
----------------------------

The 2-parameter logistic model (2PL) consists of parameters for the effective dose, and steepness of the curve. Here,
the lower-limit and upper-limit are pre-specified and no longer being estimated.

The wrapper function for the 3PL model should look like

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_2p_logistic

    def psi(theta):
        # Estimating equations for the 2PL model
        return ee_2p_logistic(theta=theta, X=dose_data, y=resp_data,
                              lower=0, upper=8)

While a lower-limit of zero makes sense in this example, the upper-limit of 8 is poorly motivated (and thus this should
only be viewed as an example of the 2PL model and not how it should be applied in practice). Setting the limits as
constants should be motivated by substantive knowledge of the problem.

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the coefficients for the
2PL model and their variance

.. code::

    estr = MEstimator(psi, init=[(np.max(resp_data)+np.min(resp_data)) / 2,
                                 (np.max(resp_data)+np.min(resp_data)) / 2])
    estr.estimate(solver='lm')

    print(estr.theta)
    print(estr.variance)

As before, you may notice convergence errors. This estimating equation can be hard to optimize since it has implicit
bounds the root-finder isn't aware of. To avoid these issues, we can give the root-finder good starting values.

For the 2PL, good general starting values I have found are the following. For ED50, give the mid-point between the
maximum response and the minimum response. The initial value for steepness is more difficult. Ideally, we would give a
starting value of zero, but that will fail in this 2PL.

To summarize, be sure to examine your data (e.g., scatterplot). This will help to determine the initial starting values
for the root-finding procedure. Otherwise, you may come across a convergence error.


ED(:math:`\delta`)
----------------------------

In addition to the :math:`x`-parameter logistic models, an estimating equation to estimate a corresponding
:math:`\delta` effective dose is available. Notice that this estimating equation should be stacked with one of the
:math:`x`PL models. Here, we demonstrate with the 3PL model.

Here, our interest is in the following effective doses: 0.05, 0.10, 0.20, 0.80. The wrapper function for the 3PL model
and estimating equations for these effective doses are

.. code::

    def psi(theta):
        lower_limit = 0

        # Estimating equations for the 3PL model
        pl3 = ee_3p_logistic(theta=theta, X=d[:, 1], y=d[:, 0],
                             lower=lower_limit)

        # Estimating equations for the effective concentrations
        ed05 = ee_effective_dose_delta(theta[3], y=resp_data, delta=0.05,
                                       steepness=theta[0], ed50=theta[1],
                                       lower=lower_limit, upper=theta[2])
        ed10 = ee_effective_dose_delta(theta[4], y=resp_data, delta=0.10,
                                       steepness=theta[0], ed50=theta[1],
                                       lower=lower_limit, upper=theta[2])
        ed20 = ee_effective_dose_delta(theta[5], y=resp_data, delta=0.20,
                                       steepness=theta[0], ed50=theta[1],
                                       lower=lower_limit, upper=theta[2])
        ed80 = ee_effective_dose_delta(theta[6], y=resp_data, delta=0.80,
                                       steepness=theta[0], ed50=theta[1],
                                       lower=lower_limit, upper=theta[2])

        # Returning stacked estimating equations
        return np.vstack((pl3,
                          ed05,
                          ed10,
                          ed20,
                          ed80))

Notice that the estimating equations are stacked together in the order of the ``theta`` vector.


# Optimization procedure
mest = MEstimator(psi, init=[2, 1, 10, 1, 5])
mest.estimate(solver='lm')

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the coefficients for the
3PL model, the ED for the :math:`\delta` values, and their variance

.. code::

    midpoint = (np.max(resp_data)+np.min(resp_data)) / 2
    estr = MEstimator(psi, init=[midpoint,
                                 midpoint,
                                 np.max(resp_data),
                                 midpoint,
                                 midpoint,
                                 midpoint,
                                 midpoint])
    estr.estimate(solver='lm')
    print(estr.theta)
    print(estr.variance)

Since the ED for :math:`\delta`'s are transformations of the other parameters, there starting values are less important
(the root-finders are better at solving those equations). Again, we can make it easy on the solver by having the
starting point for each being the mid-point of the response values.


Causal Inference
=============================

To demonstrate the utility of M-estimation, particularly how estimating equations can be 'stacked' together, then
still have an appropriate variance estimator, several causal inference estimators are provided here.

It is recommended that you are familiar with causal inference (particularly the identification conditions of these
estimators) before using this utility widely. Causal inference is a difficult endeavour, dear user!

In the following examples, we will use the generic data example here, where W is confounding the A-Y relationship

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

The arguments for ``ee_gformula`` are the :math:`\theta` values, the covariates (including an intercept (``C``) and the
treatment (``A``)), the outcome values (``Y``), and the column index for the treatment in ``X``. Here, 1 designates the
second column (python uses zero-indexing), which corresponds to ``A`` in how the ``X`` data is formatted.

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

The ``ee_gformula`` supports both binary and continuous outcomes. Inside the function, it automatically detects whether
the outcome data is binary. If the outcome data is not binary, then it defaults to using a linear regression model
(but you can also force the use of a linear regression model for binary data by setting ``force_continuous=True``

To summarize, the key advantage of M-estimation here is that it *appropriately* estimates the variance. We do *not*
need to bootstrap in this case (and more generally if the sample size is sufficiently large).

Inverse probability weighting
-------------------------------------

Rather than modeling the outcome, we can choose the inverse probability weighting (IPW) estimator, which models the
probability of treatment. The estimating equations for the IPW estimator are also built-in to ``delicatessen``.

To load the estimating equations, we call

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_ipw

As with every built-in estimating equation, we will wrap it inside a function.

.. code::

    def psi(theta):
        return ee_ipw(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

The arguments for ``ee_ipw`` are the :math:`\theta` values, the covariates (including an intercept (C) and the treatment
(A)), the outcome values (Y), and the column index for the treatment in X. Here, 1 designates the second column
(python uses zero-indexing), which corresponds to 'A' in how the X data is formatted.

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

The ``ee_ipw`` supports both binary and continuous outcomes automatically. Both of these variable types are handled in
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

The estimating equations for the AIPW estimator are also provided in ``delicatessen``. To load the estimating equations,
we call

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_aipw

As always, we will wrap the built-in estimating equation inside a function.

.. code::

    def psi(theta):
        return ee_aipw(theta, X=d[['C', 'A', 'W']], y=d['Y'], treat_index=1)

The arguments for ``ee_aipw`` are the :math:`\theta` values, the covariates (including an intercept (C) and the
action (A)), the outcome values (Y), and the column index for the treatment in X. Here, 1 designates the second column
(python uses zero-indexing), which corresponds to ``'A'`` in how the X data is formatted.

Now we can call the M-estimator to solve for the values and the variance. Here, the initial values provided must be
3+*b*+*b-1* (where *b* is the number of columns in X). This is because the AIPW estimating equations output the
average treatment effect, risk under all-treated, risk under none-treated, and the outcome model coefficients, and
the treatment model coefficients.

As for starting values, it will likely be best practice to have the initial values set as ``[0., 0.5, 0.5, ...]`` in
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

References and Further Readings
===============================
Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.

Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust estimation of causal
effects. *American Journal of Epidemiology*, 173(7), 761-767.

Huber PJ. (1992). Robust estimation of a location parameter. In Breakthroughs in statistics (pp. 492-518).
Springer, New York, NY.

Inderjit, Streibig JC & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
allelopathy research. *Physiol Plant* 114, 422–428.
