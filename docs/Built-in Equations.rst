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


Survival
=============================
Estimating equations for parametric survival models are available in v0.3+. Currently available are: exponential and
weibull models, and accelerated failure time models (AFT). As commonly done in survival analysis, we can imagine that
each person has two unique times: their event time (:math:`T_i`) and their censoring time (:math:`C_i`). However, we
(the researcher) are only able to observe whichever one of those times occurs first. Therefore the observable data is
:math:`t_i = min(T_i, C_i)` and :math:`\delta_i = I(t_i = T_i)`.

For the basic survival models, we will use the following generated data set. In accordance with the description above,
each person is assigned two possible times and then we generate the observed data (``t`` and ``delta`` here).

.. code::

    import numpy as np
    import pandas as pd

    n = 100
    d = pd.DataFrame()
    d['C'] = np.random.weibull(a=1, size=n)
    d['C'] = np.where(d['C'] > 5, 5, d['C'])
    d['T'] = 0.8 * np.random.weibull(a=0.75, size=n)
    d['delta'] = np.where(d['T'] < d['C'], 1, 0)
    d['t'] = np.where(d['delta'] == 1, d['T'], d['C'])

For an introduction to survival analysis, I would recommend Collett D. (2015). "Modelling survival data in medical
research".

Exponential
-----------------------------
The exponential model is a one-parameter model, that stipulates the hazard of the event of interest is constant. While
often too restrictive of an assumption for widespread use, we demonstrate application here.

The wrapper function for the exponential model should look like

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_exponential_model, ee_exponential_measure

    def psi(theta):
        # Estimating equations for the exponential model
        return ee_exponential_model(theta=theta, t=d['t'], delta=d['delta'])

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the parameter for the
exponential model

.. code::

    estr = MEstimator(psi, init=[1., ])
    estr.estimate(solver='lm')

    print(estr.theta)
    print(estr.variance)

Here, the parameter for the exponential model should be non-negative (the optimizer does not know this), so a positive
value should be given to help the root-finding procedure along.

While the parameter for the exponential model may be of interest, we are often more interested in the one of the
functions over time. For example, we may want to plot the estimated survival function over time. ``delicatessen``
provides a function to estimate the survival (or other measures like density, risk, hazard, cumulative hazard) at
provided time points.

Below is how we could further generate a plot of the survival function from the estimated exponential model

.. code::

    import matplotlib.pyplot as plt

    resolution = 50
    time_spacing = list(np.linspace(0.01, 5, resolution))
    fast_inits = [0.5, ]*resolution

    def psi(theta):
        ee_exp = ee_exponential_model(theta=theta[0],
                                      t=times, delta=events)
        ee_surv = ee_exponential_measure(theta[1:], scale=theta[0],
                                         times=time_spacing, n=times.shape[0],
                                         measure="survival")
        return np.vstack((ee_exp, ee_surv))

    estr = MEstimator(psi, init=[1., ] + fast_inits)
    estr.estimate(solver="lm")

    # Creating plot of survival times
    ci = mestr.confidence_intervals()[1:, :]  # Extracting relevant CI
    plt.fill_between(time_spacing, ci[:, 0], ci[:, 1], alpha=0.2)
    plt.plot(time_spacing, mestr.theta[1:], '-')
    plt.show()


Here, we set the ``resolution`` to be 50. The resolution determines how many points along the survival function we are
evaluating (and thus determines how 'smooth' our plot will appear). As this involves the root-finding of multiple
values, it is important to help the root-finder along by providing good starting values. Since survival is bounded
between [0,1], we have all the initial values for those start at 0.5 (the middle). Furthermore, we could also consider
pre-washing the exponential model parameter (i.e., use the solution from the previous estimating equation).


Weibull
-----------------------------
The Weibull model is a generalization of the exponential model to two-parameters. Therefore, we now allow for the hazard
to vary over time (it can increase or decrease monotonically). While this assumption is also quite restrictive, it may
be more useful.

The wrapper function for the Weibull model should look like

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_weibull_model, ee_weibull_measure

    def psi(theta):
        # Estimating equations for the Weibull model
        return ee_weibull_model(theta=theta, t=d['t'], delta=d['delta'])

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the parameters for the
Weibull model

.. code::

    estr = MEstimator(psi, init=[1., 1.])
    estr.estimate(solver='lm')

    print(estr.theta)
    print(estr.variance)

Here, the parameters for the Weibull model should be non-negative (the optimizer does not know this), so a positive
value should be given to help the root-finding procedure along.

While the parameters for the Weibull model may be of interest, we are often more interested in the one of the
functions over time. For example, we may want to plot the estimated survival function over time. ``delicatessen``
provides a function to estimate the survival (or other measures like density, risk, hazard, cumulative hazard) at
provided time points.

Below is how we could further generate a plot of the survival function from the estimated Weibull model

.. code::

    import matplotlib.pyplot as plt

    resolution = 50
    time_spacing = list(np.linspace(0.01, 5, resolution))
    fast_inits = [0.5, ]*resolution

    def psi(theta):
        ee_wbf = ee_weibull_model(theta=theta[0:2],
                                  t=times, delta=events)
        ee_surv = ee_weibull_measure(theta[2:], scale=theta[0], shape=theta[1],
                                     times=time_spacing, n=times.shape[0],
                                     measure="survival")
        return np.vstack((ee_wbf, ee_surv))

    estr = MEstimator(psi, init=[1., 1., ] + fast_inits)
    estr.estimate(solver="lm")

    # Creating plot of survival times
    ci = mestr.confidence_intervals()[2:, :]  # Extracting relevant CI
    plt.fill_between(time_spacing, ci[:, 0], ci[:, 1], alpha=0.2)
    plt.plot(time_spacing, mestr.theta[2:], '-')
    plt.show()


Here, we set the ``resolution`` to be 50. The resolution determines how many points along the survival function we are
evaluating (and thus determines how 'smooth' our plot will appear). As this involves the root-finding of multiple
values, it is important to help the root-finder along by providing good starting values. Since survival is bounded
between [0,1], we have all the initial values for those start at 0.5 (the middle). Furthermore, we could also consider
pre-washing the Weibull model parameter (i.e., use the solution from the previous estimating equation).


Accelerated Failure Time
-----------------------------
Currently, only an AFT model with a Weibull (Weibull-AFT) is available for use. Plans are to add support for other
AFT. Unlike the previous exponential and Weibull models, the AFT models can further include covariates, where the effect
of a covariate is interpreted as an 'acceleration' factor. In the two sample case, the AFT can be thought of as the
following

.. math::

    S_1 (t) = S_0 (t / \sigma)

where :math:`\sigma^{-1} > 0` and is interpreted as the acceleration factor. One way to describe is that the risk of
the event in group 1 at :math:`t=1` is equivalent to group 0  at :math:`t=\sigma^{-1}`. Alternatively, you can interpret
the the AFT coefficient as the ratio of the mean survival times comparing group 1 to group 0. While involving parametric
assumptions, the AFT models have the advantage of providing a single summary measure (compared to nonparametric methods,
like Kaplan-Meier) but also being relatively easy to interpret (compared to semiparametric Cox models).

For the following examples, we generate some additional survival data with baseline covariates

.. code::

    n = 200
    d = pd.DataFrame()
    d['X'] = np.random.binomial(n=1, p=0.5, size=n)
    d['W'] = np.random.binomial(n=1, p=0.5, size=n)
    d['T'] = (1 / 1.25 + 1 / np.exp(0.5) * d['X']) * np.random.weibull(a=0.75, size=n)
    d['C'] = np.random.weibull(a=1, size=n)
    d['C'] = np.where(d['C'] > 10, 10, d['C'])
    d['delta'] = np.where(d['T'] < d['C'], 1, 0)
    d['t'] = np.where(d['delta'] == 1, d['T'], d['C'])

There are variations on the AFT model. These variations place parametric assumptions on the error distribution.

Weibull AFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Weibull AFT assumes that errors follow a Weibull distribution. Therefore, the Weibull AFT consists of a shape and
scale parameter (like the Weibull model from before) but not it further includes parameters for each covariate included
in the AFT model.

The wrapper function for the Weibull AFT model should look like

.. code::

    from delicatessen import MEstimator
    from delicatessen.estimating_equations import ee_aft_weibull, ee_aft_weibull_measure

    def psi(theta):
        # Estimating equations for the Weibull AFT model
        return ee_aft_weibull(theta=theta,
                              t=d['t'], delta=d['delta'],
                              X=d[['X', 'W']])

After creating the wrapper function, we can now call the M-Estimation procedure to estimate the parameters for the
Weibull model

.. code::

    estr = MEstimator(psi, init=[0., 0., 0., 0.])
    estr.estimate(solver='lm')

    print(estr.theta)
    print(estr.variance)

Unlike the previous models, the Weibull AFT model parameters are log-transformed. Therefore, starting values of zero
can be input for the root-finding procedure.

Here, ``theta[0]`` is the log-transformed intercept term for the shape parameter, and ``theta[-1]`` is the
log-transformed scale parameter. The middle terms (``theta[1:3]`` in this case) corresponds to the acceleration factors
for the covariates as the input order in ``X``. Therefore, ``theta[1]`` is the acceleration factor for ``'X'`` and
``theta[2]`` is the acceleration factor for ``'W'``.

While the parameters for the Weibull model may be of interest, we are often more interested in the one of the
functions over time. For example, we may want to plot the estimated survival function over time. ``delicatessen``
provides a function to estimate the survival (or other measures like density, risk, hazard, cumulative hazard) at
provided time points.

Below is how we could further generate a plot of the survival function from the estimated Weibull AFT model. Unlike the
other survival models, we also need to specify the covariate pattern of interest. Here, we will generate the survival
function when both :math:`X=1` and :math:`W=1`

.. code::

    import matplotlib.pyplot as plt

    resolution = 50
    time_spacing = list(np.linspace(0.01, 5, resolution))
    fast_inits = [0.5, ]*resolution
    dc = d.copy()
    dc['X'] = 1
    dc['W'] = 1

    def psi(theta):
        ee_aft = ee_aft_weibull(theta=theta,
                                t=d['t'], delta=d['delta'],
                                X=d[['X', 'W']])
        pred_surv_t = ee_aft_weibull_measure(theta=theta[4:], X=dc[['X', 'W']],
                                             times=time_spacing, measure='survival',
                                             mu=theta[0], beta=theta[1:3], sigma=theta[3])
        return np.vstack((ee_aft, pred_surv_t))

    estr = MEstimator(psi, init=[0., 0., 0., 0., ] + fast_inits)
    estr.estimate(solver="lm")

    # Creating plot of survival times
    ci = mestr.confidence_intervals()[4:, :]  # Extracting relevant CI
    plt.fill_between(time_spacing, ci[:, 0], ci[:, 1], alpha=0.2)
    plt.plot(time_spacing, mestr.theta[4:], '-')
    plt.show()

Here, we set the ``resolution`` to be 50. The resolution determines how many points along the survival function we are
evaluating (and thus determines how 'smooth' our plot will appear).

As this involves the root-finding of multiple values, it is important to help the root-finder along by providing good
starting values. Since survival is bounded between [0,1], we have all the initial values for those start at 0.5 (the
middle). Furthermore, models like Weibull AFT should be used with pre-washing the AFT model parameters (i.e., use the
solution from the previous estimating equation).


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
    estr.estimate(solver='lm')

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
:math:`\delta` effective dose is available. Notice that this estimating equation should be stacked with one of
the :math:`x`-PL models. Here, we demonstrate with the 3PL model.

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

This next section describes a the available estimators for the causal mean. These estimators all rely on specific
identification conditions to be able to interpret the estimate of the mean (or mean difference) as an estimate of the
causal mean. For information on these assumptions, I recommend this
`paper<https://www.ncbi.nlm.nih.gov/labs/pmc/articles/PMC2652882/>`_ as a general introduction.

This section procedures that the identification conditions have been previously deliberated, and the causal mean is
identified and is estimable (see this `paper<https://arxiv.org/abs/2108.11342>`_ or this
`paper<https://arxiv.org/abs/1904.02826>`_ for more information on this concept).

With that aside, let's proceed through the available estimators of the causal means. In the following examples, we will
use the generic data example here, where :math:`Y(a)` is independent of :math:`A` conditional on :math:`W`. Below is
a sample data set

.. code::

    n = 200
    d = pd.DataFrame()
    d['W'] = np.random.binomial(1, p=0.5, size=n)
    d['A'] = np.random.binomial(1, p=(0.25 + 0.5*d['W']), size=n)
    d['Ya0'] = np.random.binomial(1, p=(0.75 - 0.5*d['W']), size=n)
    d['Ya1'] = np.random.binomial(1, p=(0.75 - 0.5*d['W'] - 0.1*1), size=n)
    d['Y'] = (1-d['A'])*d['Ya0'] + d['A']*d['Ya1']
    d['C'] = 1

Here, we don't get to see the potential outcomes :math:`Y(a)`, but instead estimate the mean under different plans
using the observed data, :math:`Y,A,W`.

Inverse probability weighting
-------------------------------------

First, we use the inverse probability weighting (IPW) estimator, which models the probability of :math:`A` conditional
on :math:`W`. In general, the IPW estimator for the mean difference can be written as

.. math::

    \frac{1}{n} \sum_{i=1}^n \frac{Y_i A_i}{Pr(A_i = 1 | W_i; \hat{\alpha})} - \frac{1}{n}
    \sum_{i=1}^n \frac{Y_i (1-A_i)}{Pr(A_i = 0 | W_i; \hat{\alpha})}

In ``delicatessen``, the built-in IPW estimator consists of 4 estimating equations, with both binary and continuous
outcomes supported by ``ee_ipw`` (since we are using the Horwitz-Thompson estimator). The stacked estimating equations
are

.. math::

    \sum_i^n \psi_d(Y_i, A_i, \pi_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

    \sum_i^n \psi_1(Y_i, A_i, \pi_i, \theta_1) = \sum_i^n \frac{A_i \times Y_i}{\pi_i} - \theta_1 = 0

    \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n \frac{(1-A_i) \times Y_i}{1-\pi_i} - \theta_2 = 0

    \sum_i^n \psi_g(A_i, W_i, \theta) = \sum_i^n (A_i - expit(W_i^T \alpha)) W_i = 0

where :math:`\theta_1` is the average causal effect, :math:`\theta_2` is the mean under the plan where
:math:`A=1` for everyone, :math:`\theta_3` is the mean under the plan where :math:`A=0` for everyone, and
:math:`\alpha` is the parameters for the logistic model used to estimate the propensity scores.

To load the estimating equations,

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_ipw

The estimating equation is then wrapped inside the wrapper ``psi`` function. Notice that the estimating equation has
4 non-optional inputs: the parameter values, the outcomes, the actions, and the covariates to model the propensity
scores with.

.. code::

    def psi(theta):
        return ee_ipw(theta,                 # Parameters
                      y=d['Y'],              # Outcome
                      A=d['A'],              # Action (exposure, treatment, etc.)
                      W=d[['C', 'W']])       # Covariates for PS model

Note that we add an intercept to the logistic model by adding a column of 1's via ``d['C']``.

Here, the initial values provided must be 3+*b* (where *b* is the number of columns in W). For binary
outcomes, it will likely be best practice to have the initial values set as ``[0., 0.5, 0.5, ...]``. followed by b
``0.``'s. For continuous outcomes, all ``0.`` can be used instead. Furthermore, a logistic model for the propensity
scores could be optimized outside of ``delicatessen`` and those (pre-washed) regression estimates can be passed as
initial values to speed up optimization.

Now we can call the M-estimator to solve for the values and the variance.

.. code::

    estr = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0.])
    estr.estimate(solver='lm')

After successful optimization, we can inspect the estimated values.

.. code::

    estr.theta[0]    # causal mean difference of 1 versus 0
    estr.theta[1]    # causal mean under X1
    estr.theta[2]    # causal mean under X0
    estr.theta[3:]   # logistic regression coefficients

The variance and Wald-type confidence intervals can also be output via

.. code::

    estr.variance
    estr.confidence_intervals()

The IPW estimators demonstrates a key advantage of M-Estimation. The stacked estimating equations means that the
sandwich variance correctly incorporates the uncertainty in estimation of the propensity scores into the parameter(s)
of interest (e.g., average causal effect). Therefore, we do not have to rely on the nonparametric bootstrap
(computationally cumbersome) or the GEE-trick (conservative estimate of the variance for the average causal effect).


G-computation
----------------------------

Second, we use g-computation, which models :math:`Y` conditional on :math:`A` and :math:`W`. In general, g-computation
for the mean difference can be written as

.. math::

    \frac{1}{n} \sum_{i=1}^n m_1(W_i; \hat{\beta}) - \frac{1}{n} \sum_{i=1}^n m_0(W_i; \hat{\beta})

where :math:`m_a(W_i; \hat{\beta}) = E[Y_i|A_i=a,W_i; \hat{\beta}]`. In ``delicatessen``, the built-in g-computation
consists of either 2 estimating equations or 4 estimating equations, with both binary and continuous outcomes supported.
The 2 stacked estimating equations are

.. math::

    \sum_i^n \psi_1(Y_i, X_i, \theta_1) = \sum_i^n g(\hat{Y}_i^a) - \theta_1 = 0

    \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - \text{expit}(X_i^T \theta)) X_i = 0


where :math:`\theta_1` is the mean under the action :math:`a`, and :math:`\beta` is the parameters for the regression
model used to estimate the outcomes. Notice that the g-computation procedure supports generic deterministic plans
(e.g., set :math:`A=1` for all, set :math:`A=0` for all, set :math:`A=1` if :math:`W=1` otherwise :math:`A=0`, etc.).
These plans are more general than those allowed by either the built-in IPW or built-in AIPW estimating equations.

The 4 stacked estimating equations instead compare the mean difference between two action plans. The estimating
equations are

.. math::

    \sum_i^n \psi_1(Y_i, X_i, \theta_1) = \sum_i^n (\theta_2 - \theta_3) - \theta_1 = 0

    \sum_i^n \psi_1(Y_i, X_i, \theta_2) = \sum_i^n g(\hat{Y}_i^a) - \theta_2 = 0

    \sum_i^n \psi_1(Y_i, X_i, \theta_3) = \sum_i^n g(\hat{Y}_i^a) - \theta_3 = 0

    \sum_i^n \psi_m(Y_i, X_i, \theta) = \sum_i^n (Y_i - \text{expit}(X_i^T \theta)) X_i = 0


where :math:`\theta_1` is the average causal effect, :math:`\theta_2` is the mean under the first plan, :math:`\theta_3`
is the mean under the second, and :math:`\beta` is the parameters for the regression model used to predict the
outcomes.

To load the estimating equations,

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_gformula

The estimating equation is then wrapped inside the wrapper ``psi`` function. In the first example, we focus on
estimating the average causal effect. Notice that for ``ee_gformula`` some additional data prep is necessary.
Specifically, we need to create a copy of the data set where ``A`` is set to the value our plan dictates
(e.g., ``A=1``). Below is code that does this step and creates the wrapper function

.. code::

    # Creating data under the plans
    d1 = d.copy()
    d1['A'] = 1
    d0 = d.copy()
    d0['A'] = 0

    # Creating interaction terms
    d['AxW'] = d['A'] * d['W']
    d1['AxW'] = d1['A'] * d1['W']
    d0['AxW'] = d0['A'] * d0['W']

    def psi(theta):
        return ee_gformula(theta,                        # Parameters
                           y=d['Y'],                     # Outcome
                           X=d[['C', 'A', 'W', 'AxW']],  # Observed
                           X=d1[['C', 'A', 'W', 'AxW']], # Plan 1
                           X=d0[['C', 'A', 'W', 'AxW']]) # Plan 2

Note that we add an intercept to the outcome model by adding a column of 1's via ``d['C']``.

Here, the initial values provided must be 3+*b* (where *b* is the number of columns in X). For binary
outcomes, it will likely be best practice to have the initial values set as ``[0., 0.5, 0.5, ...]``. followed by b
``0.``'s. For continuous outcomes, all ``0.`` can be used instead. Furthermore, a regression model for the outcomes
could be optimized outside of ``delicatessen`` and those (pre-washed) regression estimates can be passed as
initial values to speed up optimization.

Now we can call the M-estimator to solve for the values and the variance.

.. code::

    estr = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0., 0.])
    estr.estimate(solver='lm')

After successful optimization, we can inspect the estimated values.

.. code::

    estr.theta[0]    # causal mean difference of 1 versus 0
    estr.theta[1]    # causal mean under X1
    estr.theta[2]    # causal mean under X0
    estr.theta[3:]   # regression coefficients

The variance and Wald-type confidence intervals can also be output via

.. code::

    estr.variance
    estr.confidence_intervals()

Again, a key advantage of M-Estimation is demonstrated here. The stacked estimating equations means that the
sandwich variance correctly incorporates the uncertainty in estimation of the outcome model into the parameter(s)
of interest (e.g., average causal effect). Therefore, we do not have to rely on the nonparametric bootstrap
(computationally cumbersome).

As a second example, we now demonstrate the flexbility of ``ee_gformula`` to estimate other plans. Here, we estimate
the causal mean under the plan where only those with :math:`W=1` have :math:`A=1`. As before, we need to generate
this distribution of covariates and wrap the built-in estimating equations.

.. code::

    # Creating data under the plans
    da = d.copy()
    da['A'] = np.where(da['W'] == 1, 1, 0)

    # Creating interaction terms
    d['AxW'] = d['A'] * d['W']
    da['AxW'] = da['A'] * da['W']

    def psi(theta):
        return ee_gformula(theta,                        # Parameters
                           y=d['Y'],                     # Outcome
                           X=d[['C', 'A', 'W', 'AxW']],  # Observed
                           X=da[['C', 'A', 'W', 'AxW']]) # Plan

Now we can call the M-estimator to solve for the values and the variance.

.. code::

    estr = MEstimator(psi, init=[0., 0.5, 0.5, 0., 0., 0., 0.])
    estr.estimate(solver='lm')

After successful optimization, we can inspect the estimated values.

.. code::

    estr.theta[0]    # causal mean under X1
    estr.theta[1:]   # regression coefficients


Augmented inverse probability weighting
----------------------------------------------

Finally, we use the augmented inverse probability weighting (AIPW) esitmator, which incorporates both a model for
:math:`Y` conditional on :math:`A` and :math:`W`, and a model for :math:`A` conditional on :math:`W`. The AIPW estimator
for the mean difference can be written as

.. math::

    \frac{1}{n} \sum_{i=1}^n \frac{A_i \times Y_i}{\pi_i} - \frac{m_1(W_i; \hat{\beta})(A_i-\pi_i}{\pi_i} -
    \frac{1}{n} \sum_{i=1}^n \frac{(1-A_i) \times Y_i}{1-\pi_i} + \frac{m_0(W_i; \hat{\beta})(A_i-\pi_i}{1-\pi_i}


where :math:`m_a(W_i; \hat{\beta}) = E[Y_i|A_i=a,W_i; \hat{\beta}]`, and
:math:`\pi_i = Pr(A_i = 1 | W_i; \hat{\alpha})`. In ``delicatessen``, the built-in AIPW estimator consists of 5
estimating equations, with both binary and continuous outcomes supported. Similar to IPW (and unlike g-computation),
the built-in AIPW estimator only supports the average causal effect as the parameter of interest.

The stacked estimating equations are

.. math::

    \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_0) = \sum_i^n (\theta_1 - \theta_2) - \theta_0 = 0

    \sum_i^n \psi_1(Y_i, A_i, W_i, \pi_i, \theta_1) = \sum_i^n (\frac{A_i \times Y_i}{\pi_i} -
    \frac{\hat{Y^1}(A_i-\pi_i}{\pi_i}) - \theta_1 = 0

    \sum_i^n \psi_0(Y_i, A_i, \pi_i, \theta_2) = \sum_i^n (\frac{(1-A_i) \times Y_i}{1-\pi_i} +
    \frac{\hat{Y^0}(A_i-\pi_i}{1-\pi_i})) - \theta_2 = 0

    \sum_i^n \psi_g(A_i, W_i, \alpha) = \sum_i^n (A_i - expit(W_i^T \alpha)) W_i = 0

    \sum_i^n \psi_m(Y_i, X_i, \beta) = \sum_i^n (Y_i - X_i^T \beta) X_i = 0

where :math:`\theta_1` is the average causal effect, :math:`\theta_2` is the mean under the first plan, :math:`\theta_3`
is the mean under the second, :math:`\alpha` is the parameters for the propensity score logistic model, and
:math:`\beta` is the parameters for the regression model used to predict the outcomes. For binary outcomes, the final
estimating equation is replaced with the logistic model analog.

To load the estimating equations,

.. code::

    from delicatessen import MEstimators
    from delicatessen.estimating_equations import ee_aipw

The estimating equation is then wrapped inside the wrapper ``psi`` function. Like ``ee_gformula``, ``ee_aipw`` requires
some additional data prep. Specifically, we need to create a copy of the data set where :math:`A=1` for everyone and another
copy where :math:`A=0` for everyone. Below is code that does this step and creates the wrapper function

.. code::

    # Creating data under the plans
    d1 = d.copy()
    d1['A'] = 1
    d0 = d.copy()
    d0['A'] = 0

    # Creating interaction terms
    d['AxW'] = d['A'] * d['W']
    d1['AxW'] = d1['A'] * d1['W']
    d0['AxW'] = d0['A'] * d0['W']

    def psi(theta):
        return ee_gformula(theta,                        # Parameters
                           y=d['Y'],                     # Outcome
                           A=d['A'],                     # Action
                           W=d[['C', 'W']],              # PS model
                           X=d[['C', 'A', 'W', 'AxW']],  # Outcome model
                           X=d1[['C', 'A', 'W', 'AxW']], # Plan A=1
                           X=d0[['C', 'A', 'W', 'AxW']]) # Plan A=0

Note that we add an intercept to the outcome model by adding a column of 1's via ``d['C']``.

Here, the initial values provided must be 3+*b*+*c* (where *b* is the number of columns in W and *c* is the number of
columns in X). For binary outcomes, it will likely be best practice to have the initial values set as
``[0., 0.5, 0.5, ...]``. followed by b ``0.``'s. For continuous outcomes, all ``0.`` can be used instead. Furthermore,
a regression models could be optimized outside of ``delicatessen`` and those (pre-washed) regression estimates can be
passed as initial values to speed up optimization.

Now we can call the M-estimator to solve for the values and the variance.

.. code::

    estr = MEstimator(psi, init=[0., 0.5, 0.5,
                                 0., 0.,
                                 0., 0., 0., 0.])
    estr.estimate(solver='lm')

After successful optimization, we can inspect the estimated values.

.. code::

    estr.theta[0]     # causal mean difference of 1 versus 0
    estr.theta[1]     # causal mean under A=1
    estr.theta[2]     # causal mean under A=0
    estr.theta[3:5]   # propensity score regression coefficients
    estr.theta[5:]    # outcome regression coefficients

The variance and Wald-type confidence intervals can also be output via

.. code::

    estr.variance
    estr.confidence_intervals()

Here, the M-Estimation sandwich variance is the same as the influence-curve-based variance estimator. Either of these
approaches correctly incorporates the uncertainty in estimation of the outcome model into the parameter(s) of interest
(e.g., average causal effect). Therefore, we do not have to rely on the nonparametric bootstrap (computationally
cumbersome).

Additional Examples
-------------------------------
Additional examples are provided `here<https://github.com/pzivich/Delicatessen/tree/main/tutorials>`_.

References and Further Readings
===============================
Boos DD, & Stefanski LA. (2013). M-estimation (estimating equations). In Essential Statistical Inference
(pp. 297-337). Springer, New York, NY.

Cole SR, & Hernán MA. (2008). Constructing inverse probability weights for marginal structural models.
*American Journal of Epidemiology*, 168(6), 656-664.

Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust estimation of causal
effects. *American Journal of Epidemiology*, 173(7), 761-767.

Hernán MA, & Robins JM. (2006). Estimating causal effects from epidemiological data.
*Journal of Epidemiology & Community Health*, 60(7), 578-586.

Huber PJ. (1992). Robust estimation of a location parameter. In Breakthroughs in statistics (pp. 492-518).
Springer, New York, NY.

Inderjit, Streibig JC & Olofsdotter M. (2002). Joint action of phenolic acid mixtures and its significance in
allelopathy research. *Physiol Plant* 114, 422–428.

Snowden JM, Rose S, & Mortimer KM. (2011). Implementation of G-computation on a simulated data set: demonstration
of a causal inference technique. *American Journal of Epidemiology*, 173(7), 731-738.
