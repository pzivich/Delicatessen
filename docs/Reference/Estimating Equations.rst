Estimating Equations
=====================
To ease use, ``delicatessen`` comes with several built-in estimating equation functionalities. These basic
functionalities cover a variety of common use cases. Below is reference documentation for currently available
estimating equations.


Basic
-------------------

.. currentmodule:: delicatessen.estimating_equations

.. autosummary::
  :toctree: generated/

  ee_mean
  ee_mean_robust
  ee_mean_variance


Regression
-------------------

.. currentmodule:: delicatessen.estimating_equations

.. autosummary::
  :toctree: generated/

  ee_linear_regression
  ee_logistic_regression
  ee_poisson_regression
  ee_robust_linear_regression


Survival
-------------------

.. currentmodule:: delicatessen.estimating_equations

.. autosummary::
  :toctree: generated/

  ee_exponential_model
  ee_exponential_measure
  ee_weibull_model
  ee_weibull_measure
  ee_aft_weibull
  ee_aft_weibull_measure


Dose Response
-------------------

.. currentmodule:: delicatessen.estimating_equations

.. autosummary::
  :toctree: generated/

  ee_4p_logistic
  ee_3p_logistic
  ee_2p_logistic
  ee_effective_dose_delta


Causal Inference
-------------------

.. currentmodule:: delicatessen.estimating_equations

.. autosummary::
  :toctree: generated/

  ee_gformula
  ee_ipw
  ee_aipw
