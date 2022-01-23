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
  ee_robust_linear_regression


Dose Response
-------------------

.. currentmodule:: delicatessen.estimating_equations

.. autosummary::
  :toctree: generated/

  ee_4p_logistic
  ee_3p_logistic
  ee_2p_logistic
  ee_effective_dose_alpha


Causal Inference
-------------------

.. currentmodule:: delicatessen.estimating_equations

.. autosummary::
  :toctree: generated/

  ee_gformula
  ee_ipw
  ee_aipw
