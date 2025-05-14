Utilities
=====================
For manipulation of output or inputs, there are several basic utility functionalities for transformation of variables,
predicted parameters, or computations. Some are used internally by the built-in estimating equations but these methods
are also made available to users.

Data transformations
---------------------

.. currentmodule:: delicatessen.utilities

.. autosummary::
  :toctree: generated/

  logit
  inverse_logit
  identity
  robust_loss_functions
  spline
  polygamma
  digamma
  standard_normal_cdf
  standard_normal_pdf


Design matrices
---------------------

.. autosummary::
  :toctree: generated/

  additive_design_matrix


Model predictions
---------------------

.. currentmodule:: delicatessen.utilities

.. autosummary::
  :toctree: generated/

  regression_predictions
  aft_predictions_individual
  aft_predictions_function


Differentiation
---------------------

.. currentmodule:: delicatessen.derivative

.. autosummary::
  :toctree: generated/

  approx_differentiation
  auto_differentiation


Variance Estimators
---------------------------

.. currentmodule:: delicatessen.sandwich

.. autosummary::
   :toctree: generated/

   compute_sandwich
   delta_method
