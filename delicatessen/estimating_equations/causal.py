#####################################################################################################################
# Estimating functions for causal inference applications
#####################################################################################################################

import numpy as np

from .basic import ee_mean
from .regression import ee_regression, ee_glm
from delicatessen.utilities import logit, inverse_logit, identity


#################################################################
# Causal Inference (ATE) Estimating Equations


def ee_gformula(theta, y, X, X1, X0=None, force_continuous=False):
    r"""Estimating equations for the g-formula (or g-computation)."""
    X = np.asarray(X)
    y = np.asarray(y)
    X1 = np.asarray(X1)

    if X.shape != X1.shape:
        raise ValueError("The dimensions of X and X1 must be the same.")

    if X0 is None:
        mu1 = theta[0]
        beta = theta[1:]
    else:
        X0 = np.asarray(X0)
        if X.shape != X0.shape:
            raise ValueError("The dimensions of X and X0 must be the same.")
        mud = theta[0]
        mu1 = theta[1]
        mu0 = theta[2]
        beta = theta[3:]

    if np.isin(y, [0, 1]).all() and not force_continuous:
        model = 'logistic'
        transform = inverse_logit
    else:
        model = 'linear'
        transform = identity

    preds_reg = ee_regression(theta=beta, X=X, y=y, model=model)
    ya1 = transform(np.dot(X1, beta)) - mu1

    if X0 is None:
        return np.vstack((ya1[None, :], preds_reg))
    else:
        ya0 = transform(np.dot(X0, beta)) - mu0
        ace = np.ones(y.shape[0]) * (mu1 - mu0) - mud
        return np.vstack((ace, ya1[None, :], ya0[None, :], preds_reg))


def ee_ipw(theta, y, A, W, truncate=None, weights=None):
    r"""Estimating equation for inverse probability weighting (IPW) estimator."""
    W = np.asarray(W)
    A = np.asarray(A)
    y = np.asarray(y)
    beta = theta[3:]

    preds_reg = ee_regression(theta=beta, X=W, y=A, model='logistic')
    pi = inverse_logit(np.dot(W, beta))

    if truncate is not None:
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    if weights is None:
        weights = 1

    ya1 = (A * y) / pi * weights - theta[1]
    ya0 = ((1 - A) * y) / (1 - pi) * weights - theta[2]
    ate = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

    return np.vstack((ate, ya1[None, :], ya0[None, :], preds_reg))


def ee_ipw_msm(theta, y, A, W, V, distribution, link, hyperparameter=None, truncate=None, weights=None):
    r"""Estimating equation for marginal structural model using IPW."""
    W = np.asarray(W)
    V = np.asarray(V)
    A = np.asarray(A)
    y = np.asarray(y)
    alpha = theta[:V.shape[1]]
    beta = theta[V.shape[1]:]

    preds_reg = ee_regression(theta=beta, X=W, y=A, model='logistic', weights=None)
    pi = inverse_logit(np.dot(W, beta))

    if truncate is not None:
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    ipw = np.where(A == 1, 1 / pi, 1 / (1 - pi))
    if weights is not None:
        ipw = ipw * weights

    ee_msm = ee_glm(theta=alpha, X=V, y=y, distribution=distribution, link=link,
                    hyperparameter=hyperparameter, weights=ipw, offset=None)

    return np.vstack((ee_msm, preds_reg))


def ee_aipw(theta, y, A, W, X, X1, X0, truncate=None, force_continuous=False):
    r"""Estimating equation for augmented inverse probability weighting (AIPW) estimator."""
    y = np.asarray(y)
    A = np.asarray(A)
    W = np.asarray(W)
    X = np.asarray(X)
    X1 = np.asarray(X1)
    X0 = np.asarray(X0)

    if X.shape != X1.shape:
        raise ValueError("The dimensions of X and X1 must be the same.")
    if X.shape != X0.shape:
        raise ValueError("The dimensions of X and X0 must be the same.")

    mud = theta[0]
    mu1 = theta[1]
    mu0 = theta[2]
    alpha = theta[3:3 + W.shape[1]]
    beta = theta[3 + W.shape[1]:]

    pi_model = ee_regression(theta=alpha, X=W, y=A, model='logistic')
    pi = inverse_logit(np.dot(W, alpha))

    if truncate is not None:
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    if np.isin(y, [0, 1]).all() and not force_continuous:
        model = 'logistic'
        transform = inverse_logit
    else:
        model = 'linear'
        transform = identity

    m_model = ee_regression(theta=beta, y=y, X=X, model=model)
    ya1 = transform(np.dot(X1, beta))
    ya0 = transform(np.dot(X0, beta))

    ace = np.ones(y.shape[0]) * (mu1 - mu0) - mud
    y1_star = (y * A / pi - ya1 * (A - pi) / pi) - mu1
    y0_star = (y * (1 - A) / (1 - pi) + ya0 * (A - pi) / (1 - pi)) - mu0

    return np.vstack((ace, y1_star[None, :], y0_star[None, :], pi_model, m_model))


def ee_cbps_ate(theta, y, A, X, truncate=None, weights=None):
    r"""Estimating equation for the CBPS estimator for the average treatment effect (ATE).
    
    CBPS estimates propensity scores by directly optimizing covariate balance.
    
    The stacked estimating equations are:
    
    EE1: (theta[1] - theta[2]) - theta[0]  (ATE)
    EE2: A*Y/pi - theta[1]  (mean under A=1)
    EE3: (1-A)*Y/(1-pi) - theta[2]  (mean under A=0)
    EE4: (A - pi)/(pi*(1-pi)) * X  (covariate balancing condition)
    
    Parameters
    ----------
    theta : ndarray
        Theta consists of 3+`b` values.
    y : ndarray
        1-dimensional vector of `n` observed values.
    A : ndarray
        1-dimensional vector of `n` observed values (0 or 1).
    X : ndarray
        2-dimensional vector of `n` observed values for confounders.
    truncate : None, tuple
        Bounds to truncate the estimated probabilities.
    weights : ndarray, None
        1-dimensional vector of n weights.

    Returns
    -------
    array : NumPy array

    References
    ----------
    Imai K, & Ratkovic M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society: Series B, 76(1), 243-263.
    """
    X = np.asarray(X)
    A = np.asarray(A)
    y = np.asarray(y)
    beta = theta[3:]

    pi = inverse_logit(np.dot(X, beta))

    if truncate is not None:
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    if weights is None:
        weights = 1

    # CBPS covariate balancing condition
    cbps_balance = ((A - pi) / (pi * (1 - pi)))[:, None] * X
    cbps_model = cbps_balance.T

    # IPW estimating equations
    ya1 = (A * y) / pi * weights - theta[1]
    ya0 = ((1 - A) * y) / (1 - pi) * weights - theta[2]
    ate = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

    return np.vstack((ate, ya1[None, :], ya0[None, :], cbps_model))


def ee_cbps_att(theta, y, A, X, truncate=None, weights=None):
    r"""Estimating equation for the CBPS estimator for the average treatment effect on the treated (ATT).
    
    CBPS estimates propensity scores by directly optimizing covariate balance.
    
    The stacked estimating equations are:
    
    EE1: theta[1] - theta[0]  (ATT)
    EE2: A*Y - theta[1]*A  (mean among treated)
    EE3: (1-A)*(Y - theta[2])/pi  (counterfactual mean)
    EE4: (A - pi)/(1-pi) * X  (covariate balancing condition for ATT)
    
    Parameters
    ----------
    theta : ndarray
        Theta consists of 3+`b` values.
    y : ndarray
        1-dimensional vector of `n` observed values.
    A : ndarray
        1-dimensional vector of `n` observed values (0 or 1).
    X : ndarray
        2-dimensional vector of `n` observed values for confounders.
    truncate : None, tuple
        Bounds to truncate the estimated probabilities.
    weights : ndarray, None
        1-dimensional vector of n weights.

    Returns
    -------
    array : NumPy array

    References
    ----------
    Imai K, & Ratkovic M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society: Series B, 76(1), 243-263.
    """
    X = np.asarray(X)
    A = np.asarray(A)
    y = np.asarray(y)
    beta = theta[3:]

    pi = inverse_logit(np.dot(X, beta))

    if truncate is not None:
        if truncate[0] > truncate[1]:
            raise ValueError("truncate values must be specified in ascending order")
        pi = np.clip(pi, a_min=truncate[0], a_max=truncate[1])

    if weights is None:
        weights = 1

    # CBPS covariate balancing condition for ATT
    cbps_balance = ((A - pi) / (1 - pi))[:, None] * X
    cbps_model = cbps_balance.T

    # IPW estimating equations for ATT
    mu1 = A * y * weights - theta[1] * A * weights
    mu0 = (1 - A) * (y - theta[2]) / pi * weights
    att = np.ones(y.shape[0]) * (theta[1] - theta[2]) - theta[0]

    return np.vstack((att, mu1[None, :], mu0[None, :], cbps_model))


#################################################################
# Causal Inference (Instrumental) Estimating Equations


def ee_iv_causal(theta, y, A, Z, weights=None):
    r"""Estimating equation for instrumental variable (IV) analysis."""
    y = np.asarray(y)
    a = np.asarray(A)
    z = np.asarray(Z)

    if weights is None:
        weight = 1
    else:
        weight = np.asarray(weights)

    ee_prz = weight * (z - theta[1])
    ee_iva = weight * (y - theta[0] * a) * (z - theta[1])

    return np.vstack([ee_iva, ee_prz])


def ee_2sls(theta, y, A, Z, W=None, weights=None):
    r"""Estimating equations for Two-Stage Least Squares (2SLS) for IV analysis."""
    y = np.asarray(y)
    a = np.asarray(A)
    Z = np.asarray(Z)

    if W is None:
        id2s = 1
    else:
        W = np.asarray(W)
        id2s = 1 + W.shape[1]

    beta = theta[:id2s]
    alpha = theta[id2s:]

    if W is not None:
        dmatrix1 = np.c_[Z, W]
    else:
        dmatrix1 = Z

    a_hat = np.dot(dmatrix1, alpha)

    if W is not None:
        dmatrix2 = np.c_[a_hat, W]
    else:
        dmatrix2 = a_hat[:, None]

    ee_stageone = ee_regression(theta=alpha, y=a, X=dmatrix1, model='linear', weights=weights)
    ee_stagetwo = ee_regression(theta=beta, y=y, X=dmatrix2, model='linear', weights=weights)

    return np.vstack([ee_stagetwo, ee_stageone])


#################################################################
# Causal Inference (SMM) Estimating Equations


def ee_gestimation_snmm(theta, y, A, W, V, X=None, model='linear', weights=None):
    r"""Estimating equations for g-estimation of structural mean models (SMMs)."""
    y = np.asarray(y)[:, None]
    A = np.asarray(A)
    W = np.asarray(W)
    V = np.asarray(V)
    eq_add = []
    pdiv = V.shape[1]
    qdiv = W.shape[1] + pdiv

    if weights is None:
        weight = 1
    else:
        weight = np.asarray(weights)

    phi = np.asarray(theta[0:pdiv])[:, None]
    alpha = np.asarray(theta[pdiv:qdiv])

    if X is not None:
        beta = np.asarray(theta[qdiv:])

    if model.lower() == 'linear':
        h_phi = y - np.dot(V * A[:, None], phi)
        y_transform = identity
    elif model.lower() == 'poisson':
        h_phi = y * np.exp(-1 * np.dot(V * A[:, None], phi))
        y_transform = np.exp
    else:
        raise ValueError("model='" + str(model) + "' is not supported. Options: linear, poisson")

    ee_log = ee_regression(theta=alpha, X=W, y=A, model='logistic', weights=weights)
    pi = inverse_logit(np.dot(W, alpha))
    a_resid = (A - pi)[:, None]

    if X is not None:
        X = np.asarray(X)
        ee_out = ee_regression(theta=beta, X=X, y=h_phi[:, 0], model=model, weights=weights)
        yhat = y_transform(np.dot(X, beta))[:, None]
        eq_add = [ee_out]
    else:
        yhat = 0

    y0_resid = h_phi - yhat
    ee_smm = weight * (a_resid * y0_resid * V).T

    return np.vstack([ee_smm, ee_log] + eq_add)


def ee_gestimation_snmm_iv(theta, y, Z, A, W, V, X=None, model='linear',
                          model_instrument='logistic', weights=None):
    r"""G-estimation of SMMs with an IV."""
    y = np.asarray(y)[:, None]
    Z = np.asarray(Z)
    A = np.asarray(A)
    W = np.asarray(W)
    V = np.asarray(V)
    eq_add = []
    pdiv = V.shape[1]
    qdiv = W.shape[1] + pdiv

    if weights is None:
        weight = 1
    else:
        weight = np.asarray(weights)

    phi = np.asarray(theta[0:pdiv])[:, None]
    alpha = np.asarray(theta[pdiv:qdiv])

    if X is not None:
        beta = np.asarray(theta[qdiv:])

    if model.lower() == 'linear':
        h_phi = y - np.dot(V * A[:, None], phi)
        y_transform = identity
    elif model.lower() == 'poisson':
        h_phi = y * np.exp(-1 * np.dot(V * A[:, None], phi))
        y_transform = np.exp
    else:
        raise ValueError("model='" + str(model) + "' is not supported.")

    if model_instrument.lower() == 'linear':
        z_transform = identity
    elif model_instrument.lower() == 'poisson':
        z_transform = np.exp
    elif model_instrument.lower() == 'logistic':
        z_transform = inverse_logit
    else:
        raise ValueError("model_instrument='" + str(model_instrument) + "' is not supported.")

    ee_log = ee_regression(theta=alpha, X=W, y=Z, model=model_instrument, weights=weights)
    pi = z_transform(np.dot(W, alpha))
    a_resid = (Z - pi)[:, None]

    if X is not None:
        X = np.asarray(X)
        ee_out = ee_regression(theta=beta, X=X, y=h_phi[:, 0], model=model, weights=weights)
        yhat = y_transform(np.dot(X, beta))[:, None]
        eq_add = [ee_out]
    else:
        yhat = 0

    y0_resid = h_phi - yhat
    ee_smm = weight * (a_resid * y0_resid * V).T

    return np.vstack([ee_smm, ee_log] + eq_add)


#################################################################
# Causal Inference (Sensitivity Analysis) Estimating Equations


def ee_mean_sensitivity_analysis(theta, y, delta, X, q_eval, H_function):
    r"""Estimating equation for weighted sensitivity analysis estimator of the mean."""
    delta = np.asarray(delta)[:, None]
    y = np.asarray(y)[:, None]
    X = np.asarray(X)
    qy = np.asarray(q_eval)[:, None]
    beta = np.asarray(theta[1:])[:, None]

    pred_values = np.dot(X, beta)
    numerator = delta * y
    denominator = H_function(pred_values + qy)
    ym_ind = np.where(delta == 1, numerator / denominator, 0)
    ef_mean = ym_ind - theta[0]
    ef_H = (delta / H_function(pred_values + qy) - 1) * X

    return np.vstack((ef_mean.T, ef_H.T))