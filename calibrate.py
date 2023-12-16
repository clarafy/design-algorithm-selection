import numpy as np
from statsmodels.stats.weightstats import _zstat_generic


def _calc_lhat_glm(
    grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=None, clip=False
):
    """
    Calculates the optimal value of lhat for the prediction-powered confidence interval for GLMs.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        clip (bool, optional): Whether to clip the value of lhat to be non-negative. Defaults to `False`.

    Returns:
        float: Optimal value of `lhat`. Lies in [0,1].
    """
    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]
    cov_grads = np.zeros((d, d))

    for i in range(n):
        cov_grads += (1 / n) * (
            np.outer(
                grads[i] - grads.mean(axis=0),
                grads_hat[i] - grads_hat.mean(axis=0),
            )
            + np.outer(
                grads_hat[i] - grads_hat.mean(axis=0),
                grads[i] - grads.mean(axis=0),
            )
        )
    var_grads_hat = np.cov(  # normalize by N as done above for cov_grads
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T, bias=True
    )

    if coord is None:
        vhat = inv_hessian
    else:
        vhat = inv_hessian @ np.eye(d)[coord]

    if d > 1:
        num = (
            np.trace(vhat @ cov_grads @ vhat)
            if coord is None
            else vhat @ cov_grads @ vhat
        )
        denom = (
            2 * (1 + (n / N)) * np.trace(vhat @ var_grads_hat @ vhat)
            if coord is None
            else 2 * (1 + (n / N)) * vhat @ var_grads_hat @ vhat
        )
    else:
        num = vhat * cov_grads * vhat
        denom = 2 * (1 + (n / N)) * vhat * var_grads_hat * vhat

    lhat = num / denom
    if clip:
        lhat = np.clip(lhat, 0, 1)
    return lhat.item()

def rectified_p_value(
    rectifier,
    rectifier_std,
    imputed_mean,
    imputed_std,
    null=0,
    alternative='larger',
):
    """Computes a rectified p-value.

    Args:
        rectifier (float or ndarray): Rectifier value.
        rectifier_std (float or ndarray): Rectifier standard deviation.
        imputed_mean (float or ndarray): Imputed mean.
        imputed_std (float or ndarray): Imputed standard deviation.
        null (float, optional): Value of the null hypothesis to be tested. Defaults to `0`.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.

    Returns:
        float or ndarray: P-value.
    """
    rectified_point_estimate = imputed_mean + rectifier
    rectified_std = np.maximum(
        np.sqrt(imputed_std**2 + rectifier_std**2), 1e-16
    )
    return _zstat_generic(
        rectified_point_estimate, 0, rectified_std, alternative, null
    )[1]

def ppi_mean_pval(
    Y,
    Yhat,
    Yhat_unlabeled,
    null=0,
    alternative='larger',
    lhat=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered p-value for a 1D mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        null (float): Value of the null hypothesis to be tested.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        float or ndarray: Prediction-powered p-value for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    if lhat is None:
        if len(Y.shape) > 1 and Y.shape[1] > 1:
            lhat = 1
        else:
            ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean() + (
                w * (Y - Yhat)
            ).mean()
            grads = w * (Y - ppi_pointest)
            grads_hat = w * (Yhat - ppi_pointest)
            grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
            inv_hessian = np.ones((1, 1))
            lhat = _calc_lhat_glm(
                grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=None
            )

    # print('pp point estimate, lhat, rectifier std, imputed std, label std, unweighted rect std, unweighted rect std w/ lhat:')
    # print('{:.4f} {:.4f} {:.4f} {:.4f}, {:.4f} {:.4f} {:.4f}'.format(
    #     (w_unlabeled * lhat * Yhat_unlabeled).mean() + (w * Y - lhat * w * Yhat).mean(),
    #     lhat,
    #     (w * Y - lhat * w * Yhat).std() / np.sqrt(n),
    #     (w_unlabeled * lhat * Yhat_unlabeled).std() / np.sqrt(N),
    #     Y.std() / np.sqrt(n),
    #     (Y - Yhat).std() / np.sqrt(n),
    #     (Y - lhat * Yhat).std() / np.sqrt(n)
    #     ))
    # print('lhat: {:.3f}'.format(lhat))

    return rectified_p_value(
        (w * Y - lhat * w * Yhat).mean(),
        (w * Y - lhat * w * Yhat).std() / np.sqrt(n),
        (w_unlabeled * lhat * Yhat_unlabeled).mean(),
        (w_unlabeled * lhat * Yhat_unlabeled).std() / np.sqrt(N),
        null,
        alternative,
    )