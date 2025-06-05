from multiprocessing.dummy import Pool
from itertools import repeat

import numpy as np
import scipy as sc
from statsmodels.stats.weightstats import _zstat_generic
from sklearn.isotonic import IsotonicRegression
from scipy.integrate import quad_vec
import editdistance


RNA_NUCLEOTIDES = 'UGCA'
RNANUC2COMPLEMENT = {"A": "U", "C": "G", "G": "C", "U": "A"}

# settings for GMMForecasts hyperparameter q
DEFAULT_GMM_QS = [0, 0.5, 1]


# ===== general biological sequence utilities =====

def get_mutant(seq, p_mut, alphabet):
    mutant = []
    for s in seq:
        if np.random.rand() < p_mut:
            mutant.append(np.random.choice(list(alphabet)))
        else:
            mutant.append(s)
    return "".join(mutant)

def str2onehot(sequence: str, alphabet: str) -> np.ndarray:
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out

def ohes2strs(ohe_nxlxa, alphabet: str):
    l = ohe_nxlxa.shape[1]
    idx_nl = np.where(ohe_nxlxa)[2]
    allseq = ''.join(alphabet[i] for i in idx_nl)
    assert(len(allseq) % l == 0)
    seq_n = [allseq[i : i + l] for i in range(0, len(allseq), l)]
    return seq_n


# ===== prediction-powered p-value =====

def rectified_p_value(
    rectifier,
    rectifier_std,
    imputed_mean,
    imputed_std,
    null=0,
    alternative='larger',
):
    """Computes a prediction-powered p-value.

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
        np.sqrt(imputed_std ** 2 + rectifier_std ** 2), 1e-16
    )
    return _zstat_generic(
        rectified_point_estimate, 0, rectified_std, alternative, null
    )[1]


# ===== CalibratedForecasts =====

def calibrated_cdf_vec(y, predmu_n, predsigma_n, calibrator):
    F_n = sc.stats.norm.cdf(y, loc=predmu_n, scale=predsigma_n)
    calF_n = calibrator.predict(F_n)
    return calF_n

def get_mean_from_cdf(
    predmu_n: np.array,
    predsigma_n: np.array,
    calibrator, # maps CDF value to calibrated CDF value
    positive_int_limits,
    negative_int_limits,
    quad_limit: int = 200,
    err_norm: str = 'max',
):
    if positive_int_limits is not None:
        assert(positive_int_limits[0] >= 0)
        assert(positive_int_limits[1] > positive_int_limits[0])
        term1_n, term1_max_err = quad_vec(
            lambda y: 1 - calibrated_cdf_vec(y, predmu_n, predsigma_n, calibrator),
            positive_int_limits[0], positive_int_limits[1],
            limit=quad_limit,
            norm=err_norm
        )
    else:
        term1_n = np.zeros(predmu_n.shape)

    if negative_int_limits is not None:
        assert(negative_int_limits[1] <= 0)
        assert(negative_int_limits[0] < negative_int_limits[1])
        term2_n, term2_max_err = quad_vec(
            lambda y: calibrated_cdf_vec(y, predmu_n, predsigma_n, calibrator),
            negative_int_limits[0], negative_int_limits[1],
            limit=quad_limit,
            norm=err_norm
        )
    else:
        term2_n = np.zeros(predmu_n.shape)

    return term1_n - term2_n, term1_max_err, term2_max_err
    

# ===== GMMForecasts =====

def get_exceedance_from_gmm_forecasts(
    threshold: float,
    designp_N: np.array,
    designped_N: np.array,
    designmued_N: np.array,
    designmutilde_N: np.array,
    designmuneg_N: np.array,
    designsigplus_N: np.array,
    designsigneg_N: np.array
):
    # w/ correction for covariate shift
    sfplustilde_N = sc.stats.norm.sf(threshold, loc=designmutilde_N, scale=designsigplus_N)
    sfneg_N = sc.stats.norm.sf(threshold, loc=designmuneg_N, scale=designsigneg_N)
    exceedance_tilde = np.mean(designp_N * sfplustilde_N + (1 - designp_N) * sfneg_N)

    # w/ correction to p and \tilde{\mu} for covariate shift,
    # based on edit distance to the seed sequence
    sfplused_N = sc.stats.norm.sf(threshold, loc=designmued_N, scale=designsigplus_N)
    exceedance_ed = np.mean(designped_N * sfplused_N + (1 - designped_N) * sfneg_N)
    return exceedance_tilde, exceedance_ed


def otsu_threshold(y):
    # compute histogram, where bin edges are candidate thresholds
    bincount_t, binedge_t = np.histogram(y, bins=200, range=(np.min(y), np.max(y)))
    
    # normalize bin counts to get probabilities per bin
    prob_t = bincount_t / np.sum(bincount_t)
    
    # compute class probability and mean resulting from setting each bin as the threshold
    classprob0_t = np.cumsum(prob_t)
    mean0_t = np.cumsum(prob_t * binedge_t[: -1]) / classprob0_t
    
    # global mean
    global_mean = mean0_t[-1]
    
    # compute between-class variance for each threshold
    var_t = classprob0_t * np.square(mean0_t - global_mean) / (1 - classprob0_t)
    
    # ignore invalid values (particularly divide-by-zero)
    var_t = np.nan_to_num(var_t)
    
    # get the threshold that maximizes between-class variance
    threshold_idx = np.argmax(var_t)
    threshold = binedge_t[:-1][threshold_idx]
    return threshold


def gmm_mean_forecast(ytrain_n, trainmu_n, trained_n, designmu_N, designed_N, qs = None):
    # don't need GMM sigma parameters when success criterion involves mean, e.g. Figs 3, 4
    if qs is None:
        qs = [0, 0.5, 1]
    else:
        for q in qs:
            assert((q <= 1.) and (q >= 0.))

    threshold = otsu_threshold(ytrain_n)

    # ===== probability of functionality, p =====
    
    I_n = (ytrain_n >= threshold).astype(float)
    irp = IsotonicRegression(out_of_bounds='clip')
    irp.fit(trainmu_n, I_n)


    # covariate shift correction
    trainp_n = irp.predict(trainmu_n)
    irpres = IsotonicRegression(out_of_bounds='clip')
    irpres.fit(trained_n, trainp_n - I_n)


    designp_N = irp.predict(designmu_N)
    designpres_N = irpres.predict(designed_N)
    designped_N = designp_N - designpres_N

    # ===== means of functional and non functional modes, mu^+ and mu^- =====

    plus_idx = np.where(ytrain_n >= threshold)[0]
    neg_idx = np.where(ytrain_n < threshold)[0]

    irmuplus = IsotonicRegression(out_of_bounds='clip')
    irmuplus.fit(trainmu_n[plus_idx], ytrain_n[plus_idx])
    irmuneg = IsotonicRegression(out_of_bounds='clip')
    irmuneg.fit(trainmu_n[neg_idx], ytrain_n[neg_idx])

    designmuplus_N = irmuplus.predict(designmu_N)
    designmuneg_N = irmuneg.predict(designmu_N)

    # ===== "semi-calibration" with covariate shift correction =====

    ind_Nxn = (trainmu_n[None, :] < designmu_N[:, None]).astype(float)
    designmucdf_N = np.mean(ind_Nxn, axis=1)
    ind_nxn = (trainmu_n[:, None] < trainmu_n[None, :]).astype(float)
    trainmucdf_n = np.mean(ind_nxn, axis=1)
    trainmuplus_n = irmuplus.predict(trainmu_n)

    q2functionalmus = {q: None for q in qs}
    irres = IsotonicRegression(out_of_bounds='clip')  # predict residual from edit distance to seed
    for q in qs:
        trainmutilde_n = q * trainmucdf_n * trainmu_n + (1 - q * trainmucdf_n) * trainmuplus_n
        trainres_n = trainmutilde_n - ytrain_n
        irres.fit(trained_n, trainres_n)
        
        designmutilde_N = q * designmucdf_N * designmu_N + (1 - q * designmucdf_N) * designmuplus_N
        designres_N = irres.predict(designed_N)
        designmued_N = designmutilde_N - designres_N
        q2functionalmus[q] = (designmutilde_N, designmued_N)

    return designp_N, designped_N, q2functionalmus, designmuneg_N


# ===== conformal prediction method =====

def parallelized_cumsum(arr_bxn, out_bxn: np.array = None, return_copy: bool = False):
    # output array of matching size,
    # each row of which can be populated by a thread
    if out_bxn is None:
        if not return_copy:
            raise ValueError('Need to return as out_bxn was not pre-allocated.')
        out_bxn = np.empty(arr_bxn.shape)

    # cumsum in parallel over first dimension
    with Pool() as pool:
        # avoid needing to define worker to do slicing
        pool.starmap(
            np.cumsum,
            zip(np.rollaxis(arr_bxn, 0), repeat(0), repeat(None), np.rollaxis(out_bxn, 0))
        )

    return out_bxn if return_copy else None


def get_weighted_quantiles(scores_n, w_n, w_N, alpha, batch_sz: int = 100000):
    N = w_N.shape[0]
    
    # sort calibration scores and their weights accordingly
    idx_n = np.argsort(scores_n)
    sortedscores_n = scores_n[idx_n]
    sortedw_n = w_n[idx_n]
    sortedw_bxn = np.tile(sortedw_n[None, :], [np.min([batch_sz, N]), 1])

    # extra inf for edge cases when due to floating pt error, an element of np.sum(p_Nxn1, axis=1)
    # is less than level (i.e. for alpha = 1 - 1e-8)
    scoreswithinf_n = np.hstack([sortedscores_n, [np.inf, np.inf]])

    # in batches due to OOM for 1M x 5k array for GB1 experiments
    n_batch = int(np.ceil(N / batch_sz))
    quantile_N, quantile_nobonf_N = [], []
    cdf_bxn1 = np.empty((np.min([batch_sz, N]), scores_n.size + 1))  # pre-allocate
    for batch_idx in range(n_batch):

        # p_Nxn1: N x (n + 1) matrix of normalized weights
        # each row contains the n weights of the calibration data,
        # plus the weight of one unlabeled example in the last column
        w_bx1 = w_N[batch_idx * batch_sz : (batch_idx + 1) * batch_sz, None]
        p_bxn1 = np.hstack([sortedw_bxn, w_bx1])
        p_bxn1 = p_bxn1 / np.sum(p_bxn1, axis=1, keepdims=True)  # normalize
        # assert(np.all(np.abs(np.sum(p_bxn1, axis=1) - 1.) < 1e-6))  # not parallelized
        
        # locate quantiles of weighted scores for each unlabeled example
        cdf_bxn1 = parallelized_cumsum(p_bxn1, out_bxn=cdf_bxn1, return_copy=True)
        # cdf_bxn1 = np.cumsum(p_bxn1, axis=1)  # slow af
        qidx_b = np.array([np.searchsorted(cdf_n1, 1 - alpha / N) for cdf_n1 in cdf_bxn1])
        qidx_nobonf_b = np.array([np.searchsorted(cdf_n1, 1 - alpha) for cdf_n1 in cdf_bxn1])

        # extract quantiles of scores
        quantile_b = scoreswithinf_n[qidx_b]
        quantile_N.append(quantile_b)
        quantile_nobonf_b = scoreswithinf_n[qidx_nobonf_b]
        quantile_nobonf_N.append(quantile_nobonf_b)

    quantile_N = np.hstack(quantile_N)
    quantile_nobonf_N = np.hstack(quantile_nobonf_N)

    return quantile_N, quantile_nobonf_N


def get_conformal_prediction_lower_bound(
        ycal_n,
        predcal_n,
        w_n,
        preddesign_N,
        w_N,
        alpha: float = 0.1,
        batch_sz: int = 100000
    ):
    # w_n: weights on n labeled inputs
    # w_N: weights on N unlabeled inputs
    scores_n = predcal_n - ycal_n  # signed residuals
    quantile_N, quantile_nobonf_N = get_weighted_quantiles(scores_n, w_n, w_N, alpha, batch_sz=batch_sz)
    lb = np.mean(preddesign_N - quantile_N)
    lb_nobonf = np.mean(preddesign_N - quantile_nobonf_N)
    return lb, lb_nobonf
