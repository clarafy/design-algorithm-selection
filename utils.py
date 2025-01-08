from time import time
from multiprocessing.dummy import Pool
from itertools import repeat

import numpy as np
import scipy as sc
from sklearn.isotonic import IsotonicRegression
from scipy.integrate import quad_vec
import matplotlib.pyplot as plt

import flexs.utils.sequence_utils as s_utils

import editdistance

RNA_NUCLEOTIDES = 'UGCA'
RNANUC2COMPLEMENT = {"A": "U", "C": "G", "G": "C", "U": "A"}

# ===== quantile-calibrated forecasts =====

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
    tol: float = 0.005,
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
        # assert(term1_max_err < tol)
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
        # assert(term2_max_err < tol)
    else:
        term2_n = np.zeros(predmu_n.shape)

    return term1_n - term2_n, term1_max_err, term2_max_err
    


# ===== Wheelock et al. forecasts =====

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


def wheelock_forecast(ytrain_n, trainpred_nxm, designpred_Nxm, q: float = 0):
    threshold = otsu_threshold(ytrain_n)
    
    mu_n = np.mean(trainpred_nxm, axis=1)
    var_n = np.var(trainpred_nxm, axis=1)
    res2_n = np.square(ytrain_n - mu_n)
    I_n = (ytrain_n >= threshold).astype(float)

    plus_idx = np.where(ytrain_n >= threshold)[0]
    neg_idx = np.where(ytrain_n < threshold)[0]
    
    irp = IsotonicRegression(out_of_bounds='clip')
    irp.fit(mu_n, I_n)

    irmuplus = IsotonicRegression(out_of_bounds='clip')
    irmuplus.fit(mu_n[plus_idx], ytrain_n[plus_idx])
    irmuneg = IsotonicRegression(out_of_bounds='clip')
    irmuneg.fit(mu_n[neg_idx], ytrain_n[neg_idx])

    irsigplus = IsotonicRegression(out_of_bounds='clip')
    irsigplus.fit(var_n[plus_idx], res2_n[plus_idx])
    irsigneg = IsotonicRegression(out_of_bounds='clip')
    irsigneg.fit(var_n[neg_idx], res2_n[neg_idx])
    
    designmu_N = np.mean(designpred_Nxm, axis=1)
    designvar_N = np.var(designpred_Nxm, axis=1)
    
    # forecast for designs
    designp_N = irp.predict(designmu_N)

    designmuplus_N = irmuplus.predict(designmu_N)
    designmuneg_N = irmuneg.predict(designmu_N)
    if q > 0:
        ind_Nxn = (mu_n[None, :] < designmu_N[:, None]).astype(float)
        designmucdf_N = np.mean(ind_Nxn, axis=1)
        designmuplus_N = q * designmucdf_N * designmu_N + (1 - q * designmucdf_N) * designmuplus_N

    designsigplus_N = irsigplus.predict(designvar_N)
    designsigneg_N = irsigneg.predict(designvar_N)
    
    return designp_N, designmuplus_N, designmuneg_N, designsigplus_N, designsigneg_N


def wheelock_mean_forecast(ytrain_n, trainmu_n, trained_n, designmu_N, designed_N, qs = None):
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


# ===== conformal prediction-based lower bounds =====

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

# ==========

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

def rmse(x, y, axis=None):
    return np.sqrt(np.mean(np.square(x - y), axis=axis))


def plot_xy(x, y, s: int = 50, linewidth: float = 2, alpha: float = 0.8):
    plt.scatter(x, y, s=s, alpha=alpha)
    minval = np.min([np.min(x), np.min(y)])
    maxval = np.max([np.max(x), np.max(y)])
    plt.plot([minval, maxval], [minval, maxval], '--', c='orange', linewidth=linewidth)

def process_flexs_outputs(df, alphabet, n_train: int):
    # TODO: split calibration data properly
    df1 = df[df['round'] == 1]
    df2 = df[df['round'] == 2]
    X0_nxd = np.array([s_utils.string_to_one_hot(seq, alphabet).flatten() for seq in df2.sequence[: n_train]])
    y0_n = df2.true_score.to_numpy()
    pred0_n = df2.model_score.to_numpy()
    Xm_nxd = np.array([s_utils.string_to_one_hot(seq, alphabet).flatten() for seq in df1.sequence[: n_train]])
    Xmcal_nxd = np.array([s_utils.string_to_one_hot(seq, alphabet).flatten() for seq in df1.sequence[n_train :]])
    ymcal_n = df1.true_score.to_numpy()[n_train :]
    predmcal_n = df1.model_score.to_numpy()[n_train:]
    return X0_nxd, y0_n, pred0_n, Xm_nxd, Xmcal_nxd, ymcal_n, predmcal_n

# ========== sequence distribution comparisons ==========

def pairwise_distances(seqs1, seqs2, normalize: bool = True, n_pairs: int = 1000):
    s1_n = np.random.choice(seqs1, size=n_pairs, replace=True)
    s12_n = np.random.choice(seqs1, size=n_pairs, replace=True)
    s13_n = np.random.choice(seqs1, size=n_pairs, replace=True)

    s2_n = np.random.choice(seqs2, size=n_pairs, replace=True)
    s22_n = np.random.choice(seqs2, size=n_pairs, replace=True)
    s23_n = np.random.choice(seqs2, size=n_pairs, replace=True)

    dists_between = np.array([editdistance.eval(s1, s2) for s1, s2 in zip(s1_n, s2_n)])
    dists_within1 = np.array([editdistance.eval(s12, s13) for s12, s13 in zip(s12_n, s13_n)])
    dists_within2 = np.array([editdistance.eval(s22, s23) for s22, s23 in zip(s22_n, s23_n)])
    
    if normalize:
        l = len(seqs1[0])
        print('Normalizing by sequence length {}'.format(l))
        dists_between = dists_between / l
        dists_within1 = dists_within1 / l
        dists_within2 = dists_within2 / l
    return dists_within1, dists_within2, dists_between


# ========== plotting ==========

def process_gb1_selection_experiments(
    df,
    target_values,
    temperatures,
    temp2mean,
    n_trial: int,
    imp_or_pp: str = 'imp',
    alpha: float = 0.1
):

    n_temp = temperatures.shape[0]
    alpha_bonferroni = alpha / n_temp
    print('Processing {} results with {} temperatures in [{:.3f}, {:.3f}], {} target values in [{:.2f}, {:.2f}], {} trials, and alpha = {:.1f}'.format(
        imp_or_pp, n_temp, np.min(temperatures), np.max(temperatures),
        target_values.shape[0], np.min(target_values), np.max(target_values),
        n_trial, alpha
    ))

    worst_v = []  # worst (i.e. lowest) mean label achieved by any selected temperature
    err_v = []    # error rate
    disc_v = []   # discovery rate
    val2temprange = {}  # map from target value to range of selected temperatures

    t0 = time()
    for v, val in enumerate(target_values):
        val = round(val, 4)
            
        worst_t = []  # worst (i.e. lowest) mean label for each trial
        temprange_t = []  # range of selected temperatures for each trial
        
        for i in range(n_trial):
            selected = [temp for temp in temperatures if df.loc[val]['tr{}_{}_pval_temp{:.4f}'.format(i, imp_or_pp, temp)] < alpha_bonferroni]
            achieved = [temp2mean[round(t, 4)] for t in selected]

            if len(selected):
                worst_t.append(np.min(achieved))
                temprange_t.append([np.min(selected), np.max(selected)])
            # if no discovery/selection, no worst achieved value
            else:
                temprange_t.append([])
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        disc_v.append(len(worst_t) / n_trial)
        val2temprange[val] = temprange_t

    print('Done processing ({} s)'.format(int(time() - t0)))
    return worst_v, err_v, disc_v, val2temprange
    

def process_rna_selection_experiments(
    df,
    target_values,
    design_names,
    name2truemeans,
    n_trial: int,
    imp_or_pp: str,
    alpha: float = 0.1,
    print_worst: bool = False
):
    n_config = len(design_names)
    alpha_bonferroni = alpha / n_config
    for name in design_names:
        assert(name in name2truemeans)
    print('Processing {} results with the following menu of size {}, {} target values in [{:.2f}, {:.2f}], {} trials, and alpha = {:.1f}:'.format(
        imp_or_pp, n_config, target_values.shape[0], np.min(target_values), np.max(target_values), n_trial, alpha
    ))
    for name in design_names:
        print(f'  {name}')

    format_tokens = '{:.4f} '
    format_str = '  {}: ' + ''.join(n_trial * [format_tokens])
    report_high_var = False
    for name, truemean_t in name2truemeans.items():
        vmin, vmax = np.min(truemean_t), np.max(truemean_t)
        if (vmax - vmin) / vmax > 0.05 * vmax:
            if not report_high_var:
                report_high_var = True
                print('High-ish variance estimates of true mean labels for the following. Using the average:')
            print(format_str.format(name, *truemean_t))

    worst_v = []  # worst (i.e. lowest) mean label achieved by any selected configuration
    err_v = []    # error rate
    disc_v = []   # discovery rate
    val2configs = {}  # map from target value to lists selected configurations (per trial)

    t0 = time()
    if print_worst:
        print('Worst selected configuration for:')
    for v, val in enumerate(target_values):
        val = round(val, 4)
        if print_worst:
            print('Target value {:.4f}'.format(val))
            
        worst_t = []    # worst (i.e. lowest) mean label for each trial
        configs_t = []  # selected configurations for each trial
        
        for i in range(n_trial):
            selected = [name for name in design_names if df.loc[val]['tr{}_{}_pval_{}'.format(i, imp_or_pp, name)] < alpha_bonferroni]
            achieved = [np.mean(name2truemeans[name]) for name in selected]

            if len(selected):
                worst_t.append(np.min(achieved))
                configs_t.append(selected)

                if worst_t[-1] < val and print_worst:
                    idx = np.argmin(np.array(achieved))
                    print('  Trial {} is {} with true mean label of {:.4f}'.format(i, selected[idx], worst_t[-1]))
            # if no discovery/selection, no worst achieved value
            else:
                configs_t.append([])
        if print_worst:
            print()
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        disc_v.append(len(worst_t) / n_trial)
        val2configs[val] = configs_t
        
        if (v + 1) % 20 == 0:
            print('Done processing {} / {} target values ({} s)'.format(v + 1, target_values.size, int(time() - t0)))

    print('Done processing ({} s)'.format(int(time() - t0)))
    return worst_v, err_v, disc_v, val2configs


def process_wheelock_selection_experiments(
    df,
    target_values,
    design_names,
    name2truemeans,
    n_trial: int,
    qs: np.array
):
    type2results = {
        'no-cs': {round(q, 2): None for q in qs},
        'cs': {round(q, 2): None for q in qs} 
    }
    target_values = [round(val, 4) for val in target_values]

    for cs in ['no-cs', 'cs']:
        s = 'cs_' if cs == 'cs' else ''
        for q in qs:

            val2selected = {val: [] for val in target_values}
            for i in range(n_trial):
                for val in target_values:
                    val2selected[val].append([])

                for name in design_names:
                    forecast = df.loc[i]['wf_mean_q{:.2f}_{}{}'.format(q, s, name)]

                    for val in target_values:
                        if val <= forecast:
                            val2selected[val][i].append(name)

            worst_v = []
            err_v = []
            disc_v = []
            for val in target_values:                    
                worst_t = []    # worst (i.e. lowest) mean label for each trial
                for i in range(n_trial):
                    achieved = [np.mean(name2truemeans[name]) for name in val2selected[val][i]]
                    if len(achieved):
                        worst_t.append(np.min(achieved))
                    # if no discovery/selection, no worst achieved value
                        
                worst_v.append(worst_t)
                err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
                disc_v.append(len(worst_t) / n_trial)
            
            type2results[cs][q] = (worst_v, err_v, disc_v, val2selected)
            
    return type2results


def process_gb1_cp_selection_experiments(
    df,
    target_values,
    temperatures,
    temp2mean,
    n_trial: int,
):
    target_values = [round(val, 4) for val in target_values]
    temperatures = [round(temp, 4) for temp in temperatures]
    val2selected = {val: [] for val in target_values}
    for i in range(n_trial):
        for val in target_values:
            val2selected[val].append([])

        for temp in temperatures:
            forecast = df.loc[i]['qc_forecast_mean_temp{:.4f}'.format(temp)]  # cp_nobonf_lb_

            for val in target_values:
                if val <= forecast:
                    val2selected[val][i].append(temp)

    worst_v = []
    err_v = []
    disc_v = []
    for val in target_values:                    
        worst_t = []    # worst (i.e. lowest) mean label for each trial
        for i in range(n_trial):
            achieved = [np.mean(temp2mean[temp]) for temp in val2selected[val][i]]
            if len(achieved):
                worst_t.append(np.min(achieved))
            # if no discovery/selection, no worst achieved value
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        disc_v.append(len(worst_t) / n_trial)
                        
    return worst_v, err_v, disc_v

def process_rna_qc_selection_experiments(
    df,
    target_values,
    design_names,
    name2truemeans,
    n_trial: int,
):
    target_values = [round(val, 4) for val in target_values]
    val2selected = {val: [] for val in target_values}
    for i in range(n_trial):
        for val in target_values:
            val2selected[val].append([])

        for name in design_names:
            forecast = df.loc[i][f'qc_forecast_mean_{name}']

            for val in target_values:
                if val <= forecast:
                    val2selected[val][i].append(name)

    worst_v = []
    err_v = []
    disc_v = []
    for val in target_values:                    
        worst_t = []    # worst (i.e. lowest) mean label for each trial
        for i in range(n_trial):
            achieved = [np.mean(name2truemeans[name]) for name in val2selected[val][i]]
            if len(achieved):
                worst_t.append(np.min(achieved))
            # if no discovery/selection, no worst achieved value
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        disc_v.append(len(worst_t) / n_trial)
                        
    return worst_v, err_v, disc_v


    
