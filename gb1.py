from pathlib import Path
import os.path
from itertools import product
from time import time

import numpy as np
import scipy as sc
from sklearn.isotonic import IsotonicRegression
from pandas import DataFrame, read_csv
from statsmodels.stats.weightstats import _zstat_generic
from Bio.Seq import Seq

from models import EnrichmentFeedForward, ExceedancePredictor, TorchClassifierEnsemble
import utils
from utils import str2onehot, get_conformal_prediction_lower_bound, editdistance, ohes2strs, wheelock_mean_forecast
from calibrate import rectified_p_value

# TODO: move constants to a util.py if rna.py also has
AA2CODON = {
        'l': ['tta', 'ttg', 'ctt', 'ctc', 'cta', 'ctg'],
        's': ['tct', 'tcc', 'tca', 'tcg', 'agt', 'agc'],
        'r': ['cgt', 'cgc', 'cga', 'cgg', 'aga', 'agg'],
        'v': ['gtt', 'gtc', 'gta', 'gtg'],
        'a': ['gct', 'gcc', 'gca', 'gcg'],
        'p': ['cct', 'ccc', 'cca', 'ccg'],
        't': ['act', 'acc', 'aca', 'acg'],
        'g': ['ggt', 'ggc', 'gga', 'ggg'],
        '*': ['taa', 'tag', 'tga'],
        'i': ['att', 'atc', 'ata'],
        'y': ['tat', 'tac'],
        'f': ['ttt', 'ttc'],
        'c': ['tgt', 'tgc'],
        'h': ['cat', 'cac'],
        'q': ['caa', 'cag'],
        'n': ['aat', 'aac'],
        'k': ['aaa', 'aag'],
        'd': ['gat', 'gac'],
        'e': ['gaa', 'gag'],
        'w': ['tgg'],
        'm': ['atg']
    }

AA = ''.join(AA2CODON.keys())
AA2IDX = {aa: idx for idx, aa in enumerate(AA)}

AA_NOSTOP = ''.join([aa for aa in AA2CODON.keys() if aa != '*'])
ALL_NOSTOP_AA_SEQS = [''.join(aas) for aas in product(*(4 *[AA_NOSTOP]))]
ALL_NOSTOP_AA_OHE = np.stack([str2onehot(seq, AA) for seq in ALL_NOSTOP_AA_SEQS])
WT_GB1 = 'vdgv'

NUCLEOTIDES = 'atcg'
NUC2IDX = {nuc: idx for idx, nuc in enumerate(NUCLEOTIDES)}

df = read_csv('/homefs/home/wongfanc/density-ratio-estimation/data/gb1-with-variance.csv')
seq_n = list(df['Variants'].str.lower())
y_n = df['log_fitness'].to_numpy()
var_n = df['estimated_variance'].to_numpy()
SEQ2YVAR = {seq: [y, var] for seq, y, var in zip(seq_n, y_n, var_n)}
assert(set(seq_n) == set(ALL_NOSTOP_AA_SEQS))

# ===== sampling sequences from design distribution =====
        
def normalize_theta(theta_lxa, compute_log: bool = False):
    """
    Normalizes unnormalized site-wise categorical distributions.
    """
    # log-sum-exp trick
    c_lx1 = np.max(theta_lxa, axis=1, keepdims=True)
    normalization_lx1 = c_lx1 + np.log(np.sum(np.exp(theta_lxa - c_lx1), axis=1, keepdims=True))
    logp_lxa = theta_lxa - normalization_lx1
    if compute_log:
        return logp_lxa
    return np.exp(logp_lxa)

def sample_ohe_from_nuc_distribution(p_lxa, n_seq, normalize: bool = False, reject_stop_codon: bool = True):
    """
    Given nucleotide site-wise categorical distributions, sample one-hot-encoded nucleotide and corresponding AA sequences.
    """
    nuc_seq_len, alphabet_sz = p_lxa.shape
    assert(nuc_seq_len % 3 == 0)
    assert(alphabet_sz == len(NUCLEOTIDES))

    if normalize:
        p_lxa = normalize_theta(p_lxa)

    # ----- sample nucleotides -----
    # for each sequence, sample nucleotide index at each site
    # propose 10x sequences to account for rejecting those with stop codons
    nucidx_nxl = np.array([np.random.choice(len(NUCLEOTIDES), 10 * n_seq, p=p_lxa[i]) for i in range(nuc_seq_len)]).T
    # convert to OHE nucleotides
    nucohe_nxlxa = np.eye(len(NUCLEOTIDES))[nucidx_nxl]

    # ----- convert to amino acids -----
    # convert each sequence of nucleotide indices into OHE amino acids
    aaidx_nxl = np.empty([n_seq, int(nuc_seq_len / 3)])
    aaseq_n = []
    sample_idx = -1
    proposal_idx = -1
    accepted_proposal_idx = []
    while sample_idx + 1 < n_seq:
        proposal_idx += 1
        if proposal_idx >= nucidx_nxl.shape[0]:
            raise ValueError('Not enough nucidx_l proposed ({}), increase number.'.format(nucidx_nxl.shape[0]))
        
        nucidx_l = nucidx_nxl[proposal_idx]
        # convert nucleotide indices to nucleotide bases
        nucseq = ''.join([NUCLEOTIDES[idx] for idx in nucidx_l])
        # translate nucleotides into amino acids
        aaseq = str(Seq(nucseq).translate()).lower()

        if reject_stop_codon:
            if '*' not in aaseq:
                sample_idx += 1
            else:
                continue
        else:
            sample_idx += 1
        accepted_proposal_idx.append(proposal_idx)

        aaseq_n.append(aaseq)
        # convert amino acids to amino acid indices
        aaidx_nxl[sample_idx] = [AA2IDX[aa] for aa in aaseq]

    # convert to OHE amino acids
    aaohe_nxlxa = np.eye(len(AA))[aaidx_nxl.astype(int)]
    assert(len(aaseq_n) == aaohe_nxlxa.shape[0])
    accepted_proposal_idx = np.array(accepted_proposal_idx)
    assert(accepted_proposal_idx.size == len(aaseq_n))

    return nucohe_nxlxa[accepted_proposal_idx], aaohe_nxlxa, aaseq_n

def get_aa_probs_from_nuc_probs(pnuc_lxa: np.array):
    """
    Computes amino acid site-wise categorical distribution probabilities
    given nucleotide site-wise categorical distribution probabilities.
    """
    nuc_seq_len = pnuc_lxa.shape[0]
    paa_axl = DataFrame(0., index=list(AA), columns=range(1, int(nuc_seq_len / 3) + 1))
    for aa in AA:
        codons = AA2CODON[aa]
        for cod_site in range(int(nuc_seq_len / 3)):  # for each codon site
            for cod in codons:
                p_cod = 1
                for i in range(3):  # for each nucleotide in this codon
                    nuc_idx = NUC2IDX[cod[i]]
                    p_cod *= pnuc_lxa[cod_site * 3 + i, nuc_idx]
                paa_axl[cod_site + 1].loc[aa] += p_cod
    return np.array(paa_axl).T


# ===== solving optimization problem to define design distribution =====

# NNK library (training sequence distribution):
# nucleotide categorical distributions per site in one codon
PNUC_NNK_ONECODON = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0, 0.5, 0, 0.5],
])
PNUC_NNK_LXA = np.tile(PNUC_NNK_ONECODON, [4, 1])

# amino acid categorical distribution corresponding to NNK
PAA_NNK_LXA = get_aa_probs_from_nuc_probs(PNUC_NNK_LXA)

def get_entropy(p_lxa, normalize: bool = False):
    """
    Calculates entropy from normalized probabilities of site-wise categorical distributions.
    """
    if normalize:
        p_lxa = normalize_theta(p_lxa)
    p_ma_lxa = np.ma.masked_where(p_lxa == 0, p_lxa)
    logp_lxa = np.log(p_ma_lxa)
    H = -np.sum(p_ma_lxa * logp_lxa)
    return H

# TODO: can delete, just convenient for debugging
def fit_mle_paa(aaohe_nxlxa: np.array, weight_n: np.array = None):
    if weight_n is None:
        weight_n = np.ones([aaohe_nxlxa.shape[0]])
    counts_lxa = np.sum(weight_n[:, None, None] * aaohe_nxlxa, axis=0, keepdims=False)
    paa_lxa = counts_lxa / np.sum(counts_lxa, axis=1, keepdims=True)
    return paa_lxa

def get_expected_pairwise_distance(pnuc_lxa, normalize: bool = False):  # TODO: test
    """
    Calculates the expected pairwise distance between amino acid sequences
    given the site-wise nucleotide categorical distributions.
    """
    if normalize:
        pnuc_lxa = normalize_theta(pnuc_lxa)
    paa_lxa = get_aa_probs_from_nuc_probs(pnuc_lxa)
    epd = paa_lxa.shape[0] - np.sum(np.square(paa_lxa))
    return epd

def get_nostop_normalizing_constant(logp_lxa: np.array):
    alllogp_n = get_loglikelihood(ALL_NOSTOP_AA_OHE, logp_lxa)
    allp_n = np.exp(alllogp_n)
    return np.sum(allp_n)

def get_nostop_loglikelihood(ohe_nxlxa: np.array, p_lxa: np.array):
    """
    Calculates the log-probability of OHE sequences, accounting for rejecting sequences with stop codons,
    given the probabilities of site-wise categorical distributions.
    """
    logp_lxa = np.log(p_lxa)
    normalizing_const = get_nostop_normalizing_constant(logp_lxa)
    logp_withstop_n = get_loglikelihood(ohe_nxlxa, logp_lxa)
    return logp_withstop_n - np.log(normalizing_const)

def get_loglikelihood(ohe_nxlxa: np.array, logp_lxa: np.array):
    """
    Calculates the log-probability of OHE sequences given the log-probabiliies of site-wise categorical distributions.
    """
    logp_n = np.sum(ohe_nxlxa * logp_lxa[None, :, :], axis=(1, 2), keepdims=False)
    return logp_n

def solve_max_entropy_library(
    model: EnrichmentFeedForward,
    temperature: float,
    lr: float = 0.1,
    n_sample: int = 1000,
    n_iter: int = 3000,
    print_every: int = 500,
    initialization: str = 'rand',
    save_path: str = '/homefs/home/wongfanc/density-ratio-estimation/gb1-models',
    save_fname_no_ftype: str = None,
    overwrite_old_library: bool = False
):
    if not overwrite_old_library and save_path is not None and save_fname_no_ftype is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        npz_fname = os.path.join(save_path, save_fname_no_ftype + '.npz')
        if Path(npz_fname).is_file():
            raise ValueError(f'{npz_fname} already exists, set `overwrite_old_library` to True to overwrite.')

    # initialize parameters of nucleotide site-wise categorical distributions
    nuc_seq_len = 3 * model.seq_len
    if initialization == 'rand':
        theta_lxa = np.random.randn(nuc_seq_len, len(NUCLEOTIDES))
    elif initialization == 'uniform':
        theta_lxa = np.ones([nuc_seq_len, len(NUCLEOTIDES)])
    else:
        raise ValueError('Unrecognized initialization: {}'.format(initialization))
    
    df_rows = []
    for t in range(n_iter):
        # compute normalized site-wise nucleotide categorical distribution probabilities
        logp_lxa = normalize_theta(theta_lxa, compute_log=True)
        p_lxa = np.exp(logp_lxa)

        nucohe_nxlxa, aaohe_nxlxa, _ = sample_ohe_from_nuc_distribution(
            p_lxa, n_sample, normalize=False, reject_stop_codon=False
        )
        grad_logp_nxlxa = nucohe_nxlxa - p_lxa[None, :, :]

        pred_n = model.predict(aaohe_nxlxa)
        logp_n = get_loglikelihood(nucohe_nxlxa, logp_lxa)
        w_n = pred_n - temperature * (1 + logp_n)

        grad_theta_nxlxa = w_n[:, None, None] * grad_logp_nxlxa
        theta_lxa = theta_lxa + lr * np.mean(grad_theta_nxlxa, axis=0)

        # record and print metrics
        _, aaohe_nostop_nxlxa, _ = sample_ohe_from_nuc_distribution(
            p_lxa, n_sample, normalize=False, reject_stop_codon=False
        )
        prednostop_n = model.predict(aaohe_nostop_nxlxa)
        meanpred = np.mean(pred_n)
        entropy = get_entropy(p_lxa, normalize=False)
        epd = get_expected_pairwise_distance(p_lxa, normalize=False)
        obj = meanpred + temperature * entropy
        df_rows.append([obj, meanpred, entropy, epd])
        if t == 0 or (t + 1) % print_every == 0:
            print('Iter: {}. Objective: {:.2f}. Mean prediction: {:.2f}. Mean no-stop prediction: {:.2f}. Entropy: {:.2f}. AA EPD: {:.2f}'.format(
                t + 1, obj, meanpred, np.mean(prednostop_n), entropy, epd
            ))

    df = DataFrame(data=df_rows, index=range(1, n_iter + 1), columns=['objective', 'mean_prediction', 'entropy', 'epd'])
    if save_path is not None and save_fname_no_ftype is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        npz_fname = os.path.join(save_path, save_fname_no_ftype + '.npz')
        csv_fname = os.path.join(save_path, save_fname_no_ftype + '.csv')
        np.savez(npz_fname, theta_lxa=theta_lxa)
        df.to_csv(csv_fname)
        print('Saved parameters to:           {}'.format(npz_fname))
        print('Saved optimization metrics to: {}'.format(csv_fname))
    return theta_lxa, df


# ===== temperature selection through multiple hypothesis testing =====

def get_density_ratios(aaohe_nxlxa: np.array, theta_lxa: np.array, logptrain_n: np.array = None):
    pnuc_lxa = normalize_theta(theta_lxa, compute_log=False)
    paa_lxa = get_aa_probs_from_nuc_probs(pnuc_lxa)
    logpdesign_n = get_nostop_loglikelihood(aaohe_nxlxa, paa_lxa)
    if logptrain_n is None:
        logptrain_n = get_nostop_loglikelihood(aaohe_nxlxa, PAA_NNK_LXA)
    return np.exp(logpdesign_n - logptrain_n)

# TODO: can delete, just convenient for debugging
def get_true_mean_prediction_from_theta(temp2theta, model, threshold: float = None, verbose: bool = False):
    temp2mean = {}
    if verbose:
        print('True mean prediction for temperature...')
    t0 = time()
    pred_n = model.predict(ALL_NOSTOP_AA_OHE)
    if threshold is not None:
        pred_n = (pred_n >= threshold).astype(float)
    for temp, theta_lxa in temp2theta.items():
        paa_lxa = get_aa_probs_from_nuc_probs(normalize_theta(theta_lxa))
        pdesign_n = np.exp(get_nostop_loglikelihood(ALL_NOSTOP_AA_OHE, paa_lxa))
        truemean = np.sum([p * pred for p, pred in zip(pdesign_n, pred_n)])
        temp2mean[temp] = truemean
        if verbose:
            print('    {:.4f} is {:.4f}. ({} sec)'.format(temp, truemean, int(time() - t0)))
    return temp2mean

def get_true_mean_label_from_theta(temp2theta, threshold: float = None, verbose: bool = False):
    temp2mean = {}
    if verbose:
        print('True mean for temperature...')
    t0 = time()
    ohe_nxlxa = np.stack([str2onehot(seq, AA) for seq in ALL_NOSTOP_AA_SEQS])
    for temp, theta_lxa in temp2theta.items():
        paa_lxa = get_aa_probs_from_nuc_probs(normalize_theta(theta_lxa))
        pdesign_n = np.exp(get_nostop_loglikelihood(ohe_nxlxa, paa_lxa))
        truemean = np.sum([
            p * SEQ2YVAR[seq][0] if threshold is None else p * (SEQ2YVAR[seq][0] >= threshold).astype(float)
            for seq, p in zip(ALL_NOSTOP_AA_SEQS, pdesign_n)
        ])
        temp2mean[temp] = truemean
        if verbose:
            print('    {:.4f} is {:.4f}. ({} sec)'.format(temp, truemean, int(time() - t0)))
    return temp2mean


def run_imputation_selection_experiments(
    model: EnrichmentFeedForward,
    temp2theta,
    target_values: np.array,
    n_trial: int,
    trainseq_n,
    ytrain_n: np.array,
    wheelock_forecast_qs,
    exceedance_threshold: float = None,
    n_design: int = 1000000,
    save_path: str = '/data/wongfanc/gb1-results',
    imp_csv_fname: str = None,
    n_wheelock_designs: int = 100000,
    wheelock_csv_fname: str = None,
    design_samples_fname_prefix: str = None,
    load_design_samples: bool = False,
    save_design_samples: bool = False
):
    
    if imp_csv_fname is not None:
        assert(wheelock_csv_fname is not None)
    if wheelock_csv_fname is not None:
        assert(imp_csv_fname is not None)
        
    temperatures = [round(t, 4) for t in list(temp2theta.keys())]
    target_values = np.array([round(v, 4) for v in target_values])
    imp_selected_column_names = ['tr{}_imp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    df = DataFrame(
        index=target_values, columns=imp_selected_column_names
    )

    # dataframe for Wheelock forecasting results
    wf_column_names = ['wf_mean_q{:.2f}_temp{:.4f}'.format(q, temp) for q in wheelock_forecast_qs for temp in temperatures]
    wf_cs_column_names = ['wf_mean_q{:.2f}_cs_temp{:.4f}'.format(q, temp) for q in wheelock_forecast_qs for temp in temperatures]
    wf_df = DataFrame(index=range(n_trial), columns=wf_column_names + wf_cs_column_names)

    # predictions on training data and edit distance from WT for Wheelock forecasts
    predtrain_n = model.predict(trainseq_n)
    trained_n = np.array([editdistance.eval(WT_GB1, seq) for seq in trainseq_n])

    if load_design_samples and save_design_samples:
        raise ValueError('Only one of load_design_samples or save_design_samples can be True.') 
    if load_design_samples or save_design_samples:
        if design_samples_fname_prefix is None:
            raise ValueError('Provide design_samples_fname_prefix.')
        
    if exceedance_threshold is not None:
        print('Selection quantity is probability of exceeding {}.'.format(exceedance_threshold))
        predictor = ExceedancePredictor(model, exceedance_threshold)
        # seq2y = {seq: (SEQ2YVAR[seq][0] >= exceedance_threshold).astype(float) for seq in SEQ2YVAR.keys()}
    else:
        print('Selection quantity is the mean label.')
        predictor = model
        # seq2y = SEQ2YVAR
    # print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(np.min(target_values), np.max(target_values)))

    
    t0 = time() 
    for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

        for i in range(n_trial):

            # sampling design sequences is the bottleneck for computation
            if load_design_samples:
                design_samples_fname = os.path.join(save_path, '{}-t{:.4f}-{}.npz'.format(design_samples_fname_prefix, temp, i))
                d = np.load(design_samples_fname)
                designohe_nxlxa = d['designohe_nxlxa']
                designed_n = d['designed_n']
                if designohe_nxlxa.shape[0] != n_design:
                    raise ValueError('Loaded {} != n_design = {} design sequences from {}.'.format(
                        designohe_nxlxa.shape[0], n_design, design_samples_fname
                    ))
                print('Loaded {} design sequences from {}.'.format(n_design, design_samples_fname))
            else:
                # sample unlabeled sequences from design distribution
                _, designohe_nxlxa, _ = sample_ohe_from_nuc_distribution(
                    theta_lxa, n_design, normalize=True, reject_stop_codon=True
                )
                designseq_N = ohes2strs(designohe_nxlxa, AA)  # TODO: memory probably can't handle all of this
                designed_n = np.array([editdistance.eval(WT_GB1, seq) for seq in designseq_N])
                print('Sampled {} design sequences for temperature {:.2f} ({} s).'.format(n_design, temp, int(time() - t0)))
                if save_design_samples:
                    design_samples_fname = os.path.join(save_path, '{}-t{:.4f}-{}.npz'.format(design_samples_fname_prefix, temp, i))
                    np.savez(design_samples_fname, designohe_nxlxa=designohe_nxlxa)
                    print('Saved design samples to {}.'.format(design_samples_fname))
                
            # predictions for unlabeled design sequences
            preddesign_n = predictor.predict(designohe_nxlxa)
            imputed_mean = np.mean(preddesign_n)
            imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)

            for target_val in target_values:
                imp_pval = _zstat_generic(
                    imputed_mean,
                    0,
                    imputed_se,
                    alternative='larger',
                    diff=target_val
                )[1]
                df.loc[target_val]['tr{}_imp_pval_temp{:.4f}'.format(i, temp)] = imp_pval
            
            # ===== Wheelock forecast ===== 
            # subsample to avoid OOM
            samp_idx = np.random.choice(preddesign_n.size, size=(n_wheelock_designs), replace=False)
            designp_N, designped_N, q2functionalmus, designmuneg_N = wheelock_mean_forecast(
                ytrain_n, predtrain_n, trained_n, preddesign_n[samp_idx], designed_n[samp_idx], qs=wheelock_forecast_qs
            )

            # record forecast
            for q, (designmutilde_N, designmued_N) in q2functionalmus.items():
                # w/o correction for covariate shift
                forecast_tilde = np.mean(designp_N * designmutilde_N + (1 - designp_N) * designmuneg_N)
                wf_df.loc[i]['wf_mean_q{:.2f}_temp{:.4f}'.format(q, temp)] = forecast_tilde

                # w/ correction to p and \tilde{\mu} for covariate shift,
                # based on edit distance to the seed sequence
                forecast_ed = np.mean(designped_N * designmued_N + (1 - designped_N) * designmuneg_N)
                wf_df.loc[i]['wf_mean_q{:.2f}_cs_temp{:.4f}'.format(q, temp)] = forecast_ed
                print('Temp {:.4f}, trial {}, q = {}. Wheelock forecast {:.3f}, w/ covariate shift {:.3f}'.format(
                    temp, i, q, forecast_tilde, forecast_ed
                ))
        
        print('Done with temperature {:.4f} ({} / {}) ({} s)'.format(
            temp, t + 1, len(temperatures), int(time() - t0))
        )
        if imp_csv_fname is not None:
            df.to_csv(imp_csv_fname)
            wf_df.to_csv(wheelock_csv_fname, index_label='trial')
            print('Saved to {} and {} ({} s).'.format(imp_csv_fname, wheelock_csv_fname, int(time() - t0)))
    
    return df

    
def select_for_mean_with_calibration_data(
    model: EnrichmentFeedForward,
    temp2theta,
    target_values: np.array,
    n_cal: int = 5000,
    n_design: int = 1000000,
    n_trial: int = 200,
    n_forecast_designs: int = 1000,
    tol: float = 0.01,
    quad_limit: int = 500,
    self_normalize_weights: bool = True,
    # cp_batch_sz: int = 10000,
    save_path: str = '/data/wongfanc/gb1-results',
    pp_csv_fname: str = None,
    forecast_csv_fname: str = None,
    design_samples_fname_prefix: str = None,
    load_design_samples: bool = False,
    save_design_samples: bool = False
):
    
    temperatures = [round(t, 4) for t in temp2theta.keys()]
    target_values = np.array([round(v, 4) for v in target_values])
    
    # initialize dataframes for recording results
    pp_selected_column_names = ['tr{}_pp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    # cp_column_names = ['cp_lb_temp{:.4f}'.format(temp) for temp in temperatures]
    # cp_nobonf_column_names = ['cp_nobonf_lb_temp{:.4f}'.format(temp) for temp in temperatures]
    # qc_column_names = ['qc_forecast_mean_temp{:.4f}'.format(temp) for temp in temperatures]
    df = DataFrame(index=target_values, columns=pp_selected_column_names)  # for our method
    # forecast_df = DataFrame(index=range(n_trial), columns=qc_column_names) # for calibrated forecast baseline

    # if pp_csv_fname is not None:
    #     assert(forecast_csv_fname is not None)

    if load_design_samples and save_design_samples:
        raise ValueError('Only one of load_design_samples or save_design_samples can be True.') 
    if load_design_samples or save_design_samples:
        if design_samples_fname_prefix is None:
            raise ValueError('Provide design_samples_fname_prefix.')
        
    print('Selection quantity is the mean label.')
    print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(np.min(target_values), np.max(target_values)))

    t0 = time()
    for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

        # ----- load or sample designs -----
        if load_design_samples:
            design_samples_fname = os.path.join(save_path, '{}-t{:.4f}.npz'.format(design_samples_fname_prefix, temp))
            d = np.load(design_samples_fname)
            designohe_nxlxa = d['designohe_nxlxa']
            if designohe_nxlxa.shape[0] != n_design:
                raise ValueError('Loaded {} != n_design = {} design sequences from {}.'.format(
                    designohe_nxlxa.shape[0], n_design, design_samples_fname
                ))
            print('Loaded {} design sequences from {}.'.format(n_design, design_samples_fname))
        else:
            # sample unlabeled sequences from design distribution
            _, designohe_nxlxa, _ = sample_ohe_from_nuc_distribution(
                theta_lxa, n_design, normalize=True, reject_stop_codon=True
            )
            print('Sampled {} design sequences for temperature {:.2f} ({} s).'.format(n_design, temp, int(time() - t0)))
            if save_design_samples:
                design_samples_fname = os.path.join(save_path, '{}-t{:.4f}.npz'.format(design_samples_fname_prefix, temp))
                np.savez(design_samples_fname, designohe_nxlxa=designohe_nxlxa)
                print('Saved design samples to {}.'.format(design_samples_fname))
            

        # ----- predictions for designs ----
        preddesign_Nxm = model.ensemble_predict(designohe_nxlxa)
        preddesign_N = np.mean(preddesign_Nxm, axis=1)
        imputed_mean = np.mean(preddesign_N)
        imputed_se = np.std(preddesign_N) / np.sqrt(preddesign_N.size)

        # subsample some designs for calibrated forecast baseline
        # forecast_idx = np.random.choice(preddesign_N.size, size=n_forecast_designs, replace=False)
        # designmu_n = preddesign_N[forecast_idx]
        # designsigma_n = np.std(preddesign_Nxm[forecast_idx, :], axis=1, keepdims=False)
        # # heuristic for limits of integration for computing E[Y] given CDF of Y
        # pos_int_limit = 2 * np.max(designmu_n + 3 * designsigma_n)
        # neg_int_limit = 2 * np.min(designmu_n - 3 * designsigma_n)


        # ----- get density ratio weights for N designs (for CP baseline) -----
        # designlogptrain_N = get_nostop_loglikelihood(designohe_nxlxa, PAA_NNK_LXA)
        # designdr_N = get_density_ratios(designohe_nxlxa, theta_lxa, logptrain_n=designlogptrain_N)
        # if self_normalize_weights:
        #     designdr_N = designdr_N / np.sum(designdr_N) * designdr_N.size
        # print('Done getting density ratios for design sequences from temperature {:.4f}. ({} s)'.format(
        #     temp, int(time() - t0)
        # ))


        # ----- trials of draws of calibration data -----
        for i in range(n_trial):

            # sample calibration data from NNK library
            _, calohe_nxlxa, calseq_n = sample_ohe_from_nuc_distribution(
                PNUC_NNK_LXA, n_cal, normalize=False, reject_stop_codon=True
            )
            ycal_n = np.array([SEQ2YVAR[seq][0] for seq in calseq_n])

            # predictions for calibration sequences
            predcal_nxm = model.ensemble_predict(calohe_nxlxa)
            predcal_n = np.mean(predcal_nxm, axis=1, keepdims=False)
            callogptrain_n = get_nostop_loglikelihood(calohe_nxlxa, PAA_NNK_LXA)

            # density ratios for calibration sequences
            caldr_n = get_density_ratios(calohe_nxlxa, theta_lxa, logptrain_n=callogptrain_n)
            # cal_ess = np.square(np.sum(caldr_n)) / np.sum(np.square(caldr_n))
            if self_normalize_weights:
                caldr_n = caldr_n / np.sum(caldr_n) * caldr_n.size
        
            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal_n)
            rectifier_mean = np.mean(rect_n)
            rectifier_se = np.std(rect_n) / np.sqrt(rect_n.size)

            # get prediction-powered p-values for each desired threshold value
            for target_val in target_values:
                pp_pval = rectified_p_value(
                    rectifier_mean,
                    rectifier_se,
                    imputed_mean,
                    imputed_se,
                    null=target_val,
                    alternative='larger'
                )
                df.loc[target_val]['tr{}_pp_pval_temp{:.4f}'.format(i, temp)] = pp_pval

            # ----- calibrated forecast baseline -----
            # calsigma_n = np.std(predcal_nxm, axis=1, keepdims=False)
            # calF_n = sc.stats.norm.cdf(ycal_n, loc=predcal_n, scale=calsigma_n)
            # calempF_n = np.mean(calF_n[:, None] <= calF_n[None, :], axis=0, keepdims=False)
            # ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            # ir.fit(calF_n, calempF_n)

            # # subsample designs
            # qcmu_N, t1_err, t2_err = utils.get_mean_from_cdf(
            #     designmu_n,
            #     designsigma_n,
            #     ir,
            #     (0, pos_int_limit),
            #     (neg_int_limit, 0),
            #     quad_limit=quad_limit,
            #     err_norm='max',
            #     tol=tol,
            # )
            # # if t1_err > tol or t2_err > tol:
            # #     print('temp {:.4f}, trial {}, t1_err {:.4f}, t2_err {:.4f} ({} s)'.format(
            # #         temp, i, t1_err, t2_err, int(time() - t0)
            # #     ))
            # forecast_df.loc[i, 'qc_forecast_mean_temp{:.4f}'.format(temp)] = np.mean(qcmu_N)

            # ----- conformal prediction-based baseline -----
            # CP-based lower bound
            # lb, lb_nobonf = get_conformal_prediction_lower_bound(
            #     ycal_n, predcal_n, caldr_n, preddesign_N, designdr_N,
            #     alpha=alpha / len(temperatures), batch_sz=cp_batch_sz
            # )
            # # conformal prediction-based test outcomes (reject/fail to reject)
            # cp_df.loc[i, 'cp_lb_temp{:.4f}'.format(temp)] = lb
            # cp_df.loc[i, 'cp_nobonf_lb_temp{:.4f}'.format(temp)] = lb_nobonf
            # if lb_nobonf > -np.inf:
            #     print('Temp {:.4f}, trial {} has CP-based LBs {:.4f} (Bonferroni), {:.4f} (uncorrected) ({} s)'.format(
            #         temp, i, lb, lb_nobonf, int(time() - t0)
            #     ))

                    
        print('Done with {} trials for temperature {:.4f} ({} / {}) ({} s).'.format(
            n_trial, temp, t + 1, len(temperatures), int(time() - t0)
        ))
        if pp_csv_fname is not None:
            df.to_csv(pp_csv_fname) # time sink
            # forecast_df.to_csv(forecast_csv_fname)
            print('Saved PP results to {}.'.format(pp_csv_fname))
            # print('Saved calibrated forecast results to {} ({} s).\n'.format(forecast_csv_fname, int(time() - t0)))

    return df, forecast_csv_fname


def select_for_exceedance_no_calibration_data(
    model: EnrichmentFeedForward,
    temp2theta,
    target_values: np.array,
    n_trial: int,
    trainseq_n,
    ytrain_n: np.array,
    gmm_forecast_qs,
    exceedance_threshold: float = None,
    n_design: int = 1000000,
    save_path: str = '/data/wongfanc/gb1-results',
    imp_csv_fname: str = None,
    n_gmm_designs: int = 100000,
    gmm_csv_fname: str = None,
    design_samples_fname_prefix: str = None,
    load_design_samples: bool = False,
    save_design_samples: bool = False
):
    
    if imp_csv_fname is not None:
        assert(gmm_csv_fname is not None)
    if gmm_csv_fname is not None:
        assert(imp_csv_fname is not None)
        
    temperatures = [round(t, 4) for t in list(temp2theta.keys())]
    target_values = np.array([round(v, 4) for v in target_values])
    imp_selected_column_names = ['tr{}_imp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    df = DataFrame(index=target_values, columns=imp_selected_column_names)

    # dataframe for GMM forecast results
    gmm_column_names = ['wf_mean_q{:.2f}_temp{:.4f}'.format(q, temp) for q in gmm_forecast_qs for temp in temperatures]
    gmm_cs_column_names = ['wf_mean_q{:.2f}_cs_temp{:.4f}'.format(q, temp) for q in gmm_forecast_qs for temp in temperatures]
    gmm_df = DataFrame(index=range(n_trial), columns=gmm_column_names + gmm_cs_column_names)

    # (real-valued) predictions and edit distance from WT for training sequences, for GMM forecasts
    realpredtrain_nxm = model.ensemble_predict(trainseq_n)
    trained_n = np.array([editdistance.eval(WT_GB1, seq) for seq in trainseq_n])

    if load_design_samples and save_design_samples:
        raise ValueError('Only one of load_design_samples or save_design_samples can be True.') 
    if load_design_samples or save_design_samples:
        if design_samples_fname_prefix is None:
            raise ValueError('Provide design_samples_fname_prefix.')
        
    print('Selection quantity is probability of exceeding {}.'.format(exceedance_threshold))
    print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(
        np.min(target_values), np.max(target_values)
    ))

    predictor = ExceedancePredictor(model, exceedance_threshold)
    t0 = time() 
    for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

        for i in range(n_trial):

            # sampling design sequences is the bottleneck for computation
            if load_design_samples:

                design_samples_fname = os.path.join(save_path, '{}-t{:.4f}-{}.npz'.format(design_samples_fname_prefix, temp, i))
                d = np.load(design_samples_fname)

                designohe_nxlxa = d['designohe_nxlxa']
                designed_n = d['designed_n']

                if designohe_nxlxa.shape[0] != n_design:
                    raise ValueError('Loaded {} != n_design = {} design sequences from {}.'.format(
                        designohe_nxlxa.shape[0], n_design, design_samples_fname
                    ))
                print('Loaded {} design sequences from {}.'.format(n_design, design_samples_fname))

            else:
                # sample unlabeled sequences from design distribution
                _, designohe_nxlxa, _ = sample_ohe_from_nuc_distribution(
                    theta_lxa, n_design, normalize=True, reject_stop_codon=True
                )
                designseq_N = ohes2strs(designohe_nxlxa, AA)  # TODO: memory probably can't handle all of this
                designed_n = np.array([editdistance.eval(WT_GB1, seq) for seq in designseq_N])
                print('Sampled {} design sequences for temperature {:.2f} ({} s).'.format(n_design, temp, int(time() - t0)))
                if save_design_samples:
                    design_samples_fname = os.path.join(save_path, '{}-t{:.4f}-{}.npz'.format(design_samples_fname_prefix, temp, i))
                    np.savez(design_samples_fname, designohe_nxlxa=designohe_nxlxa)
                    print('Saved design samples to {}.'.format(design_samples_fname))
                
            # predictions for unlabeled design sequences 
            realpreddesign_nxm = model.ensemble_predict(designohe_nxlxa)  # real-valued, for GMM
            preddesign_n = predictor.predict(designohe_nxlxa)         # binary-valued, for PO
            imputed_mean = np.mean(preddesign_n)
            imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)

            for target_val in target_values:
                imp_pval = _zstat_generic(
                    imputed_mean,
                    0,
                    imputed_se,
                    alternative='larger',
                    diff=target_val
                )[1]
                df.loc[target_val]['tr{}_imp_pval_temp{:.4f}'.format(i, temp)] = imp_pval
            
            # ===== GMM forecast ===== 
            # subsample to avoid OOM
            samp_idx = np.random.choice(preddesign_n.size, size=(n_gmm_designs), replace=False)
            
            designp_N, designped_N, q2functionalmus, designmuneg_N, designsig2plus_N, designsig2neg_N = utils.wheelock_forecast(
                ytrain_n, realpredtrain_nxm, trained_n, realpreddesign_nxm[samp_idx], designed_n[samp_idx], qs=gmm_forecast_qs
            )
            designsigplus_N = np.sqrt(designsig2plus_N)
            designsigneg_N = np.sqrt(designsig2neg_N)

            # get exceedance estimates based on GMM forecasts, for different values of q ("semi-calibration")
            for q, (designmutilde_N, designmued_N) in q2functionalmus.items():

                exceedance_tilde, exceedance_ed =  utils.get_exceedance_from_gmm_forecasts(
                    exceedance_threshold,
                    designp_N,
                    designped_N,
                    designmued_N,
                    designmutilde_N,
                    designmuneg_N,
                    designsigplus_N,
                    designsigneg_N
                )
                gmm_df.loc[i]['wf_mean_q{:.2f}_temp{:.4f}'.format(q, temp)] = exceedance_tilde
                gmm_df.loc[i]['wf_mean_q{:.2f}_cs_temp{:.4f}'.format(q, temp)] = exceedance_ed
                print('Temp {:.4f}, trial {}, q = {}. GMM forecast {:.3f}, w/ covariate shift {:.3f}'.format(
                    temp, i, q, exceedance_tilde, exceedance_ed
                ))
        
        print('Done with temperature {:.4f} ({} / {}) ({} s)'.format(
            temp, t + 1, len(temperatures), int(time() - t0))
        )
        if imp_csv_fname is not None:
            df.to_csv(imp_csv_fname)
            gmm_df.to_csv(gmm_csv_fname, index_label='trial')
            print('Saved to {} and {} ({} s).'.format(imp_csv_fname, gmm_csv_fname, int(time() - t0)))
    
    return df


def select_for_exceedance_with_calibration_data(
    model: EnrichmentFeedForward,
    temp2theta,
    target_values: np.array,
    exceedance_threshold: float = None,
    n_cal: int = 5000,
    n_design: int = 1000000,
    n_trial: int = 200,
    n_design_subsample: int = 1000,
    weight_isonotic_regression: bool = False,
    n_train_lr: int = 1000,
    self_normalize_weights: bool = True,
    save_path: str = '/data/wongfanc/gb1-results',
    pp_csv_fname: str = None,
    forecast_csv_fname: str = None,
    design_samples_fname_prefix: str = None,
    load_design_samples: bool = False,
    save_design_samples: bool = False
):
    
    temperatures = [round(t, 4) for t in temp2theta.keys()]
    target_values = np.array([round(v, 4) for v in target_values])

    pp_selected_column_names = ['tr{}_pp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    df = DataFrame(index=target_values, columns=pp_selected_column_names)
    forecast_column_names = ['cp_lb_temp{:.4f}'.format(temp) for temp in temperatures] \
        + ['cp_nobonf_lb_temp{:.4f}'.format(temp) for temp in temperatures] \
        + ['qc_forecast_mean_temp{:.4f}'.format(temp) for temp in temperatures]
    forecast_df = DataFrame(index=range(n_trial), columns=forecast_column_names)

    if pp_csv_fname is not None:
        assert(forecast_csv_fname is not None)

    if load_design_samples and save_design_samples:
        raise ValueError('Only one of load_design_samples or save_design_samples can be True.') 
    if load_design_samples or save_design_samples:
        if design_samples_fname_prefix is None:
            raise ValueError('Provide design_samples_fname_prefix.')
        
    print('Selection quantity is probability of exceeding {}.'.format(exceedance_threshold))
    print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(np.min(target_values), np.max(target_values)))


    t0 = time()
    for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

        # ----- load or sample designs -----
        if load_design_samples:
            design_samples_fname = os.path.join(save_path, '{}-t{:.4f}.npz'.format(design_samples_fname_prefix, temp))
            d = np.load(design_samples_fname)
            designohe_nxlxa = d['designohe_nxlxa']
            if designohe_nxlxa.shape[0] != n_design:
                raise ValueError('Loaded {} != n_design = {} design sequences from {}.'.format(
                    designohe_nxlxa.shape[0], n_design, design_samples_fname
                ))
            print('Loaded {} design sequences from {}.'.format(n_design, design_samples_fname))
        else:
            _, designohe_nxlxa, _ = sample_ohe_from_nuc_distribution(
                theta_lxa, n_design, normalize=True, reject_stop_codon=True
            )
            print('Sampled {} design sequences for temperature {:.2f} ({} s).'.format(n_design, temp, int(time() - t0)))
            if save_design_samples:
                design_samples_fname = os.path.join(save_path, '{}-t{:.4f}.npz'.format(design_samples_fname_prefix, temp))
                np.savez(design_samples_fname, designohe_nxlxa=designohe_nxlxa)
                print('Saved design samples to {}.'.format(design_samples_fname))


        # ----- get density ratio weights for N designs (for CP baseline) -----
        # designlogptrain_N = get_nostop_loglikelihood(designohe_nxlxa, PAA_NNK_LXA)
        # designdr_N = get_density_ratios(designohe_nxlxa, theta_lxa, logptrain_n=designlogptrain_N)
        # if self_normalize_weights:
        #     designdr_N = designdr_N / np.sum(designdr_N) * designdr_N.size
        # print('Done getting density ratios for design sequences from temperature {:.4f}. ({} s)'.format(
        #     temp, int(time() - t0)
        # ))


        # ----- trials of draws of calibration data -----
        for i in range(n_trial):

            # sample calibration data from NNK
            _, calohe_nxlxa, calseq_n = sample_ohe_from_nuc_distribution(
                PNUC_NNK_LXA, n_cal, normalize=False, reject_stop_codon=True
            )
            # get real-valued labels for calibration sequences
            realycal_n = np.array([SEQ2YVAR[seq][0] for seq in calseq_n])


            # ===== marginally calibrated forecasts (do this first before splitting cal data for PP)=====

            # real-valued predictions for calibration sequences
            realpredcal_nxm = model.ensemble_predict(calohe_nxlxa)
            realpredcal_n = np.mean(realpredcal_nxm, axis=1, keepdims=False)
            
            # fit isotonic regression to calibrate forecasts
            calsigma_n = np.std(realpredcal_nxm, axis=1, keepdims=False)
            calF_n = sc.stats.norm.cdf(realycal_n, loc=realpredcal_n, scale=calsigma_n)
            calempF_n = np.mean(calF_n[:, None] <= calF_n[None, :], axis=0, keepdims=False)
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            ir.fit(calF_n, calempF_n, sample_weight=caldr_n if weight_isonotic_regression else None)

            # subsample designs for CP and marginally calibrated forecasts, for speed
            sample_idx = np.random.choice(n_design, size=n_design_subsample, replace=False)

            # real-valued predictions for design sequences
            realpreddesign_nxm = model.ensemble_predict(designohe_nxlxa[sample_idx])
            designmu_n = np.mean(realpreddesign_nxm, axis=1, keepdims=False)
            designsigma_n = np.std(realpreddesign_nxm, axis=1, keepdims=False)

            # get exceedance from calibrated forecasts
            F_n = sc.stats.norm.cdf(exceedance_threshold, loc=designmu_n, scale=designsigma_n)
            calibratedF_n = ir.predict(F_n)
            calibrated_exceedance = np.mean(1 - calibratedF_n)  # SF = 1 - CDF and tower property
            forecast_df.loc[i, 'qc_forecast_mean_temp{:.4f}'.format(temp)] = calibrated_exceedance


            # ===== CP =====
            # CP-based lower bound
            # lb, lb_nobonf = get_conformal_prediction_lower_bound(
            #     ycal_n, predcal_n, caldr_n, preddesign_N, designdr_N,
            #     alpha=alpha / len(temperatures), batch_sz=cp_batch_sz
            # )
            # # conformal prediction-based test outcomes (reject/fail to reject)
            # cp_df.loc[i, 'cp_lb_temp{:.4f}'.format(temp)] = lb
            # cp_df.loc[i, 'cp_nobonf_lb_temp{:.4f}'.format(temp)] = lb_nobonf
            # if lb_nobonf > -np.inf:
            #     print('Temp {:.4f}, trial {} has CP-based LBs {:.4f} (Bonferroni), {:.4f} (uncorrected) ({} s)'.format(
            #         temp, i, lb, lb_nobonf, int(time() - t0)
            #     ))


            # ===== PP =====

            # binarize labels
            ycal_n = (realycal_n >= exceedance_threshold).astype(float)

            # ----- train [0, 1] exceedance predictor for this trial -----
            predictor = ExceedancePredictor(model, exceedance_threshold)
            sampled_both_labels = False
            # ensure enough positive labels in both training and calibration data
            while not sampled_both_labels:  
                shuffle_idx = np.random.permutation(n_cal)
                train_idx, cal_idx = shuffle_idx[: n_train_lr], shuffle_idx[n_train_lr :]
                sampled_both_labels = np.sum(ycal_n[train_idx]) > 2 and np.sum(ycal_n[cal_idx]) > 2
            trainohe_nxlxa, ytrain_n = calohe_nxlxa[train_idx], ycal_n[train_idx]
            predictor.fit(trainohe_nxlxa, ytrain_n)

            # use the remaining data for calibration
            calohe_nxlxa = calohe_nxlxa[cal_idx]
            ycal_n = ycal_n[cal_idx]

            # ----- [0, 1] predictions -----
            # for design sequences
            preddesign_n = predictor.predict(designohe_nxlxa)
            imputed_mean = np.mean(preddesign_n)
            imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)

            # for calibration sequences
            predcal_n = predictor.predict(calohe_nxlxa)

            # ----- compute rectifier -----
            # density ratios on calibration sequences
            callogptrain_n = get_nostop_loglikelihood(calohe_nxlxa, PAA_NNK_LXA)
            caldr_n = get_density_ratios(calohe_nxlxa, theta_lxa, logptrain_n=callogptrain_n)
            # cal_ess = np.square(np.sum(caldr_n)) / np.sum(np.square(caldr_n))
            if self_normalize_weights:
                caldr_n = caldr_n / np.sum(caldr_n) * caldr_n.size
        
            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal_n)
            rectifier_mean = np.mean(rect_n)
            rectifier_se = np.std(rect_n) / np.sqrt(rect_n.size)

            # PP p-values
            for target_val in target_values:
                pp_pval = rectified_p_value(
                    rectifier_mean,
                    rectifier_se,
                    imputed_mean,
                    imputed_se,
                    null=target_val,
                    alternative='larger'
                )
                df.loc[target_val]['tr{}_pp_pval_temp{:.4f}'.format(i, temp)] = pp_pval

                    
        print('Done with {} trials for temperature {:.4f} ({} / {}) ({} s).'.format(
            n_trial, temp, t + 1, len(temperatures), int(time() - t0)
        ))
        if pp_csv_fname is not None:
            df.to_csv(pp_csv_fname) # time sink
            forecast_df.to_csv(forecast_csv_fname)
            print('Saved PP results to {}.'.format(pp_csv_fname))
            print('Saved CP and marginally calibrated forecasts results to {} ({} s).\n'.format(forecast_csv_fname, int(time() - t0)))

    return df, forecast_df
