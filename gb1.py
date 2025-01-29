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

from models import EnrichmentFeedForward
import utils
from utils import str2onehot, editdistance, ohes2strs, gmm_mean_forecast, rectified_p_value

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

DEFAULT_GMM_QS = [0, 0.5, 1]

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


# ===== temperature selection experiments =====

def get_density_ratios(aaohe_nxlxa: np.array, theta_lxa: np.array, logptrain_n: np.array = None):
    pnuc_lxa = normalize_theta(theta_lxa, compute_log=False)
    paa_lxa = get_aa_probs_from_nuc_probs(pnuc_lxa)
    logpdesign_n = get_nostop_loglikelihood(aaohe_nxlxa, paa_lxa)
    if logptrain_n is None:
        logptrain_n = get_nostop_loglikelihood(aaohe_nxlxa, PAA_NNK_LXA)
    return np.exp(logpdesign_n - logptrain_n)


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


def select_for_mean_without_labeled_data(
    model: EnrichmentFeedForward,
    temp2theta,
    desired_values: np.array,
    n_trial: int,
    trainseq_n,
    ytrain_n: np.array,
    n_design: int = 1000000,
    save_path: str = '/data/wongfanc/gb1-results',
    po_csv_fname: str = None,
    gmm_csv_fname: str = None,
    n_gmm_designs: int = 10000,
    gmm_qs = None,
    design_samples_fname_prefix: str = 'gb1-h10-10k-051324-samples',
    load_design_samples: bool = True,
    save_design_samples: bool = False
):
    
    if po_csv_fname is not None:
        assert(gmm_csv_fname is not None)
    if gmm_csv_fname is not None:
        assert(po_csv_fname is not None)

    if gmm_qs is None:
        gmm_qs = DEFAULT_GMM_QS
        
    temperatures = [round(t, 4) for t in list(temp2theta.keys())]
    desired_values = np.array([round(v, 4) for v in desired_values])
    po_column_names = ['tr{}_po_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    po_df = DataFrame(index=desired_values, columns=po_column_names)

    # dataframe for Wheelock forecasting results
    gmm_column_names = ['gmm-q{:.2f}_mean_temp{:.4f}'.format(q, temp) for q in gmm_qs for temp in temperatures]
    gmm_cs_column_names = ['gmm-cs-q{:.2f}_mean_temp{:.4f}'.format(q, temp) for q in gmm_qs for temp in temperatures]
    gmm_df = DataFrame(index=range(n_trial), columns=gmm_column_names + gmm_cs_column_names)

    # predictions on training data and edit distance from WT for Wheelock forecasts
    predtrain_n = model.predict(trainseq_n)
    trained_n = np.array([editdistance.eval(WT_GB1, seq) for seq in trainseq_n])

    if load_design_samples and save_design_samples:
        raise ValueError('Only one of load_design_samples or save_design_samples can be True.') 
    if load_design_samples or save_design_samples:
        if design_samples_fname_prefix is None:
            raise ValueError('Provide design_samples_fname_prefix.')
        
    print('Selection quantity is the mean label.')
    print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(np.min(desired_values), np.max(desired_values)))

    
    t0 = time() 
    for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

        for i in range(n_trial):

            # sampling design sequences is bottleneck, load pre-sampled
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
            preddesign_n = model.predict(designohe_nxlxa)
            po_mean = np.mean(preddesign_n)
            po_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)

            # ===== PO =====
            for target_val in desired_values:
                po_pval = _zstat_generic(
                    po_mean,
                    0,
                    po_se,
                    alternative='larger',
                    diff=target_val
                )[1]
                po_df.loc[target_val]['tr{}_po_pval_temp{:.4f}'.format(i, temp)] = po_pval
            
            # ===== GMMForecasts ===== 
            # subsample designs to avoid OOM
            samp_idx = np.random.choice(preddesign_n.size, size=(n_gmm_designs), replace=False)
            designp_N, designped_N, q2functionalmus, designmuneg_N = gmm_mean_forecast(
                ytrain_n, predtrain_n, trained_n, preddesign_n[samp_idx], designed_n[samp_idx], qs=gmm_qs
            )

            # vary GMMForecasts hyperparameter q
            for q, (designmutilde_N, designmued_N) in q2functionalmus.items():
                # w/o correction for covariate shift
                forecast_tilde = np.mean(designp_N * designmutilde_N + (1 - designp_N) * designmuneg_N)
                gmm_df.loc[i]['gmm-q{:.2f}_mean_temp{:.4f}'.format(q, temp)] = forecast_tilde

                # w/ correction to p and \tilde{\mu} for covariate shift,
                # based on edit distance to the seed sequence
                forecast_ed = np.mean(designped_N * designmued_N + (1 - designped_N) * designmuneg_N)
                gmm_df.loc[i]['gmm-cs-q{:.2f}_mean_temp{:.4f}'.format(q, temp)] = forecast_ed
    
        print('Done with temperature {:.4f} ({} / {}) ({} s)'.format(
            temp, t + 1, len(temperatures), int(time() - t0))
        )
        if po_csv_fname is not None:
            po_df.to_csv(po_csv_fname)
            gmm_df.to_csv(gmm_csv_fname, index_label='trial')
            print('Saved to {} and {} ({} s).'.format(po_csv_fname, gmm_csv_fname, int(time() - t0)))
    
    return po_df, gmm_df

    
def select_for_mean_with_labeled_data(
    model: EnrichmentFeedForward,
    temp2theta,
    desired_values: np.array,
    n_labeled: int = 5000,
    n_design: int = 1000000,
    n_trial: int = 200,
    n_design_forecasts: int = 1000,
    tol: float = 0.01,
    quad_limit: int = 1000,
    save_path: str = '/data/wongfanc/gb1-results',
    pp_csv_fname: str = None,
    cal_csv_fname: str = None,
    design_samples_fname_prefix: str = 'gb1-h10-5k-030123-samples',
    load_design_samples: bool = True,
    save_design_samples: bool = False
):
    
    # standardize temperatures and desired values (tau)
    temperatures = [round(t, 4) for t in temp2theta.keys()]
    desired_values = np.array([round(v, 4) for v in desired_values])
    
    # initialize dataframes for recording results
    pp_selected_column_names = ['tr{}_pp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    cal_column_names = ['cal_mean_temp{:.4f}'.format(temp) for temp in temperatures]
    pp_df = DataFrame(index=desired_values, columns=pp_selected_column_names)  # for our method
    cal_df = DataFrame(index=range(n_trial), columns=cal_column_names)  # for calibrated forecast baseline

    if pp_csv_fname is not None:
        assert(cal_csv_fname is not None)

    if load_design_samples and save_design_samples:
        raise ValueError('Only one of load_design_samples or save_design_samples can be True.') 
    if load_design_samples or save_design_samples:
        if design_samples_fname_prefix is None:
            raise ValueError('Provide design_samples_fname_prefix.')
        
    print('Selection quantity is the mean label.')
    print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(np.min(desired_values), np.max(desired_values)))

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

        # subsample designs for calibrated forecast baseline, for speed
        forecast_idx = np.random.choice(preddesign_N.size, size=n_design_forecasts, replace=False)
        designmu_n = preddesign_N[forecast_idx]
        designsigma_n = np.std(preddesign_Nxm[forecast_idx, :], axis=1, keepdims=False)
        # heuristic for limits of integration for computing E[Y] given CDF of Y
        pos_int_limit = np.max(designmu_n + 3 * designsigma_n)
        neg_int_limit = np.min(designmu_n - 3 * designsigma_n)


        # ----- trials over draws of held-out labeled data -----
        for i in range(n_trial):

            # sample sequences from NNK library
            _, calohe_nxlxa, calseq_n = sample_ohe_from_nuc_distribution(
                PNUC_NNK_LXA, n_labeled, normalize=False, reject_stop_codon=True
            )
            # get labels
            ycal_n = np.array([SEQ2YVAR[seq][0] for seq in calseq_n])

            # predictions for labeled sequences
            predcal_nxm = model.ensemble_predict(calohe_nxlxa)
            predcal_n = np.mean(predcal_nxm, axis=1, keepdims=False)
            callogptrain_n = get_nostop_loglikelihood(calohe_nxlxa, PAA_NNK_LXA)

            # density ratios for labeled sequences
            caldr_n = get_density_ratios(calohe_nxlxa, theta_lxa, logptrain_n=callogptrain_n)
            caldr_n = caldr_n / np.sum(caldr_n) * caldr_n.size  # self-normalize importance weights
        
            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal_n)
            rectifier_mean = np.mean(rect_n)
            rectifier_se = np.std(rect_n) / np.sqrt(rect_n.size)

            # ===== PP (our method) =====
            for tau in desired_values:
                pp_pval = rectified_p_value(
                    rectifier_mean,
                    rectifier_se,
                    imputed_mean,
                    imputed_se,
                    null=tau,
                    alternative='larger'
                )
                pp_df.loc[tau]['tr{}_pp_pval_temp{:.4f}'.format(i, temp)] = pp_pval

            # ===== calibrated forecasts method =====
            calsigma_n = np.std(predcal_nxm, axis=1, keepdims=False)
            calF_n = sc.stats.norm.cdf(ycal_n, loc=predcal_n, scale=calsigma_n)
            calempF_n = np.mean(calF_n[:, None] <= calF_n[None, :], axis=0, keepdims=False)
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            ir.fit(calF_n, calempF_n)

            # subsample designs
            calmu_N, t1_err, t2_err = utils.get_mean_from_cdf(
                designmu_n,
                designsigma_n,
                ir,
                (0, pos_int_limit),
                (neg_int_limit, 0),
                quad_limit=quad_limit,
                err_norm='max',
                tol=tol,
            )
            cal_df.loc[i, 'cal_mean_temp{:.4f}'.format(temp)] = np.mean(calmu_N)
                    
        print('Done with {} trials for temperature {:.4f} ({} / {}) ({} s).'.format(
            n_trial, temp, t + 1, len(temperatures), int(time() - t0)
        ))

        if pp_csv_fname is not None:
            pp_df.to_csv(pp_csv_fname)
            cal_df.to_csv(cal_csv_fname)
            print('Saved PP results to {} and CalibratedForecasts results to {}.'.format(pp_csv_fname, cal_csv_fname))

    return pp_df, cal_df


# ===== process selection results for plotting =====

def process_pvalues_for_plotting(
    df,
    desired_values,
    temperatures,
    temp2mean,
    n_trial: int,
    method_name: str = 'po',  # 'po' or 'pp'
    alpha: float = 0.1
):

    n_temp = temperatures.shape[0]
    alpha_bonferroni = alpha / n_temp
    print('Processing {} results with {} temperatures in [{:.3f}, {:.3f}], {} desired values in [{:.2f}, {:.2f}], {} trials, and alpha = {:.1f}'.format(
        method_name, n_temp, np.min(temperatures), np.max(temperatures),
        desired_values.shape[0], np.min(desired_values), np.max(desired_values),
        n_trial, alpha
    ))

    worst_v = []  # worst (i.e. lowest) mean design label achieved by any selected temperature, for each desired value (tau)
    err_v = []    # error rate, for each desired value
    sel_v = []    # selection rate, for each desired value

    t0 = time()
    for val in desired_values:
        val = round(val, 4)
            
        worst_t = []  # worst (i.e. lowest) mean label for trials where a temperature was selected
        
        for i in range(n_trial):
            selected = [temp for temp in temperatures if df.loc[val]['tr{}_{}_pval_temp{:.4f}'.format(i, method_name, temp)] < alpha_bonferroni]
            achieved = [temp2mean[round(t, 4)] for t in selected]

            if len(selected):
                worst_t.append(np.min(achieved))
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        sel_v.append(len(worst_t) / n_trial)

    print('Done processing ({} s)'.format(int(time() - t0)))
    return err_v, sel_v, worst_v


def process_gmmforecasts_for_plotting(
    gmm_df,
    desired_values,
    temperatures,
    temp2mean,
    n_trial: int,
    gmm_qs = None,
):
    if gmm_qs is None:
        gmm_qs = DEFAULT_GMM_QS

    q2results = {}
    for q in gmm_qs:
        err_v, sel_v, worst_v = process_forecasts_for_plotting(
            gmm_df, desired_values, temperatures, temp2mean, n_trial, method_name='gmm-cs-q{:.2f}'.format(q)
        )
        q2results[q] = err_v, sel_v, worst_v
    return q2results


def process_forecasts_for_plotting(
    df,
    desired_values,
    temperatures,
    temp2mean,
    n_trial: int,
    method_name: str,
):
    desired_values = [round(val, 4) for val in desired_values]
    temperatures = [round(temp, 4) for temp in temperatures]
    val2selected = {val: [] for val in desired_values}

    # for each desired value, collect the selected temperatures
    for i in range(n_trial):
        for val in desired_values:
            val2selected[val].append([])

        for temp in temperatures:
            forecast = df.loc[i]['{}_mean_temp{:.4f}'.format(method_name, temp)]

            for val in desired_values:
                if forecast >= val:
                    val2selected[val][i].append(temp)

    worst_v = []  # worst (i.e. lowest) mean design label achieved by any selected temperature, for each desired value (tau)
    err_v = []    # error rate, for each desired value
    sel_v = []    # selection rate, for each desired value
    for val in desired_values:                    
        worst_t = []    # worst (i.e. lowest) mean design label for each trial
        for i in range(n_trial):
            achieved = [np.mean(temp2mean[temp]) for temp in val2selected[val][i]]
            if len(achieved):
                worst_t.append(np.min(achieved))
                
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        sel_v.append(len(worst_t) / n_trial)
                        
    return err_v, sel_v, worst_v