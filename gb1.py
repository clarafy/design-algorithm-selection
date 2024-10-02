from pathlib import Path
import os.path
from itertools import product
from time import time

import numpy as np
import scipy as sc
from pandas import DataFrame, read_csv
from statsmodels.stats.weightstats import _zstat_generic
from Bio.Seq import Seq

from models import EnrichmentFeedForward, ExceedancePredictor
import utils
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
ALL_NOSTOP_AA_OHE = np.stack([utils.str2onehot(seq, AA) for seq in ALL_NOSTOP_AA_SEQS])

NUCLEOTIDES = 'atcg'
NUC2IDX = {nuc: idx for idx, nuc in enumerate(NUCLEOTIDES)}

df = read_csv('../data/gb1-with-variance.csv')
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
    given the probabiliies of site-wise categorical distributions.
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
    ohe_nxlxa = np.stack([utils.str2onehot(seq, AA) for seq in ALL_NOSTOP_AA_SEQS])
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
    exceedance_threshold: float = None,
    n_design: int = 1000000,
    save_path: str = '/data/wongfanc/gb1-results',
    results_csv_fname: str = None,
    design_samples_fname_prefix: str = None,
    load_design_samples: bool = False,
    save_design_samples: bool = False
):
    
    temperatures = list(temp2theta.keys())
    target_values = np.array([round(v, 4) for v in target_values])
    imp_selected_column_names = ['tr{}_imp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    df = DataFrame(
        index=target_values, columns=imp_selected_column_names
    )

    if results_csv_fname is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results_fname = os.path.join(save_path, results_csv_fname)

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
        
        print('Done with temperature {:.4f} ({} / {}) ({} s)'.format(
            temp, t + 1, len(temperatures), int(time() - t0))
        )
        if results_csv_fname is not None:
            df.to_csv(results_fname) # time sink
            print('Saved to {} ({} s).'.format(results_fname, int(time() - t0)))
    
    return df

    
def run_temperature_selection_experiments(
    model: EnrichmentFeedForward,
    temp2theta,
    target_values: np.array,
    exceedance_threshold: float = None,
    n_cal: int = 5000,
    n_design: int = 1000000,
    n_trial: int = 1000,
    n_train_lr: int = 500,
    self_normalize_weights: bool = True,
    save_path: str = '/data/wongfanc/gb1-results',
    results_csv_fname: str = None,
    design_samples_fname_prefix: str = None,
    load_design_samples: bool = False,
    save_design_samples: bool = False
):
    
    temperatures = list(temp2theta.keys())
    target_values = np.array([round(v, 4) for v in target_values])
    imp_selected_column_names = ['imp_pval_temp{:.4f}'.format(temp) for temp in temperatures]
    pp_selected_column_names = ['tr{}_pp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    df = DataFrame(
        index=target_values, columns=imp_selected_column_names + pp_selected_column_names
    )

    if results_csv_fname is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results_fname = os.path.join(save_path, results_csv_fname)

    if load_design_samples and save_design_samples:
        raise ValueError('Only one of load_design_samples or save_design_samples can be True.') 
    if load_design_samples or save_design_samples:
        if design_samples_fname_prefix is None:
            raise ValueError('Provide design_samples_fname_prefix.')
        
    if exceedance_threshold is not None:
        print('Selection quantity is probability of exceeding {}.'.format(exceedance_threshold))
        # seq2y = {seq: (SEQ2YVAR[seq][0] >= exceedance_threshold).astype(float) for seq in SEQ2YVAR.keys()}
    else:
        print('Selection quantity is the mean label.')
        predictor = model
        # seq2y = SEQ2YVAR
    print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(np.min(target_values), np.max(target_values)))

    
    t0 = time() 
    for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

        # sampling design sequences is the bottleneck for computation
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
            
        # predictions for unlabeled design sequences
        if exceedance_threshold is not None:
            predictor = ExceedancePredictor(model, exceedance_threshold)
        preddesign_n = predictor.predict(designohe_nxlxa)
        imputed_mean = np.mean(preddesign_n)
        imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)

        for target_val in target_values:

            # imputation hypothesis test. only run one trial since N so large
            imp_pval = _zstat_generic(
                imputed_mean,
                0,
                imputed_se,
                alternative='larger',
                diff=target_val
            )[1]
            df.loc[target_val]['imp_pval_temp{:.4f}'.format(temp)] = imp_pval

        for i in range(n_trial):

            # sample labeled calibration data from NNK
            _, calohe_nxlxa, calseq_n = sample_ohe_from_nuc_distribution(
                PNUC_NNK_LXA, n_cal, normalize=False, reject_stop_codon=True
            )
            ycal_n = np.array([SEQ2YVAR[seq][0] for seq in calseq_n])

            if exceedance_threshold is not None:
                ycal_n = (ycal_n >= exceedance_threshold).astype(float)
                sampled_both_labels = False

                # TODO: valueerror if exceedance_threshold unreasonably high
                # make sure have enough labels above exceedance
                while not sampled_both_labels:  
                    shuffle_idx = np.random.permutation(n_cal)
                    train_idx, cal_idx = shuffle_idx[: n_train_lr], shuffle_idx[n_train_lr :]
                    sampled_both_labels = any(ycal_n[train_idx]) and any(ycal_n[cal_idx])

                trainohe_nxlxa = calohe_nxlxa[train_idx]
                ytrain_n = ycal_n[train_idx]
                calohe_nxlxa = calohe_nxlxa[cal_idx]
                ycal_n = ycal_n[cal_idx]
                predictor.fit(trainohe_nxlxa, ytrain_n)

                # predictions for unlabeled design sequences
                preddesign_n = predictor.predict(designohe_nxlxa)
                imputed_mean = np.mean(preddesign_n)
                imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)

            # ----- quantities for prediction-powered hypothesis test -----

            # predictions for calibration sequences
            predcal_n = predictor.predict(calohe_nxlxa)
            callogptrain_n = get_nostop_loglikelihood(calohe_nxlxa, PAA_NNK_LXA)

            # density ratios on labeled calibration sequences
            caldr_n = get_density_ratios(calohe_nxlxa, theta_lxa, logptrain_n=callogptrain_n)
            # cal_ess = np.square(np.sum(caldr_n)) / np.sum(np.square(caldr_n))
            if self_normalize_weights:
                caldr_n = caldr_n / np.sum(caldr_n) * caldr_n.size
        
            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal_n)
            rectifier_mean = np.mean(rect_n)
            rectifier_se = np.std(rect_n) / np.sqrt(rect_n.size)

            for target_val in target_values:

                # run prediction-powered hypothesis test
                pp_pval = rectified_p_value(
                    rectifier_mean,
                    rectifier_se,
                    imputed_mean,
                    imputed_se,
                    null=target_val,
                    alternative='larger'
                )
                df.loc[target_val]['tr{}_pp_pval_temp{:.4f}'.format(i, temp)] = pp_pval
                    
        print('Done with {} / {} trials for temperature {:.4f} ({} / {}) ({} s).'.format(
            n_trial, n_trial, temp, t + 1, len(temperatures), int(time() - t0)
        ))
        if results_csv_fname is not None:
            df.to_csv(results_fname) # time sink
            print('Saved to {} ({} s).\n'.format(results_fname, int(time() - t0)))

    return df


def run_temperature_selection_experiments_maxt(
    model: EnrichmentFeedForward,
    temp2theta,
    target_values: np.array,
    n_cal: int = 5000,
    n_design: int = 1000000,
    n_trial: int = 1000,
    n_bootstrap: int = 10000,
    self_normalize_weights: bool = True,
    save_path: str = '/data/wongfanc/gb1-results',
    results_csv_fname: str = None,
    design_samples_fname_prefix: str = None,
):
    
    temperatures = list(temp2theta.keys())
    target_values = np.array([round(v, 4) for v in target_values])
    imp_selected_column_names = ['imp_pval_temp{:.4f}'.format(temp) for temp in temperatures]
    pp_selected_column_names = ['tr{}_pp_pval_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    df = DataFrame(index=target_values, columns=imp_selected_column_names + pp_selected_column_names)

    if results_csv_fname is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results_fname = os.path.join(save_path, results_csv_fname)
        
    print('Selection quantity is the mean label.')
    predictor = model
    print('Range of provided target values: [{:.3f}, {:.3f}].\n'.format(np.min(target_values), np.max(target_values)))

    # ===== imputation =====
    t0 = time()
    imputed_mean_se_tx2 = np.zeros([len(temperatures), 2])
    for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

        # sampling design sequences is the bottleneck for computation
        design_samples_fname = os.path.join(save_path, '{}-t{:.4f}.npz'.format(design_samples_fname_prefix, temp))
        d = np.load(design_samples_fname)
        designohe_nxlxa = d['designohe_nxlxa']
        if designohe_nxlxa.shape[0] != n_design:
            raise ValueError('Loaded {} != n_design = {} design sequences from {}.'.format(
                designohe_nxlxa.shape[0], n_design, design_samples_fname
            ))
        print('Loaded {} design sequences from {}.'.format(n_design, design_samples_fname))
            
        # predictions for unlabeled design sequences
        preddesign_n = predictor.predict(designohe_nxlxa)
        imputed_mean = np.mean(preddesign_n)
        imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)
        imputed_mean_se_tx2[t] = imputed_mean, imputed_se

        for target_val in target_values:

            # imputation hypothesis test. only run one trial since N so large
            imp_pval = _zstat_generic(
                imputed_mean,
                0,
                imputed_se,
                alternative='larger',
                diff=target_val
            )[1]
            df.loc[target_val]['imp_pval_temp{:.4f}'.format(temp)] = imp_pval

    # ===== PP =====
    t0 = time()
    for i in range(n_trial):

        # ----- sample labeled calibration data from NNK -----
        _, calohe_nxlxa, calseq_n = sample_ohe_from_nuc_distribution(
            PNUC_NNK_LXA, n_cal, normalize=False, reject_stop_codon=True
        )
        ycal_n = np.array([SEQ2YVAR[seq][0] for seq in calseq_n])

        # predictions for calibration sequences
        predcal_n = predictor.predict(calohe_nxlxa)
        callogptrain_n = get_nostop_loglikelihood(calohe_nxlxa, PAA_NNK_LXA)

        # ---- PP point estimate for each temperature -----
        thetahat_t = np.zeros([len(temperatures)])
        caldr_txn = np.zeros([len(temperatures), n_cal])
        for t, (temp, theta_lxa) in enumerate(temp2theta.items()):

            # density ratios on labeled calibration sequences
            caldr_n = get_density_ratios(calohe_nxlxa, theta_lxa, logptrain_n=callogptrain_n)
            caldr_txn[t] = caldr_n
            if self_normalize_weights:
                caldr_n = caldr_n / np.sum(caldr_n) * caldr_n.size
        
            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal_n)
            rectifier_mean = np.mean(rect_n)

            # PP point estimate
            thetahat_t[t] = imputed_mean_se_tx2[t, 0] + rectifier_mean

        # --- bootstrap to estimate covariance of all temps' PP point estimates -----
        bootidx_bxn = np.random.choice(n_cal, size=(n_bootstrap, n_cal), replace=True)
        thetahatboot_txb = np.zeros([len(temperatures), n_bootstrap])
        for b, bootidx_n in enumerate(bootidx_bxn):
            caldrboot_txn = caldr_txn[:, bootidx_n]
            if self_normalize_weights:
                caldrboot_txn = n_cal * caldrboot_txn / np.sum(caldrboot_txn, axis=1)
            rectboot_txn = caldrboot_txn * (ycal_n - predcal_n)[None, :]
            thetahatboot_txb[:, b] = imputed_mean_se_tx2[:, 0] + np.mean(rectboot_txn, axis=1)

        # ----- simulate max-t statistics -----
        Sigmahat_txt = np.cov(thetahatboot_txb)
        z_xt = sc.stats.multivariate_normal.rvs(mean=None, cov=Sigmahat_txt, size=100000)
        sigmahat_t = np.sqrt(np.diag(Sigmahat_txt))
        maxt_ = np.max(z_xt / sigmahat_t[None, :], axis=1)

        # ----- compute prediction-powered max-t p-values -----
        for target_val in target_values:
            score_t = (thetahat_t - target_val) / sigmahat_t
            cdf_t = np.mean(score_t[:, None] >= maxt_[None, :], axis=1)
            p_t = 1 - cdf_t

            for t, temp in enumerate(temperatures):
                df.loc[target_val]['tr{}_pp_pval_temp{:.4f}'.format(i, temp)] = p_t[t]
        
        # ----- print and save progress -----
        print(f'Done with {i + 1} / {n_trial} trials ({int(time() - t0)} s).')
        if results_csv_fname is not None:
            df.to_csv(results_fname) # time sink
            print('Saved to {} ({} s).\n'.format(results_fname, int(time() - t0)))

    return df