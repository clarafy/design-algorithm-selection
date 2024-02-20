from pathlib import Path
import os.path
import copy

from time import time
from tqdm import tqdm
import numpy as np
import scipy as sc
from pandas import DataFrame
from Bio.Seq import Seq
from statsmodels.stats.weightstats import _zstat_generic

import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
from calibrate import rectified_p_value
from aav_util import SequenceTools

AA = ''.join(SequenceTools.protein2codon_.keys())
AA2IDX = {aa: idx for idx, aa in enumerate(AA)}
NUCLEOTIDES = 'atcg'
NUC2IDX = {nuc: idx for idx, nuc in enumerate(NUCLEOTIDES)}

# ===== models for predicting AAV packaging from sequence =====

def type_check_and_one_hot_encode_sequences(seq_n, alphabet, verbose: bool = False):
    if isinstance(seq_n[0], str):
        t0 = time()
        ohe_nxla = np.stack([utils.str2onehot(seq, alphabet).flatten() for seq in seq_n])
        if verbose:
            print('One-hot encoded sequences to shape = {} ({} sec)'.format(ohe_nxla.shape, int(time() - t0)))

    elif type(seq_n[0]) is np.ndarray:
        if verbose:
            print('Sequences are already one-hot encoded.')
        if len(seq_n.shape) == 2:
            # assume seq_n is already shaped like ohe_nxla
            ohe_nxla = seq_n.copy()
        elif len(seq_n.shape) == 3:
            # assume seq_n is shaped like ohe_nxlxa
            shape = seq_n.shape
            ohe_nxla = np.reshape(seq_n, [shape[0], shape[1] * shape[2]])
        else:
            raise ValueError('seq_n has shape {} with length {}, unclear how to reshape.'.format(seq_n.shape, len(seq_n.shape)))
        
    else:
        raise ValueError('Unrecognized seq_n type: {} is type {}'.format(seq_n[0], type(seq_n[0])))
    return ohe_nxla

class EnrichmentFeedForward(torch.nn.Module):
    def __init__(
         self,
            seq_len: int = 7,
            alphabet: str = AA,
            n_hidden: int = 100,
            n_model: int = 3,
            device = None,
            dtype = torch.float
    ):
        super().__init__()

        self.seq_len = seq_len
        self.alphabet = alphabet
        self.input_sz = seq_len * len(alphabet)
        self.device = device
        self.dtype = dtype

        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_sz, n_hidden),
                nn.ReLU(), 
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
            )
        for _ in range(n_model)])
        self.models.to(device, dtype=dtype)
        
    def forward(self, tX_nxla):
        pred_nxm = torch.cat([model(tX_nxla) for model in self.models], dim=1)
        return torch.mean(pred_nxm, dim=1, keepdim=False)
    
    def weighted_mse_loss(self, y_b, pred_b, weight_b):
        return torch.mean(weight_b * (y_b - pred_b) ** 2)
    
    def fit(
        self,
        seq_n,
        y_nx2: np.array,
        batch_size: int = 64,
        n_epoch: int = 5,
        lr: float = 0.001,
        val_frac: float = 0.1,
        n_data_workers: int = 1
    ):
        if val_frac < 0:
            raise ValueError('val_frac = {} must be positive.'.format(val_frac))
        
        ohe_nxla = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=True)
        dataset = [(ohe_la, y_mean_var[0], y_mean_var[1]) for ohe_la, y_mean_var in zip(ohe_nxla, y_nx2)]

        # split into training and validation
        shuffle_idx = np.random.permutation(len(dataset))
        n_val = int(val_frac * len(dataset))
        n_train = len(dataset) - n_val
        train_dataset = [dataset[i] for i in shuffle_idx[: n_train]]
        val_dataset = [dataset[i] for i in shuffle_idx[n_train :]]
        assert(len(val_dataset) == n_val)
        print('{} training data points, {} validation data points.'.format(n_train, n_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_data_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_data_workers)
        
        optimizer = torch.optim.Adam(self.models.parameters(), lr=lr)
        loss_tx2 = np.zeros([n_epoch, 2])
        best_val_loss = np.inf
        best_model_parameters = None
        for t in range(n_epoch):
            
            t0 = time()

            # validation loss
            self.requires_grad_(False)
            total_val_loss = 0.
            for _, data in enumerate(tqdm(val_loader)):
                tX_bxla, tymean_b, tyvar_b = data
                tX_bxla = tX_bxla.to(device=self.device, dtype=self.dtype)
                tymean_b = tymean_b.to(device=self.device, dtype=self.dtype)
                tyvar_b = tyvar_b.to(device=self.device, dtype=self.dtype)

                pred_b = self(tX_bxla)
                loss = self.weighted_mse_loss(tymean_b, pred_b, 1 / (2 * tyvar_b))
                total_val_loss += loss.item() * tX_bxla.shape[0]
            total_val_loss /= n_val

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_parameters = copy.deepcopy(self.state_dict())

            # gradient step on training loss
            self.requires_grad_(True)
            total_train_loss = 0.
            for _, data in enumerate(tqdm(train_loader)):
                tX_bxla, tymean_b, tyvar_b = data
                tX_bxla = tX_bxla.to(device=self.device, dtype=self.dtype)
                tymean_b = tymean_b.to(device=self.device, dtype=self.dtype)
                tyvar_b = tyvar_b.to(device=self.device, dtype=self.dtype)

                optimizer.zero_grad()

                pred_b = self(tX_bxla)

                loss = self.weighted_mse_loss(tymean_b, pred_b, 1 / (2 * tyvar_b))
                loss.backward()

                optimizer.step()
                total_train_loss += loss.item() * tX_bxla.shape[0]

            total_train_loss /= n_train
            loss_tx2[t] = total_train_loss, total_val_loss
            print('Epoch {}. Train loss: {:.2f}. Val loss: {:.2f}. {} sec.'.format(t, total_train_loss, total_val_loss, int(time() - t0)))
        
        self.load_state_dict(best_model_parameters)
        self.requires_grad_(False)
        return loss_tx2

    def predict(self, seq_n, verbose: bool = False):
        ohe_nxla = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=verbose)
        tohe_nxla = torch.from_numpy(ohe_nxla).to(device=self.device, dtype=self.dtype)
        return self(tohe_nxla).cpu().detach().numpy()
    
    def save(self, save_fname_no_ftype, save_path: str = '/homefs/home/wongfanc/density-ratio-estimation/aav-models'):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(save_path, save_fname_no_ftype + '.pt')
        torch.save(self.state_dict(), fname)
        print('Saved models to {}.'.format(fname))
    
    def load(self, save_fname_no_ftype, save_path: str = '/homefs/home/wongfanc/density-ratio-estimation/aav-models'):
        fname = os.path.join(save_path, save_fname_no_ftype + '.pt')
        self.load_state_dict(torch.load(fname))

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


def sample_ohe_from_nuc_distribution(p_lxa, n_seq, normalize: bool = False):
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
    nucidx_nxl = np.array([np.random.choice(len(NUCLEOTIDES), n_seq, p=p_lxa[i]) for i in range(nuc_seq_len)]).T
    # convert to OHE nucleotides
    nucohe_nxlxa = np.eye(len(NUCLEOTIDES))[nucidx_nxl]

    # ----- convert to amino acids -----
    # convert each sequence of nucleotide indices into OHE amino acids
    aaidx_nxl = np.empty([n_seq, int(nuc_seq_len / 3)])
    for i in range(n_seq):
        nucidx_l = nucidx_nxl[i]
        # convert nucleotide indices to nucleotide bases
        nucseq = ''.join([NUCLEOTIDES[idx] for idx in nucidx_l])
        # translate nucleotides into amino acids
        aaseq = str(Seq(nucseq).translate()).lower()
        # convert amino acids to amino acid indices
        aaidx_nxl[i] = [AA2IDX[aa] for aa in aaseq]
    # convert to OHE amino acids
    aaohe_nxlxa = np.eye(len(AA))[aaidx_nxl.astype(int)]

    return nucohe_nxlxa, aaohe_nxlxa


# ===== solving optimization problem to define design distribution =====

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

def fit_mle_paa(aaohe_nxlxa: np.array):
    counts_lxa = np.sum(aaohe_nxlxa, axis=0, keepdims=False)
    paa_lxa = counts_lxa / np.sum(counts_lxa, axis=1, keepdims=True)
    return paa_lxa

def get_aa_probs_from_nuc_probs(pnuc_lxa: np.array):  # TODO: test
    """
    Computes amino acid site-wise categorical distribution probabilities
    given nucleotide site-wise categorical distribution probabilities.
    """
    nuc_seq_len = pnuc_lxa.shape[0]
    paa_axl = DataFrame(0., index=list(AA), columns=range(1, int(nuc_seq_len / 3) + 1))
    for aa in AA:
        codons = SequenceTools.protein2codon_[aa]
        for cod_site in range(int(nuc_seq_len / 3)):  # for each codon site
            for cod in codons:
                p_cod = 1
                for i in range(3):  # for each nucleotide in this codon
                    nuc_idx = NUC2IDX[cod[i]]
                    p_cod *= pnuc_lxa[cod_site * 3 + i, nuc_idx]
                paa_axl[cod_site + 1].loc[aa] += p_cod
    return np.array(paa_axl).T

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

def get_loglikelihood(ohe_nxlxa: np.array, logp_lxa: np.array):
    """
    Calculates the log-probability of OHE sequences given the log-probabiliies of site-wise categorical distributions.
    """
    logp_n = np.sum(ohe_nxlxa * logp_lxa[None, :, :], axis=(1, 2), keepdims=False)
    return logp_n

def solve_max_entropy_library(
    model: EnrichmentFeedForward,
    temperature: float,
    lr: float = 0.01,
    n_sample: int = 1000,
    n_iter: int = 2000,
    print_every: int = 500,
    initialization: str = 'rand',
    save_path: str = '/homefs/home/wongfanc/density-ratio-estimation/aav-models',
    save_fname_no_ftype: str = None
):
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

        nucohe_nxlxa, aaohe_nxlxa = sample_ohe_from_nuc_distribution(p_lxa, n_sample, normalize=False)
        grad_logp_nxlxa = nucohe_nxlxa - p_lxa[None, :, :]

        pred_n = model.predict(aaohe_nxlxa)
        logp_n = get_loglikelihood(nucohe_nxlxa, logp_lxa)
        w_n = pred_n - temperature * (1 + logp_n)

        grad_theta_nxlxa = w_n[:, None, None] * grad_logp_nxlxa
        theta_lxa = theta_lxa + lr * np.mean(grad_theta_nxlxa, axis=0)

        # record and print metrics
        meanpred = np.mean(pred_n)
        entropy = get_entropy(p_lxa, normalize=False)
        epd = get_expected_pairwise_distance(p_lxa, normalize=False)
        obj = meanpred + temperature * entropy
        df_rows.append([obj, meanpred, entropy, epd])
        if t == 0 or (t + 1) % print_every == 0:
            print('Iter: {}. Objective: {:.2f}. Mean prediction: {:.2f}. Entropy: {:.2f}. EPD: {:.2f}'.format(
                t + 1, obj, meanpred, entropy, epd
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
    

# ===== rejection sampling labeled sequences from design distribution =====
        
# NNK library (training sequence distribution):
# nucleotide categorical distributions per site in one codon
pnuc_nnk_onecodon = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0, 0.5, 0, 0.5],
])
pnuc_nnk_lxa = np.tile(pnuc_nnk_onecodon, [7, 1])

# amino acid categorical distribution corresponding to NNK
PAA_NNK_LXA = get_aa_probs_from_nuc_probs(pnuc_nnk_lxa)
LOG_PAA_NNK_LXA = np.log(PAA_NNK_LXA)

PAA_TRAIN_LXA = np.array([[0.11761193, 0.10138472, 0.05234972, 0.04028293, 0.0213146 ,
        0.052903  , 0.06628948, 0.01273418, 0.02993992, 0.08095019,
        0.05965383, 0.0891105 , 0.01228438, 0.05747265, 0.02829436,
        0.06989968, 0.02867845, 0.03098742, 0.01180407, 0.00625449,
        0.02979949],
       [0.07070959, 0.10591742, 0.0317805 , 0.05700566, 0.02261185,
        0.01913062, 0.06026451, 0.02132664, 0.02921909, 0.09188927,
        0.08406369, 0.09549338, 0.02913339, 0.0359691 , 0.01658605,
        0.08830269, 0.03262058, 0.05244712, 0.02216111, 0.00547755,
        0.02789016],
       [0.08291938, 0.11901897, 0.04583929, 0.04722025, 0.02426138,
        0.03359442, 0.06821308, 0.02453159, 0.02679402, 0.08013197,
        0.07285406, 0.10083951, 0.04038009, 0.04053981, 0.01664393,
        0.07148338, 0.02506124, 0.03692494, 0.01392035, 0.00936847,
        0.01945987],
       [0.07935584, 0.12050446, 0.04084813, 0.04881787, 0.02629629,
        0.03079087, 0.06813053, 0.02455345, 0.0252316 , 0.08218196,
        0.07497829, 0.10949733, 0.03628035, 0.0380765 , 0.01382483,
        0.07510842, 0.02598457, 0.03873804, 0.01385827, 0.00812536,
        0.01881703],
       [0.09299593, 0.09486072, 0.03356753, 0.05783359, 0.02381883,
        0.0310868 , 0.05627689, 0.02035748, 0.02591687, 0.09496688,
        0.07822228, 0.12213248, 0.02630038, 0.04649791, 0.01689192,
        0.07505873, 0.02109619, 0.04610832, 0.01513412, 0.00525201,
        0.01562414],
       [0.11247837, 0.08311172, 0.02778528, 0.04948233, 0.01456482,
        0.02145327, 0.04332757, 0.01730278, 0.02873434, 0.11364338,
        0.07822275, 0.16170114, 0.02837925, 0.03457902, 0.01375035,
        0.07244916, 0.02437526, 0.03224713, 0.01258604, 0.00650927,
        0.02331677],
       [0.09364111, 0.11485527, 0.02239484, 0.02783334, 0.01943964,
        0.03182166, 0.06675881, 0.01239616, 0.03424556, 0.09220823,
        0.09352699, 0.13134767, 0.02100487, 0.03702023, 0.01522789,
        0.08378764, 0.03034774, 0.02979891, 0.01248677, 0.00953509,
        0.02032158]])

LOG_PAA_TRAIN_LXA = np.log(PAA_TRAIN_LXA)

def get_rejection_sampling_acceptance_probabilities(trainseq_n, theta_lxa):
    pnuc_lxa = normalize_theta(theta_lxa)
    paa_lxa = get_aa_probs_from_nuc_probs(pnuc_lxa)
    ratio_lxa = paa_lxa / PAA_NNK_LXA
    maxp_l = np.max(ratio_lxa, axis=1)
    M = np.prod(maxp_l)

    # compute test likelihoods of training sequences
    ohe_nxlxa = np.stack([utils.str2onehot(seq, AA) for seq in trainseq_n])
    logptest_n = get_loglikelihood(ohe_nxlxa, np.log(paa_lxa))

    # compute training likelihoods of training sequences
    logptrain_n = get_loglikelihood(ohe_nxlxa, LOG_PAA_NNK_LXA)
    paccept_n = np.exp(logptest_n - (np.log(M) + logptrain_n))
    return paccept_n

def rejection_sample_from_test_distribution(proposal_seq_n, y_n, theta_lxa: np.array):
    paccept_n = get_rejection_sampling_acceptance_probabilities(proposal_seq_n, theta_lxa)
    nonzero_samples_from_test = False
    while not nonzero_samples_from_test:
        accept_n = sc.stats.bernoulli.rvs(paccept_n)
        samp_idx = np.where(accept_n)[0]
        n_test = samp_idx.size
        if n_test:
            nonzero_samples_from_test = True
    sampseq_n = [proposal_seq_n[i] for i in samp_idx]
    ysamp_n = y_n[samp_idx]
    return sampseq_n, ysamp_n


# ===== temperature selection through multiple hypothesis testing =====

def get_density_ratios(aaohe_nxlxa: np.array, theta_lxa: np.array, logptrain_n: np.array = None):
    pnuc_lxa = normalize_theta(theta_lxa, compute_log=False)
    paa_lxa = get_aa_probs_from_nuc_probs(pnuc_lxa)
    logpdesign_n = get_loglikelihood(aaohe_nxlxa, np.log(paa_lxa))
    if logptrain_n is None:
        logptrain_n = get_loglikelihood(aaohe_nxlxa, LOG_PAA_NNK_LXA)
    return np.exp(logpdesign_n - logptrain_n)

def get_true_means_from_theta(temp2theta, valseq_n, yval_n: np.array):
    temp2mean = {}
    print('One-hot encoding validation sequences...')
    t0 = time()
    valohe_nxlxa = np.stack([utils.str2onehot(seq, AA) for seq in valseq_n])
    print('Done ({} sec)'.format(int(time() - t0)))
    logptrain_n = get_loglikelihood(valohe_nxlxa, LOG_PAA_NNK_LXA)

    print('True mean for temperature...')
    t0 = time()
    for temp, theta_lxa in temp2theta.items():
        valdr_n = get_density_ratios(valohe_nxlxa, theta_lxa, logptrain_n=logptrain_n)
        ess = np.square(np.sum(valdr_n)) / np.sum(np.square(valdr_n))
        temp2mean[temp] = np.mean(valdr_n * yval_n)
        print('    {:.4f} is {:.4f}, ESS = {}. ({} sec)'.format(temp, temp2mean[temp], int(ess), int(time() - t0)))
    return temp2mean
    
def run_temperature_selection_experiments(
    model: EnrichmentFeedForward,
    temp2theta,
    target_values: np.array,
    candidate_cal_seq_n,
    y_candidate_cal_n: np.array,
    n_cal: int = 100000,
    alpha: float = 0.1,
    n_trial: int = 100,
    n_design: int = 100000,
    print_every: int = 10,
):
    
    temperatures = list(temp2theta.keys())
    imp_selected_column_names = ['tr{}_imp_selected_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    pp_selected_column_names = ['tr{}_pp_selected_temp{:.4f}'.format(i, temp) for i in range(n_trial) for temp in temperatures]
    true_column_names = ['temp{:.4f}_true_mean'.format(temp) for temp in temperatures]
    df = DataFrame(
        index=target_values, columns=imp_selected_column_names + pp_selected_column_names + true_column_names
    )
    
    t0 = time() 
    for i in range(n_trial):
        cal_idx = np.random.choice(len(candidate_cal_seq_n), n_cal, replace=False)
        calseq_n = [candidate_cal_seq_n[i] for i in cal_idx]
        ycal_n = y_candidate_cal_n[cal_idx]
        calohe_nxlxa = np.stack([utils.str2onehot(seq, AA) for seq in calseq_n])

        temp_df = DataFrame(index=temperatures, columns=['imputed_mean', 'imputed_stderr', 'rectifier_mean', 'rectifier_stderr'])
        for temp, theta_lxa in temp2theta.items(): 
            # sample unlabeled sequences from design distribution
            _, designohe_nxlxa = sample_ohe_from_nuc_distribution(theta_lxa, n_design, normalize=True)

            # predictions for design sequences
            preddesign_n = model.predict(designohe_nxlxa)

            # imputation sample mean and standard error
            temp_df.loc[temp]['imputed_mean'] = np.mean(preddesign_n)
            temp_df.loc[temp]['imputed_stderr'] = np.std(preddesign_n) / np.sqrt(preddesign_n.size)

            # ----- quantities for prediction-powered hypothesis test -----
            # density ratios on labeled calibration sequences
            caldr_n = get_density_ratios(calohe_nxlxa, theta_lxa)
            caldr_n = caldr_n / np.sum(caldr_n) * len(calseq_n)  
            cal_ess = np.square(np.sum(caldr_n)) / np.sum(np.square(caldr_n))
            # print(i, temp, int(cal_ess))

            # predictions for calibration sequences
            predcal = model.predict(calohe_nxlxa)

            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal)
            temp_df.loc[temp]['rectifier_mean'] = np.mean(rect_n)
            temp_df.loc[temp]['rectifier_stderr'] = np.std(rect_n) / np.sqrt(len(calseq_n))

            # print('temp {:.4f}, imp mean {:.4f}, imp stderr {:.4f}, rect mean {:.4f}, rect stderr {:.4f}'.format(
            #     temp, temp_df.loc[temp]['imputed_mean'], temp_df.loc[temp]['imputed_stderr'],
            #     temp_df.loc[temp]['rectifier_mean'], temp_df.loc[temp]['rectifier_stderr']
            # ))

            for target_val in target_values:

                # run imputation hypothesis test
                imp_pval = _zstat_generic(
                    temp_df.loc[temp]['imputed_mean'],
                    0,
                    temp_df.loc[temp]['imputed_stderr'],
                    alternative='larger',
                    diff=target_val
                )[1]

                # run prediction-powered hypothesis test
                pp_pval = rectified_p_value(
                    temp_df.loc[temp]['rectifier_mean'],
                    temp_df.loc[temp]['rectifier_stderr'],
                    temp_df.loc[temp]['imputed_mean'],
                    temp_df.loc[temp]['imputed_stderr'],
                    null=target_val,
                    alternative='larger'
                )

                # Bonferroni correction
                if imp_pval < alpha / target_values.size:
                    df.loc[target_val]['tr{}_imp_selected_temp{:.4f}'.format(i, temp)] = 1
                    # print('    Imputation selected temp = {:.4f}'.format(temp))
                else:
                    df.loc[target_val]['tr{}_imp_selected_temp{:.4f}'.format(i, temp)] = 0

                if pp_pval < alpha / target_values.size:
                    df.loc[target_val]['tr{}_pp_selected_temp{:.4f}'.format(i, temp)] = 1
                    # print('    PP selected temp = {:.4f}'.format(temp))
                else:
                    df.loc[target_val]['tr{}_pp_selected_temp{:.4f}'.format(i, temp)] = 0
        if (i + 1) % print_every == 0:
            print('Done with {} / {} trials. {} s'.format(i + 1, n_trial, int(time() - t0)))

    return temp_df, df

