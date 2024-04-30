import copy
from tqdm import tqdm
from time import time
import re
import pickle

import numpy as np
import pandas as pd

import torch
from torch import nn

from models import type_check_and_one_hot_encode_sequences
from utils import RNA_NUCLEOTIDES


class MultiMDRE():

    def __init__(self, group_regex_strs, device = None):
        self.group_regex_strs = group_regex_strs
        self.group_regex = [re.compile(s) for s in group_regex_strs]
        self.n_groups = len(self.group_regex)
        self.idx2mdre = {}
        self.idx2lossdf = {}
        self.device = device

    def _name2idx(self, name):
        for i, regex in enumerate(self.group_regex):
            if regex.match(name) is not None:
                return i
        raise ValueError(f'{name} does not match any of the group regexs.')

    def fit(
            self,
            name2designdata,
            seq_len: int = 50,
            n_hidden = 256,
            n_epoch = 500,
            verbose: bool = False
        ):

        assert('train' in name2designdata)
        idx2group = {
            i: {'train': name2designdata['train']} for i in range(self.n_groups)
        }

        # assign each design algorithm to its group
        for name, data in name2designdata.items():
            if name != 'train':
                group_idx = self._name2idx(name)
                assert(name not in idx2group[group_idx])
                idx2group[group_idx][name] = data

        # fit MDRE per group
        for i, name2dd in idx2group.items():
            group_str = self.group_regex_strs[i]

            # if no design algorithms were assigned to this group, raise error
            if len(name2dd) == 1:
                assert('train' in name2dd)
                raise ValueError('Group {} has no design algorithms in name2designdata.'.format(group_str))
            
            if verbose:
                print('Fitting MDRE for {}, which has {} design algorithms:'.format(group_str, len(name2dd) - 1))
                for name in name2dd:
                    print(name)

            mdre = MultinomialLogisticRegresssionDensityRatioEstimator(
                seq_len=seq_len,
                n_distribution=len(name2dd),
                n_hidden=n_hidden,
                device=self.device,
            )
            loss_df = mdre.fit(name2dd, n_epoch=n_epoch, verbose=verbose)
            if verbose:
                print('Min train loss {:.2f}, min val loss {:.2f}'.format(
                    np.min(loss_df["train_loss"]), np.min(loss_df["val_loss"])
                ))
                print()
            self.idx2mdre[i] = mdre
            self.idx2lossdf[i] = loss_df
            

    def get_dr(
            self,
            calseq_n,
            design_name: str,
            self_normalize: bool = True,
            verbose: bool = False
        ):
        # get group index of this design algorithm
        group_idx = self._name2idx(design_name)
        if verbose:
            print('{} belongs to group {}'.format(design_name, self.group_regex_strs[group_idx]))
        
        # get estimated DRs from corresponding group MDRE
        designname2dr = self.idx2mdre[group_idx].get_dr(calseq_n, self_normalize=self_normalize)
        dr_n = designname2dr[design_name]
        return dr_n

    
class MultinomialLogisticRegresssionDensityRatioEstimator(nn.Module):
    def __init__(
            self,
            seq_len: int,
            n_distribution: int,  # including train distribution
            n_hidden: int,
            alphabet: str = RNA_NUCLEOTIDES,
            device = None,
            dtype = torch.float,
        ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * len(alphabet), n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden, n_distribution),
        )
        self.model.to(device=device, dtype=dtype)

        self.alphabet = alphabet
        self.n_distribution = n_distribution
        self.designname2idx = None

        self.device = device
        self.dtype = dtype

        self.numer_kd_m = []
        self.denom_kd_m = []

    def fit(
            self,
            designname2data: dict,
            n_epoch: int,
            lr: float = 1e-3,
            val_frac: float = 0.1,
            weight_loss: bool = True,
            verbose: bool = True,
        ):

        names = list(designname2data.keys())
        assert('train' in names)
        assert(len(names) == self.n_distribution)
        names.remove('train')
        names.sort()
        names = ['train'] + names  # convention will be that train distribution is category 0
        self.designname2idx = {name: i for i, name in enumerate(names)}
        
        # for training convenience, convert `designname2seqs`,
        # a dictionary that maps design name (e.g., 'adalead0.1-ridge') to tuple (sequences, labels, predictions),
        # to `tX_nxlxa_m`, a list of one-hot-encoded sequence representations.
        # `self.designname2idx[name]` gives the index i such that
        # tX_nxlxa_m[i] gives the sequence representations corresponding to `name`.
        tX_nxlxa_m = []
        if verbose:
            print('One-hot-encoding all {} categories of sequences...'.format(self.n_distribution))
        t0 = time()
        for name in names:
            X_nxlxa = type_check_and_one_hot_encode_sequences(designname2data[name][0], self.alphabet)
            X_nxlxa = X_nxlxa[np.random.permutation(X_nxlxa.shape[0])]
            tX_nxlxa_m.append(torch.from_numpy(X_nxlxa).to(device=self.device, dtype=self.dtype))
        if verbose:
            print('  Done. ({} s)'.format(int(time() - t0)))
        
        if val_frac > 0:
            Xtr_nxd_m = []
            ztr_n_m = []  # categorical labels for the different design distributions
            Xval_nxd_m = []
            zval_n_m = []
            for k, X_nxlxa in enumerate(tX_nxlxa_m):
                n = X_nxlxa.shape[0]
                n_val = int(val_frac * n)

                Xtr_nxd_m.append(X_nxlxa[n_val :])
                ztr_n_m.append(
                    k * torch.ones((X_nxlxa[n_val :].shape[0]), device=self.device, dtype=torch.int64)
                )

                Xval_nxd_m.append(X_nxlxa[: n_val])
                zval_n_m.append(
                    k * torch.ones((X_nxlxa[: n_val].shape[0]), device=self.device, dtype=torch.int64)
                )
            
            Xtr_mnxd = torch.cat(Xtr_nxd_m, dim=0)
            ztr_mn = torch.cat(ztr_n_m, dim=0)
            Xval_mnxd = torch.cat(Xval_nxd_m, dim=0)
            zval_mn = torch.cat(zval_n_m, dim=0)

            if weight_loss:
                # weights for training data
                n_tr = Xtr_nxd_m[0].shape[0]  # train distribution is category 0
                N_tr = Xtr_nxd_m[1].shape[0]

                # assume all design distributions have same amount of data
                for k in range(2, self.n_distribution):
                    assert(Xtr_nxd_m[k].shape[0] == N_tr)

                w_tr_train = self.n_distribution / ((self.n_distribution - 1) * (n_tr / N_tr) + 1)
                w_tr_design = (n_tr / N_tr) * w_tr_train

                wtr_m = w_tr_design * torch.ones((self.n_distribution), device=self.device, dtype=self.dtype)
                wtr_m[0] = w_tr_train

                # weights for validation data
                n_val = Xval_nxd_m[0].shape[0]
                N_val = Xval_nxd_m[1].shape[0]

                # assume all design distributions have same amount of data
                for k in range(2, self.n_distribution):
                    assert(Xval_nxd_m[k].shape[0] == N_val)
                
                w_val_train = self.n_distribution / ((self.n_distribution - 1) * (n_val / N_val) + 1)
                w_val_design = (n_val / N_val) * w_val_train

                wval_m = w_val_design * torch.ones((self.n_distribution), device=self.device, dtype=self.dtype)
                wval_m[0] = w_val_train
            else:
                wtr_m = wval_m = None

        else:
            raise ValueError('No validation data for density ratio estimation.')
            Xval_mnxd, zval_mn = None, None
            # TODO: implement Xtr_mnxd, ztr_mn

        val_loss_fn = torch.nn.CrossEntropyLoss(weight=wval_m)
        tr_loss_fn = torch.nn.CrossEntropyLoss(weight=wtr_m)
        self.model.requires_grad_(True)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_loss = float('inf')
        val_loss = 0
        ckpt_losses = []

        self.model.train()
        epochs = range(n_epoch)
        if verbose:
            epochs = tqdm(epochs)
        for _ in epochs:
            if val_frac > 0:
                with torch.no_grad():
                    self.model.eval()
                    vallogits = self.model(Xval_mnxd)
                    val_loss = val_loss_fn(vallogits, zval_mn)
                    self.model.train()

            self.model.zero_grad()
            trlogits = self.model(Xtr_mnxd)
            train_loss = tr_loss_fn(trlogits, ztr_mn)

            if val_frac > 0 and val_loss < best_loss:
                best_loss = val_loss.item()
                best_weights = copy.deepcopy(self.model.state_dict())
            elif train_loss < best_loss:
                best_loss = train_loss.item()
                best_weights = copy.deepcopy(self.model.state_dict())

            train_loss.backward()
            optim.step()
            ckpt_losses.append([
                train_loss.item(),
                val_loss.item() if n_val else val_loss,
            ])
        self.model.eval()

        self.model.load_state_dict(best_weights)
        self.model.requires_grad_(False)
        df = pd.DataFrame(ckpt_losses, columns=["train_loss", "val_loss"])
        return df

    def forward(self, tX_nxp: torch.Tensor):
        return self.model(tX_nxp)

    def get_dr(self, seq_n: np.array, self_normalize: bool = True):
        X_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet)
        tX_nxlxa = torch.from_numpy(X_nxlxa).to(device=self.device, dtype=self.dtype)

        th_nxm = self(tX_nxlxa)
        # train distribution is category 0
        ldr_nxm = (th_nxm[:, 1 :] - th_nxm[:, 0][:, None]).cpu().detach().numpy()
    
        if self_normalize:
            c_1xm = np.max(ldr_nxm, axis=0, keepdims=True)
            normalization_1xm = c_1xm + np.log(np.sum(np.exp(ldr_nxm - c_1xm), axis=0, keepdims=True))
            dr_nxm = np.exp(ldr_nxm - normalization_1xm) * len(seq_n)  # sum to n
            # dr_nxm = dr_nxm * len(seq_n) / np.sum(dr_nxm, axis=0, keepdims=True)  # equivalent normalization
        else:
            dr_nxm = np.exp(ldr_nxm)
        
        designname2dr = {name: dr_nxm[:, idx - 1] for name, idx in self.designname2idx.items()} 
        return designname2dr


def select_intermediate_iterations(
    name2designdata,
    design_name_prefix,
    threshold,
    n_iter: int = 20,
    verbose: bool = True,
):
    # check have designs for all candidate iterations
    all_names = []
    for i in range(n_iter):
        name = '{}t{}'.format(design_name_prefix, i)
        all_names.append(name)
        if name not in name2designdata:
            print(f'No design data for {name}, exiting.')
            return None
    
    # find interval between iterations where the mean difference in consecutive mean predictions
    # surpasses threshold
    threshold_satisfied = False
    for interval in range(1, int(n_iter / 2) + 1):
        
        # extract iterations an interval apart
        iters = list(range(interval - 1, n_iter, interval)) + [n_iter - 1]
        iters = list(set(iters))
        iters.sort()
        
        # compute difference in consecutive mean predictions
        mean_preds = []
        names = []
        for i in iters:
            name = '{}t{}'.format(design_name_prefix, i)
            names.append(name)
            _, _, preddesign_n = name2designdata[name]
            mean_preds.append(np.mean(preddesign_n))

        mean_consecutive_diff = np.mean(np.ediff1d(mean_preds))
        if verbose:
            print('Iterations {} (interval {}) have mean consecutive diff in mean prediction of {:.3f}.'.format(
                iters, interval, mean_consecutive_diff
            ))
        
        # check if surpasses threshold
        if mean_consecutive_diff > threshold:
            threshold_satisfied = True
            break
    
    # if no interval surpasses threshold, just return the final design distributions
    if not threshold_satisfied:
        names = ['{}t{}'.format(design_name_prefix, n_iter - 1)]
        _, _, preddesign_n = name2designdata[names[0]]
        if verbose:
            print('Returning iteration {} with mean prediction {:.3f}.'.format(
                n_iter - 1, np.mean(preddesign_n)
            ))
        iters = [n_iter - 1]
    
    # names to remove
    for name in names:
        all_names.remove(name)
    
    return iters, all_names


def is_intermediate_iteration_name(design_name: str, n_iter: int = 20):
    tok = design_name.split('-')  
    if 't' in tok[-1]:
        tok2 = tok[-1].split('t')
        if int(tok2[-1]) < n_iter - 1: # e.g. cbas-ridge-0.1t0
            return True
        

def prepare_name2designdata(
        design_pkl_fname,
        train_fname,
        intermediate_iter_threshold: float = 0.1,
        verbose: bool = True,
    ):

    # load labeled designs from all design algorithms
    with open(design_pkl_fname, 'rb') as f:
        name2designdata = pickle.load(f)
    for name, data in name2designdata.items():  # make sure all sequences are labeled 
        if data[1] is None and not is_intermediate_iteration_name(name):
            raise ValueError(f'No labels for {name}')

    # load labeled training sequences
    d = np.load(train_fname)
    trainseqs_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']
    if verbose:
        print(f'Loaded {ytrain_n.size} training points from {train_fname}.\n')
    name2designdata['train'] = (trainseqs_n, ytrain_n, None)

    # remove DbAS q > 0.6
    for threshold in np.arange(0.7, 0.91, 0.1):
        for it in range(20):
            name = 'dbas-ridge-{:.1f}t{}'.format(threshold, it)
            del name2designdata[name]
            if verbose:
                print(f'Removed {name} designs.')
    
    # remove Biswas
    for temperature in [0.05]:
        temp = round(temperature, 4)
        for model_name in ['ridge', 'ff', 'cnn']:
            name = f'biswas-{model_name}-{temp}'
            del name2designdata[name]
            if verbose:
                print(f'Removed {name} designs.')

    # select intermediate C/DbAS iterations to facilitate DRE for C/DbAS design distributions
    for threshold in np.arange(0.1, 1, 0.1):
        for weight_type in ['c', 'd']:
            prefix =  '{}bas-ridge-{:.1f}'.format(weight_type, threshold)
            out = select_intermediate_iterations(
                name2designdata,
                prefix,
                intermediate_iter_threshold,
                verbose=verbose
            )
            if out is not None:
                iters, names_to_remove = out
                if verbose:
                    print(f'Using the following iterations for {prefix}: {iters}.')
                    
                for name in names_to_remove:
                    del name2designdata[name] 
                    if verbose:
                        print(f'  Removed {name}')
                    
    if verbose:
        print('Design names:')
        for name in name2designdata:
            print(name)
    return name2designdata



    



    