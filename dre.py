import copy
from tqdm import tqdm
from time import time
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

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
            design_names,
            name2designdata,
            noise_sd: float = 0.,
            lr: float = 1e-3,
            quadratic_final_layer: bool = False,
            weight_loss: bool = True,
            weight_decay: float = 0.,
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
            if name != 'train' and name in design_names:
                group_idx = self._name2idx(name)
                assert(name not in idx2group[group_idx])
                idx2group[group_idx][name] = data
        
        for i, name2dd in idx2group.items():
            # if no design algorithms were assigned to this group, raise error
            if len(name2dd) == 1:
                assert('train' in name2dd)
                group_str = self.group_regex_strs[i]
                raise ValueError('Group {} has no design algorithms in name2designdata.'.format(group_str))

        # fit MDRE per group
        for i, name2dd in idx2group.items():
            group_str = self.group_regex_strs[i]
            if verbose:
                print('Fitting MDRE for {}, which has {} design algorithms:'.format(group_str, len(name2dd) - 1))
                for name in name2dd:
                    print(name)

            mdre = MultinomialLogisticRegresssionDensityRatioEstimator(
                seq_len=seq_len,
                n_distribution=len(name2dd),
                n_hidden=n_hidden,
                quadratic_final_layer=quadratic_final_layer,
                device=self.device,
            )
            loss_df = mdre.fit(
                name2dd,
                n_epoch=n_epoch,
                lr=lr,
                weight_decay=weight_decay,
                weight_loss=weight_loss,
                noise_sd=noise_sd,
                verbose=verbose
            )
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
    

class QuadraticLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuadraticLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # quadratic terms
        self.quad_oxixi = nn.Parameter(torch.Tensor(output_dim, input_dim, input_dim))
        nn.init.kaiming_uniform_(self.quad_oxixi, nonlinearity='relu')

        # linear terms
        self.lin_ixo = nn.Linear(input_dim, output_dim)
        
        # biases
        self.bias_o = nn.Parameter(torch.Tensor(output_dim))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.quad_oxixi)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias_o, -bound, bound)

    def forward(self, x_bxi): 
    
        # quadratic term 
        # tile output_dim times
        x_oxbxi = torch.unsqueeze(x_bxi, 0).expand(self.output_dim, x_bxi.shape[0], x_bxi.shape[1])
        tmp_oxbxi = torch.bmm(x_oxbxi, (self.quad_oxixi + self.quad_oxixi.transpose(1, 2)) / 2) * x_oxbxi
        quad_oxb = tmp_oxbxi.sum(axis=2, keepdim=False)

        # linear term
        lin_bxo = self.lin_ixo(x_bxi)
        
        # add bias 
        out_bxo = quad_oxb.T + lin_bxo + self.bias_o.unsqueeze(0)
        
        return out_bxo
    
    
class MultinomialLogisticRegresssionDensityRatioEstimator(nn.Module):
    def __init__(
            self,
            seq_len: int,
            n_distribution: int,  # including train distribution
            n_hidden: int,
            quadratic_final_layer: bool = False,
            alphabet: str = RNA_NUCLEOTIDES,
            device = None,
            dtype = torch.float,
        ):
        super().__init__()

        if quadratic_final_layer: 
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * len(alphabet), n_hidden),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                QuadraticLayer(n_hidden, n_distribution)
            )
        else:
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
        self.Xmean_lxa = None
        self.Xstd_lxa = None

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
            noise_sd: float = 0.,
            lr: float = 1e-3,
            weight_decay: float = 0.,
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
            seq_n = designname2data[name][0]
            X_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet)

            # randomly shuffle sequences
            shuffle_idx = np.random.permutation(len(seq_n))
            X_nxlxa = X_nxlxa[shuffle_idx]
            # pred_n = pred_n[shuffle_idx]

            # add Gaussian noise
            X_nxlxa += norm.rvs(loc=0, scale=noise_sd, size=X_nxlxa.shape)

            # convert to tensors
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

            # for input feature normalization
            # d is shorthand for lxa
            self.Xmean_lxa = Xtr_mnxd.mean(dim=0)
            self.Xstd_lxa = Xtr_mnxd.std(dim=0)

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
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float('inf')
        val_loss = 0
        ckpt_losses = []

        # training loop
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

    def forward(self, tX_nxlxa: torch.Tensor):
        tX_nxlxa = (tX_nxlxa - self.Xmean_lxa) / self.Xstd_lxa
        return self.model(tX_nxlxa)

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
    
