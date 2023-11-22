import copy
from time import time
from typing import Iterator
from itertools import chain


from torch.nn import Parameter
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_alphas(m, p: float = 1):
    a_m1 = np.sqrt(np.arange(0, m + 1) / m)  # convex combo of variances per dimension
    # a_m1 = np.power(np.arange(0, m + 1) / m, p)  # original
    return a_m1


def get_linear_waymarks(X0_nxd, Xm_nxd, a_m1):
    # X0_nxd is 'target'/test/numerator distribution
    # Xm_nxd is 'source'/train/denominator distribution
    m = a_m1.size - 1
    n, d = Xm_nxd.shape

    X_m1xnxd = np.zeros([m + 1, n, d])
    X_m1xnxd[0] = X0_nxd
    X_m1xnxd[-1] = Xm_nxd

    # randomly pick indices of samples to construct waymarks from
    i_mxnx2 = np.random.choice(n, size=(m - 1, n, 2), replace=True)
    coeff1_m = np.sqrt(1 - np.square(a_m1))
    coeff2_m = a_m1

    for k in range(1, m):
        i0_n = i_mxnx2[k - 1, :, 0]
        im_n = i_mxnx2[k - 1, :, 1]

        X_m1xnxd[k] = coeff1_m[k] * X0_nxd[i0_n] + coeff2_m[k] * Xm_nxd[im_n]

    return X_m1xnxd

def get_mixed_dim_waymarks(X0_nxd, Xm_nxd, n_ratio):
    n, d = Xm_nxd.shape

    X_m1xnxd = np.zeros([n_ratio + 1, n, d])
    X_m1xnxd[0] = X0_nxd
    X_m1xnxd[-1] = Xm_nxd

    # randomly pick indices of samples to construct waymarks from
    i_mxnx2 = np.random.choice(n, size=(n_ratio - 1, n, 2), replace=True)
    chunk_sz = int(d / n_ratio)

    for k in range(1, n_ratio):
        X0samp_nxd = X0_nxd[i_mxnx2[k - 1, :, 0]]
        Xmsamp_nxd = Xm_nxd[i_mxnx2[k - 1, :, 1]]

        X_m1xnxd[k] = np.hstack(
            [Xmsamp_nxd[:, : k * chunk_sz], X0samp_nxd[:, k * chunk_sz :]]
        )
        print('Waymark {} / {} has {}, {} dimensions from P0, Pm.'.format(
            k, n_ratio - 1, np.fmin(k * chunk_sz, d), np.fmax(0, d - k * chunk_sz)
        ))

    return X_m1xnxd


class Linear(torch.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        self.a = torch.nn.Parameter(torch.rand((in_size, 1)))
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, X_nxd: np.array):
        return self.b + torch.matmul(X_nxd, self.a)


class FeedForward(torch.nn.Module):
    def __init__(self, in_size, n_hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.model.parameters()

    def forward(self, X_nxd):
        return self.model(X_nxd)


class MixedDimensionsClassifier(torch.nn.Module):
    def __init__(self, seq_len, alphabet_sz, k, n_ratio, n_hidden):
        super().__init__()
        self.k = k
        self.n_ratio = n_ratio
        self.alphabet_sz = alphabet_sz
        self.seq_len = seq_len
        if k:
            self.modelpm = nn.Sequential(
                nn.Linear(k * alphabet_sz, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, alphabet_sz),
                nn.Threshold(0, 0)
            )
        else:
            self.pmlogprob_1xa = torch.nn.Parameter(-torch.rand(1, alphabet_sz))
        if k < n_ratio - 1:
            self.modelp0 = nn.Sequential(
                nn.Linear((seq_len - (k + 1)) * alphabet_sz, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, alphabet_sz),
                nn.Threshold(0, 0)
            )
        else:
            self.p0logprob_1xa = torch.nn.Parameter(-torch.rand(1, alphabet_sz))

    def forward(self, X_nxd):
        assert(X_nxd.shape[1] == self.seq_len * self.alphabet_sz)
        if self.k:
            comppm_xa = self.modelpm(X_nxd[:, : self.k * self.alphabet_sz])
        else:
            comppm_xa = self.pmlogprob_1xa
        if self.k < self.n_ratio - 1:
            compp0_xa = self.modelp0(X_nxd[:, (self.k + 1) * self.alphabet_sz :])
        else:
            compp0_xa = self.p0logprob_1xa

        X_nxa = X_nxd[:, self.k * self.alphabet_sz : (self.k + 1) * self.alphabet_sz]
        return torch.sum(X_nxa * (compp0_xa - comppm_xa), dim=1, keepdim=True)


class Quadratic(torch.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size

        # symmetric matrix values
        # TODO: enforce positive diagonals?
        self.W = torch.nn.Parameter(torch.randn((in_size, in_size)))
        # n_vals = int(in_size * (1 + in_size) / 2 - in_size)
        # self.vals = torch.nn.Parameter(torch.randn(n_vals))
        # self.i, self.j = np.triu_indices(self.in_size, k=1)

        # symmetric matrix diagonal values
        # self.diag = torch.nn.Parameter(torch.rand((in_size)))
        # self.diag_idx = range(in_size)

        # linear component
        self.a = torch.nn.Parameter(torch.rand((in_size, 1)))

        # bias
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, X_nxd):
        # TODO: reuse this memory
        # matrix template to fill in
        # W = torch.empty(self.in_size, self.in_size, device=self.device)
        # # fill in symmetric matrix with learned parameters
        # W[self.i, self.j] = self.vals
        # W.T[self.i, self.j] = self.vals
        # W[self.diag_idx, self.diag_idx] = self.diag

        temp_nxd = torch.matmul(X_nxd, self.W.triu() + self.W.triu(1).transpose(0, 1)) * X_nxd
        # temp_nxd = torch.matmul(X_nxd, W) * X_nxd
        quad_nx1 = temp_nxd.sum(axis=1, keepdim=True)
        return quad_nx1 + self.b + torch.matmul(X_nxd, self.a)



class UnsharedTelescopingLogDensityRatioEstimator(nn.Module):
    def __init__(self, model_class, in_size: int, n_ratio: int = 1, device = None, dtype = None,
                 use_mixeddimclassifier: bool = False, n_hidden: int = 32):
        super().__init__()
        if use_mixeddimclassifier:
            print('n_ratio must be the sequence length!')
            self.bridges = nn.ModuleList(
                [MixedDimensionsClassifier(n_ratio, 4, k, n_ratio, n_hidden) for k in range(n_ratio)]
            )
        else:
            self.bridges = nn.ModuleList([model_class(in_size) for _ in range(n_ratio)])
        for b in self.bridges:
            b.to(device=device, dtype=dtype)
        self.in_size = in_size
        self.n_ratio = n_ratio
        self.device = device
        self.dtype = dtype

    def fit(self, cfg, X_m1xnxd: np.array = None, X0_nxp: np.array = None, Xm_nxp: np.array = None):
        # generate waymarks
        if X_m1xnxd is None:
            print('Generating waymarks.')
            if cfg['waymark_type'] == 'linear':
                a_m1 = get_alphas(self.n_ratio)
                X_m1xnxd = get_linear_waymarks(X0_nxp, Xm_nxp, a_m1)
            elif cfg['waymark_type'] == 'mixed_dimensions':
                X_m1xnxd = get_mixed_dim_waymarks(X0_nxp, Xm_nxp, self.n_ratio)
            print('Generated {} waymarks.'.format(X_m1xnxd.shape[0] - 1))
        else:
            print('{} waymarks provided'.format(X_m1xnxd.shape[0] - 1))
            if X_m1xnxd.shape[0] - 1 != self.n_ratio:
                raise ValueError('Number of provided waymarks {} must match n_ratio {}'.format(
                    X_m1xnxd.shape[0], self.n_ratio
                ))

        # train each bridge
        train_dfs = []
        t0 = time()
        for i in range(self.n_ratio):
            df = self.fit_bridge(self.bridges[i], X_m1xnxd[i], X_m1xnxd[i + 1], cfg)
            print('Done fitting bridge {} / {} ({} s).'.format((i + 1), self.n_ratio, int(time() - t0)))
            train_dfs.append(df)
        return train_dfs

    def fit_bridge(self, bridge: nn.Module, X0_nxd, Xm_nxd, cfg):
        val_frac = cfg['val_frac']
        n_steps = cfg['n_steps']
        lr = cfg['lr']

        tX0_nxd = torch.from_numpy(X0_nxd).to(device=self.device, dtype=self.dtype)
        tXm_nxd = torch.from_numpy(Xm_nxd).to(device=self.device, dtype=self.dtype)
        n = X0_nxd.shape[0]

        z0_nx1 = torch.ones((n), 1, device=self.device, dtype=self.dtype)
        zm_nx1 = torch.zeros((n), 1, device=self.device, dtype=self.dtype)
        Xall_2nxd = torch.cat([tX0_nxd, tXm_nxd])
        zall_2nx1 = torch.cat([z0_nx1, zm_nx1])

        if val_frac > 0:
            n_val = int(val_frac * 2 * n)
            perm = np.random.permutation(2 * n)
            Xtr_xd = Xall_2nxd[perm[n_val:]]
            ztr_x1 = zall_2nx1[perm[n_val:]]
            Xval_xd = Xall_2nxd[perm[:n_val]]
            zval_x1 = zall_2nx1[perm[:n_val]]
        else:
            print("No validation data for bridge.")
            n_val = 0
            Xtr_xd = Xall_2nxd
            ztr_x1 = zall_2nx1
            Xval_xd = None
            zval_x1 = None

        loss_fn = torch.nn.BCEWithLogitsLoss()
        bridge.requires_grad_(True)
        optim = torch.optim.Adam(bridge.parameters(), lr=lr)

        best_loss = float('inf')
        val_loss = 0
        ckpt_losses = []

        for _ in tqdm(range(n_steps)):
            if n_val:
                with torch.no_grad():
                    val_logits = bridge(Xval_xd)
                    val_loss = loss_fn(val_logits, zval_x1)

            bridge.zero_grad()
            train_logits = bridge(Xtr_xd)
            train_loss = loss_fn(train_logits, ztr_x1)

            if n_val and val_loss < best_loss:
                best_loss = val_loss.item()
                best_weights = copy.deepcopy(bridge.state_dict())
            elif train_loss < best_loss:
                best_loss = train_loss.item()
                best_weights = copy.deepcopy(bridge.state_dict())

            train_loss.backward()
            optim.step()
            ckpt_losses.append([
                train_loss.item(),
                val_loss.item() if n_val else val_loss,
            ])

        bridge.load_state_dict(best_weights)
        bridge.requires_grad_(False)
        df = pd.DataFrame(ckpt_losses, columns=["train_loss", "val_loss"])
        return df

    def _get_ldr_nxm(self, tX_nxp: torch.Tensor):
        return torch.cat([self.bridges[i](tX_nxp) for i in range(self.n_ratio)], 1)

    def forward(self, tX_nxp: torch.Tensor):
        ldr_nxm = self._get_ldr_nxm(tX_nxp)
        return torch.sum(ldr_nxm, 1, keepdim=True)

    def predict_log_dr(self, X_nxp: np.array):
        tX_nxp = torch.from_numpy(X_nxp).to(device=self.device, dtype=self.dtype)
        ldre_nx1 = self(tX_nxp).cpu().detach().numpy()
        return ldre_nx1.squeeze(-1)

    def forecast_meany(self, Xm_nxp, ym_n, predm_n: np.array = None, pred0_n: np.array = None):
        # self-normalized estimate, as suggested by Grover et al. (2019) results
        logdr_n = self.predict_log_dr(Xm_nxp)
        # log-sum-exp trick
        c = np.max(logdr_n)
        normalization = c + np.log(np.sum(np.exp(logdr_n - c)))
        normalizeddr_n = np.exp(logdr_n - normalization)
        if predm_n is None or pred0_n is None:
            return np.sum(normalizeddr_n * ym_n)
        return np.mean(pred0_n) - np.sum(normalizeddr_n * (predm_n - ym_n))
    
    def forecast_meany_per_bridge(self, Xm_nxp, ym_n, predm_n: np.array = None, pred0_n: np.array = None, self_normalized: bool = True):
        tXm_nxp = torch.from_numpy(Xm_nxp).to(device=self.device, dtype=self.dtype)
        ldr_nxm = self._get_ldr_nxm(tXm_nxp).cpu().detach().numpy()
        # m = 0 corresponds to going to design distribution, so flip
        ldr_nxm = np.fliplr(np.cumsum(np.fliplr(ldr_nxm), axis=1))
        dr_nxm = np.exp(ldr_nxm)
        
        if self_normalized:
            c_1xm = np.max(ldr_nxm, axis=0, keepdims=True)
            normalization_1xm = c_1xm + np.log(np.sum(np.exp(ldr_nxm - c_1xm), axis=0, keepdims=True))
            normalizeddr_nxm = np.exp(ldr_nxm - normalization_1xm)
            # dr_nxm = dr_nxm / np.sum(dr_nxm, axis=0, keepdims=True)  # equivalent to normalizeddr_nxm

            if predm_n is None or pred0_n is None:
                weightedym_nxm = normalizeddr_nxm * ym_n[:, None]
                forecast_m = np.sum(weightedym_nxm, axis=0, keepdims=False)
            else:
                weightedrectm_nxm = normalizeddr_nxm * (predm_n - ym_n)[:, None]
                forecast_m = np.mean(pred0_n) - np.sum(weightedrectm_nxm, axis=0, keepdims=False)
        else:
            if predm_n is None or pred0_n is None:
                weightedym_nxm = dr_nxm * ym_n[:, None]
                forecast_m = np.mean(weightedym_nxm, axis=0, keepdims=False)
            else:
                weightedrectm_nxm = dr_nxm *  (predm_n - ym_n)[:, None]
                forecast_m = np.mean(pred0_n) - np.mean(weightedrectm_nxm, axis=0, keepdims=False)

        return forecast_m, dr_nxm



