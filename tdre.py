import copy
from time import time
from typing import Iterator
from itertools import chain


from torch.nn import Parameter
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

import torch
from torch import nn


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

def get_mixed_site_waymarks(X0_Nxd, Xm_nxd, n_waymarks_between):
    # handles N != nd, doesn't include either X0_Nxd or Xm_nxd in output
    N, d = X0_Nxd.shape
    n = Xm_nxd.shape[0]
    assert(d == Xm_nxd.shape[1])

    X_mxnxd = np.zeros([n_waymarks_between, N, d])

    # randomly pick indices of samples to construct waymarks from
    i0_mxN = np.random.choice(N, size=(n_waymarks_between, N), replace=True)
    im_mxN = np.random.choice(n, size=(n_waymarks_between, N), replace=True)
    chunk_sz = int(d / (n_waymarks_between + 1))

    for k in range(n_waymarks_between):
        X0samp_Nxd = X0_Nxd[i0_mxN[k]]
        Xmsamp_Nxd = Xm_nxd[im_mxN[k]]

        X_mxnxd[k] = np.hstack(
            [Xmsamp_Nxd[:, : (k + 1) * chunk_sz], X0samp_Nxd[:, (k + 1) * chunk_sz :]]
        )
        assert(Xmsamp_Nxd[:, : (k + 1) * chunk_sz].shape[1] == np.fmin((k + 1) * chunk_sz, d))
        assert(X0samp_Nxd[:, (k + 1) * chunk_sz :].shape[1] == np.fmax(0, d - (k + 1) * chunk_sz))
        print('Waymark {} / {} has {}, {} dimensions from P0, Pm.'.format(
            k + 1, n_waymarks_between, np.fmax(0, d - (k + 1) * chunk_sz), np.fmin((k + 1) * chunk_sz, d)
        ))
    return X_mxnxd


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
        print('Waymark {} / {} has {}, {} dimensions from Pm, P0.'.format(
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
    

class SharedFeedForward(torch.nn.Module):
    def __init__(self, in_size: int, n_heads: int, n_hidden: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.body = nn.Sequential(
            nn.Linear(in_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        # self.heads = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, 8), nn.ReLU(), nn.Linear(8, 1)) for _ in range(n_heads)])
        self.heads = nn.ModuleList([nn.Linear(n_hidden, 1) for _ in range(n_heads)])

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return chain(self.body.parameters(), self.heads.parameters())

    def forward(self, X_mxnxd, Xtr_nxd):
        assert(X_mxnxd.shape[0] == self.n_heads)

        hidden_mxnxh = self.body(X_mxnxd)
        hiddentr_nxh = self.body(Xtr_nxd)
        out_1x2n_m = []
        for k in range(self.n_heads - 1):
            out_2nx1 = torch.cat((self.heads[k](hidden_mxnxh[k]), self.heads[k](hidden_mxnxh[k + 1])), axis=0)
            out_1x2n_m.append(out_2nx1.T)

        outm_x1 = torch.cat([self.heads[-1](hidden_mxnxh[-1]), self.heads[-1](hiddentr_nxh)], axis=0)
        out_mx2n = torch.cat(out_1x2n_m, axis=0)
        return out_mx2n, outm_x1
    
    def get_ldr(self, tX_nxd):
        print('SharedFeedForward.get_ldr not tested')
        hidden_nxh = self.body(tX_nxd)
        out_mxn = torch.cat([head(hidden_nxh).T for head in self.heads], dim=0)
        return out_mxn


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
    
class MultiClass(torch.nn.Module):
    def __init__(self, in_size, n_class, n_hidden: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_class),
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.model.parameters()

    def forward(self, X_nxd):
        return self.model(X_nxd)
    
class EnsembleFeedForward(torch.nn.Module):
    def __init__(self, in_size, n_hidden: int = 64, n_ensemble: int = 3):
        super().__init__()
        self.models = nn.ModuleList([FeedForward(in_size, n_hidden=n_hidden) for _ in range(n_ensemble)])

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.models.parameters()

    def forward(self, X_nxd):
        out_nxe = torch.concat([model(X_nxd) for model in self.models], dim=1)
        return out_nxe.T



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



class TelescopingLogDensityRatioEstimator(nn.Module):
    def __init__(self, model_class, in_size: int, n_ratio: int = 1, device = None, dtype = None, multiclass: bool = False,
                 use_mixeddimclassifier: bool = False, cranmer: bool = False, shared: bool = False, n_hidden: int = 32, n_waymarks_for_last_hp: int = 0):
        super().__init__()
        if not shared:
            if use_mixeddimclassifier:
                print('n_ratio must be the sequence length!')
                print('not tested for n_waymarks_for_last_hp')
                self.bridges = nn.ModuleList(
                    [MixedDimensionsClassifier(n_ratio, 4, k, n_ratio, n_hidden) for k in range(n_ratio)]
                )
            elif multiclass:
                self.bridges = MultiClass(in_size, n_ratio + n_waymarks_for_last_hp + 1, n_hidden=n_hidden)
            else: # not multiclass
                self.bridges = nn.ModuleList([model_class(in_size, n_hidden=n_hidden) for _ in range(n_ratio + n_waymarks_for_last_hp)])
        else:
            print('model_class must be shared.')
            print('not tested for n_waymarks_for_last_hp')
            self.bridges = model_class(in_size, n_ratio + n_waymarks_for_last_hp, n_hidden=n_hidden)
        self.bridges.to(device=device, dtype=dtype)

        self.shared = shared
        self.multiclass = multiclass
        self.in_size = in_size
        self.n_ratio = n_ratio
        self.n_waymarks_for_last_hp = n_waymarks_for_last_hp
        self.device = device
        self.dtype = dtype
        self.cranmer = cranmer
        self.num_kd_m = []
        self.denom_kd_m = []

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
            assert(X_m1xnxd.shape[0] - 1 == self.n_ratio)
        else:
            if Xm_nxp is not None:  # X_m1xnxd contains m slices of design data and Xm_nxp is traning data (N != n)
                if X0_nxp is None:
                    print('X_m1xnxd is shape {}, Xm_nxp is shape {}. {} waymarks provided'.format(X_m1xnxd.shape, Xm_nxp.shape, X_m1xnxd.shape[0]))
                    if X_m1xnxd.shape[0] != self.n_ratio:
                        raise ValueError('Number of provided waymarks {} must match n_ratio {}'.format(
                            X_m1xnxd.shape[0], self.n_ratio
                        ))
                    
                    if self.n_waymarks_for_last_hp:
                        Xlast_xnxd = get_mixed_site_waymarks(X_m1xnxd[-1], Xm_nxp, self.n_waymarks_for_last_hp)
                        X_m1xnxd = np.concatenate([X_m1xnxd, Xlast_xnxd], axis=0)
                        assert(X_m1xnxd.shape[0] == self.n_ratio + self.n_waymarks_for_last_hp)
                        print('Shape of waymarks after last HP: {}. Shape of X_m1xnxd: {}'.format(Xlast_xnxd.shape, X_m1xnxd.shape))
                else:
                    raise ValueError('X_m1xnxd, X0_nxp, and Xm_nxp are all provided.')
            else:
                print('X_m1xnxd is shape {}. {} waymarks provided'.format(X_m1xnxd.shape, X_m1xnxd.shape[0] - 1))
                if X_m1xnxd.shape[0] - 1 != self.n_ratio:
                    raise ValueError('Number of provided waymarks {} must match n_ratio {}'.format(
                        X_m1xnxd.shape[0] - 1, self.n_ratio
                    ))
        
        if not self.shared:
            train_dfs = []
            t0 = time()

            if Xm_nxp is not None and X0_nxp is None: 
                X_m1xnxd = list(X_m1xnxd) + [Xm_nxp]

            if self.multiclass:
                train_dfs = self.fit_multiclass(X_m1xnxd, cfg)
            else:
                for i in range(self.n_ratio + self.n_waymarks_for_last_hp):
                    df = self.fit_bridge(self.bridges[i], X_m1xnxd[i], X_m1xnxd[i + 1], cfg)
                    print('Done fitting bridge {} / ({} + {}) ({} s).'.format((i + 1), self.n_ratio, self.n_waymarks_for_last_hp, int(time() - t0)))
                    train_dfs.append(df)
        else:
            if Xm_nxp is not None and X0_nxp is None:
                train_dfs = self.fit_shared_bridges(X_m1xnxd, Xm_nxp, cfg)
            else:
                train_dfs = self.fit_shared_bridges(X_m1xnxd[: -1], X_m1xnxd[-1], cfg)
        
        if self.cranmer:
            tX_m1xnxd = [torch.from_numpy(X_nxd).to(device=self.device, dtype=self.dtype) for X_nxd in X_m1xnxd]

            # k-th column is the LDR between the k-th design distribution and the training distribution,
            # evaluated on the training distribution
            train_ldr_nxm = self._get_ldr_nxm(tX_m1xnxd[-1]).cpu().detach().numpy()
            # k-th column is the LDR between the k-th design distribution and the training distribution,
            # evaluated on the k-th design distribution
            design_ldr_nxm = np.zeros([X_m1xnxd[0].shape[0], self.n_ratio])

            # 2 KDEs for each design distribution
            for k in range(self.n_ratio):
                ldr_nxm = self._get_ldr_nxm(tX_m1xnxd[k]).cpu().detach().numpy()
                if not self.multiclass:
                    # m = 0 corresponds to going to design distribution, so flip
                    ldr_nxm = np.fliplr(np.cumsum(np.fliplr(ldr_nxm), axis=1))

                kd = KernelDensity(bandwidth=cfg['kde_bandwidth'])
                kd.fit(ldr_nxm[:, k][:, None])
                design_ldr_nxm[:, k] = ldr_nxm[:, k]
                self.num_kd_m.append(kd)

                kd = KernelDensity(bandwidth=cfg['kde_bandwidth'])
                kd.fit(train_ldr_nxm[:, k][:, None])
                self.denom_kd_m.append(kd)

        return train_dfs

    def fit_shared_bridges(self, X_mxnxd, Xm_nxd, cfg): 
        assert(X_mxnxd.shape[0] == self.n_ratio + self.n_waymarks_for_last_hp)
        val_frac = cfg['val_frac']
        n_steps = cfg['n_steps']
        lr = cfg['lr']

        tX_mxnxd = torch.from_numpy(X_mxnxd).to(device=self.device, dtype=self.dtype)
        tXm_nxd = torch.from_numpy(Xm_nxd).to(device=self.device, dtype=self.dtype)
        n = tX_mxnxd.shape[1]
        for k in range(self.n_ratio + self.n_waymarks_for_last_hp):
            perm = np.random.permutation(n)
            tX_mxnxd[k] = tX_mxnxd[k, perm]
        tXm_nxd = tXm_nxd[np.random.permutation(tXm_nxd.shape[0])]

        if val_frac > 0:
            n_val_per_n = int(val_frac * n)
            n_tr_per_n = n - n_val_per_n
            Xtr_mxxd = tX_mxnxd[:, n_val_per_n :]
            Xval_mxxd = tX_mxnxd[:, : n_val_per_n]

            z1tr_x1 = torch.ones((n_tr_per_n), 1, device=self.device, dtype=self.dtype)
            z0tr_x1 = torch.zeros((n_tr_per_n), 1, device=self.device, dtype=self.dtype)
            ztr_1x = torch.cat([z1tr_x1, z0tr_x1], dim=0).T

            z1val_x1 = torch.ones((n_val_per_n), 1, device=self.device, dtype=self.dtype)
            z0val_x1 = torch.zeros((n_val_per_n), 1, device=self.device, dtype=self.dtype)
            zval_1x = torch.cat([z1val_x1, z0val_x1], dim=0).T

            n_val_per_ntr = int(val_frac * tXm_nxd.shape[0])
            n_tr_per_ntr = tXm_nxd.shape[0] - n_val_per_ntr
            Xmtr_xd = tXm_nxd[n_val_per_ntr :]
            Xmval_xd = tXm_nxd[: n_val_per_ntr]

            z0trm_x1 = torch.zeros((n_tr_per_ntr), 1, device=self.device, dtype=self.dtype)
            zmtr_1x = torch.cat([z1tr_x1, z0trm_x1], dim=0).T
            z0valm_x1 = torch.zeros((n_val_per_ntr), 1, device=self.device, dtype=self.dtype)
            zmval_1x = torch.cat([z1val_x1, z0valm_x1], dim=0).T

            N_tr = z1tr_x1.shape[0]
            n_tr = z0trm_x1.shape[0]
            if cfg['use_weighted_loss']: 
                wtr_ = torch.cat([
                    ((N_tr + n_tr) / (2 * N_tr)) * torch.ones((N_tr), device=self.device, dtype=self.dtype),
                    ((N_tr + n_tr) / (2 * n_tr)) * torch.ones((n_tr), device=self.device, dtype=self.dtype)
                ])
                N_val = z1val_x1.shape[0]
                n_val = z0valm_x1.shape[0]
                wval_ = torch.cat([
                    ((N_val + n_val) / (2 * N_val)) * torch.ones((N_val), device=self.device, dtype=self.dtype),
                    ((N_val + n_val) / (2 * n_val)) * torch.ones((n_val), device=self.device, dtype=self.dtype)
                ])
            else:
                wtr_ = torch.ones((N_tr + n_tr), 1, device=self.device, dtype=self.dtype)
                wval_ = torch.ones((N_val + n_val), 1, device=self.device, dtype=self.dtype)
        else:
            print('No validation data for bridges.')
            Xtr_mxxd = tX_mxnxd
            Xval_mxxd = None
            z1tr_x1 = torch.ones((n), 1, device=self.device, dtype=self.dtype)
            z0tr_x1 = torch.zeros((n), 1, device=self.device, dtype=self.dtype)
            ztr_1x = torch.cat([z1tr_x1, z0tr_x1], dim=0).T

        loss_fn = torch.nn.BCEWithLogitsLoss()
        wval_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=wval_)
        wtrain_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=wtr_)
        self.bridges.requires_grad_(True)
        optim = torch.optim.Adam(self.bridges.parameters(), lr=lr)

        best_loss = float('inf')
        val_loss = 0
        trloss_m_t = []
        valloss_m_t = []

        for _ in tqdm(range(n_steps)):
            if val_frac > 0:
                with torch.no_grad():
                    vallogits_mx2n, vallogitsm_x1 = self.bridges(Xval_mxxd, Xmval_xd)
                    valloss_m = [loss_fn(val_logits, zval_1x.flatten()) for val_logits in vallogits_mx2n] + [wval_loss_fn(vallogitsm_x1.flatten(), zmval_1x.flatten())]
                    val_loss = sum(valloss_m) / len(valloss_m)

            self.bridges.zero_grad()
            trlogits_mx2n, trlogitsm_x1 = self.bridges(Xtr_mxxd, Xmtr_xd)
            trloss_m = [loss_fn(tr_logits, ztr_1x.flatten()) for tr_logits in trlogits_mx2n] + [wtrain_loss_fn(trlogitsm_x1.flatten(), zmtr_1x.flatten())]
            train_loss = sum(trloss_m) / len(trloss_m)

            if val_frac > 0 and val_loss < best_loss:
                best_loss = val_loss.item()
                best_weights = copy.deepcopy(self.bridges.state_dict())
            elif train_loss < best_loss:
                best_loss = train_loss.item()
                best_weights = copy.deepcopy(self.bridges.state_dict())

            train_loss.backward()
            optim.step()
            trloss_m_t.append([l.cpu().detach().numpy() for l in trloss_m]) 
            if val_frac > 0:
                valloss_m_t.append([l.cpu().detach().numpy() for l in valloss_m])
            else:
                valloss_m_t.append(len(trloss_m) * [0])
            
        self.bridges.load_state_dict(best_weights)
        self.bridges.requires_grad_(False)
        trloss_txm = np.stack(trloss_m_t, axis=0)
        valloss_txm = np.stack(valloss_m_t, axis=0)
        dfs = []
        for trloss_t, valloss_t in zip(trloss_txm.T, valloss_txm.T):
            df = pd.DataFrame(list(zip(trloss_t, valloss_t)), columns=["train_loss", "val_loss"])
            dfs.append(df)
        return dfs

        

    def fit_bridge(self, bridge: nn.Module, X0_Nxd, Xm_nxd, cfg):
        val_frac = cfg['val_frac']
        n_steps = cfg['n_steps']
        lr = cfg['lr']

        tX0_Nxd = torch.from_numpy(X0_Nxd).to(device=self.device, dtype=self.dtype)
        tXm_nxd = torch.from_numpy(Xm_nxd).to(device=self.device, dtype=self.dtype)
        N = tX0_Nxd.shape[0]
        n = tXm_nxd.shape[0]
        perm_N = np.random.permutation(N)
        tX0_Nxd = tX0_Nxd[perm_N]
        perm_n = np.random.permutation(n)
        tXm_nxd = tXm_nxd[perm_n]

        z0_Nx1 = torch.ones((N), 1, device=self.device, dtype=self.dtype)
        zm_nx1 = torch.zeros((n), 1, device=self.device, dtype=self.dtype)

        if val_frac > 0:
            N_val = int(val_frac * N)
            n_val = int(val_frac * n)
            Xtr_xd = torch.cat([tX0_Nxd[N_val :], tXm_nxd[n_val :]])
            ztr_x1 = torch.cat([z0_Nx1[N_val :], zm_nx1[n_val :]])

            N_tr = N - N_val
            n_tr = n - n_val
            assert(N_tr == tX0_Nxd[N_val :].shape[0])
            assert(n_tr == tXm_nxd[n_val :].shape[0])

            Xval_xd = torch.cat([tX0_Nxd[: N_val], tXm_nxd[: n_val]])
            zval_x1 = torch.cat([z0_Nx1[: N_val], zm_nx1[: n_val]])

            if cfg['use_weighted_loss']:
                wtr_x1 = torch.cat([
                    ((N_tr + n_tr) / (2 * N_tr)) * torch.ones((N_tr), 1, device=self.device, dtype=self.dtype),
                    ((N_tr + n_tr) / (2 * n_tr)) * torch.ones((n_tr), 1, device=self.device, dtype=self.dtype)
                ])
                wval_x1 = torch.cat([
                    ((N_val + n_val) / (2 * N_val)) * torch.ones((N_val), 1, device=self.device, dtype=self.dtype),
                    ((N_val + n_val) / (2 * n_val)) * torch.ones((n_val), 1, device=self.device, dtype=self.dtype)
                ])
            else:
                wtr_x1 = torch.ones((N_tr + n_tr), 1, device=self.device, dtype=self.dtype)
                wval_x1 = torch.ones((N_val + n_val), 1, device=self.device, dtype=self.dtype)
            assert(wtr_x1.shape[0] == N_tr + n_tr)
            assert(wval_x1.shape[0] == N_val + n_val)

        else:
            print("No validation data for bridge.")
            N_tr, n_tr = N, n
            N_val, n_val = 0
            Xtr_xd = torch.cat([tX0_Nxd, tXm_nxd])
            ztr_x1 = torch.cat([z0_Nx1, zm_nx1])
            Xval_xd = None
            zval_x1 = None

            if cfg['use_weighted_loss']:
                wtr_x1 = torch.cat([
                    ((N + n) / (2 * N)) * torch.ones((N), 1, device=self.device, dtype=self.dtype),
                    ((N + n) / (2 * n)) * torch.ones((n), 1, device=self.device, dtype=self.dtype)
                ])
            else:
                wtr_x1 = torch.ones((N + n), 1, device=self.device, dtype=self.dtype)
            wval_x1 = None
            assert(wtr_x1.shape[0] == N + n)

        val_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=wval_x1)
        tr_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=wtr_x1)
        bridge.requires_grad_(True)
        optim = torch.optim.Adam(bridge.parameters(), lr=lr)

        best_loss = float('inf')
        val_loss = 0
        ckpt_losses = []

        for _ in tqdm(range(n_steps)):
            if val_frac > 0:
                with torch.no_grad():
                    vallogits = bridge(Xval_xd)
                    val_loss = val_loss_fn(vallogits, zval_x1)

            bridge.zero_grad()
            trlogits = bridge(Xtr_xd)
            train_loss = tr_loss_fn(trlogits, ztr_x1)

            if val_frac > 0 and val_loss < best_loss:
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

    def fit_multiclass(self, X_nxd_m, cfg):
        val_frac = cfg['val_frac']
        n_steps = cfg['n_steps']
        lr = cfg['lr']

        assert(len(X_nxd_m) == self.n_ratio + self.n_waymarks_for_last_hp + 1)
        m = len(X_nxd_m)
        N = X_nxd_m[0].shape[0]
        n = X_nxd_m[-1].shape[0]
        tX_nxd_m = []
        for k in range(len(X_nxd_m)):
            X_nxd = X_nxd_m[k][np.random.permutation(X_nxd_m[k].shape[0])]
            tX_nxd_m.append(torch.from_numpy(X_nxd).to(device=self.device, dtype=self.dtype))

        if val_frac > 0:
            Xtr_nxd_m = [X_nxd[int(val_frac * X_nxd.shape[0]) :] for X_nxd in tX_nxd_m]
            ztr_n_m = [k * torch.ones((Xtr_nxd.shape[0]), device=self.device, dtype=torch.int64) for k, Xtr_nxd in enumerate(Xtr_nxd_m)]
            Xtr_mnxd = torch.cat(Xtr_nxd_m, dim=0)
            ztr_mn = torch.cat(ztr_n_m, dim=0)
            
            Xval_nxd_m = [X_nxd[: int(val_frac * X_nxd.shape[0])] for X_nxd in tX_nxd_m]
            zval_n_m = [k * torch.ones((Xval_nxd.shape[0]), device=self.device, dtype=torch.int64) for k, Xval_nxd in enumerate(Xval_nxd_m)]
            Xval_mnxd = torch.cat(Xval_nxd_m, dim=0)
            zval_mn = torch.cat(zval_n_m, dim=0)

            if cfg['use_weighted_loss']:
                # weights for training data
                N_tr = Xtr_nxd_m[0].shape[0]
                n_tr = Xtr_nxd_m[-1].shape[0]

                w_tr_train = m / ((m - 1) * (n_tr / N_tr) + 1)
                w_tr_design = (n_tr / N_tr) * w_tr_train

                wtr_m = w_tr_design * torch.ones((m), device=self.device, dtype=self.dtype)
                wtr_m[-1] = w_tr_train

                # weights for validation data
                N_val = Xval_nxd_m[0].shape[0]
                n_val = Xval_nxd_m[-1].shape[0]

                w_val_train = m / ((m - 1) * (n_val / N_val) + 1)
                w_val_design = (n_val / N_val) * w_val_train

                wval_m = w_val_design * torch.ones((m), device=self.device, dtype=self.dtype)
                wval_m[-1] = w_val_design
            else:
                wtr_m = wval_m = None

        else:
            raise ValueError('No validation data for multiclass')

        # torch.nn.CrossEntropyLoss(weight=None, reduction='mean', label_smoothing=0.0)
        val_loss_fn = torch.nn.CrossEntropyLoss(weight=wval_m)
        tr_loss_fn = torch.nn.CrossEntropyLoss(weight=wtr_m)
        self.bridges.requires_grad_(True)
        optim = torch.optim.Adam(self.bridges.parameters(), lr=lr)

        best_loss = float('inf')
        val_loss = 0
        ckpt_losses = []

        for _ in tqdm(range(n_steps)):
            if val_frac > 0:
                with torch.no_grad():
                    vallogits = self.bridges(Xval_mnxd)
                    val_loss = val_loss_fn(vallogits, zval_mn)

            self.bridges.zero_grad()
            trlogits = self.bridges(Xtr_mnxd)
            train_loss = tr_loss_fn(trlogits, ztr_mn)

            if val_frac > 0 and val_loss < best_loss:
                best_loss = val_loss.item()
                best_weights = copy.deepcopy(self.bridges.state_dict())
            elif train_loss < best_loss:
                best_loss = train_loss.item()
                best_weights = copy.deepcopy(self.bridges.state_dict())

            train_loss.backward()
            optim.step()
            ckpt_losses.append([
                train_loss.item(),
                val_loss.item() if n_val else val_loss,
            ])

        self.bridges.load_state_dict(best_weights)
        self.bridges.requires_grad_(False)
        df = pd.DataFrame(ckpt_losses, columns=["train_loss", "val_loss"])
        return df


    def _get_ldr_nxm(self, tX_nxp: torch.Tensor):
        if self.shared:
            print('TelescopingLogDensityRatioEstimator._get_ldr_nxm not tested for shared')
            return self.bridges.get_ldr(tX_nxp).T
        if self.multiclass:
            h_nxm = self.bridges(tX_nxp)
            return h_nxm[:, : -1] - h_nxm[:, -1][:, None]
        out_nxm = torch.cat([self.bridges[i](tX_nxp) for i in range(self.n_ratio + self.n_waymarks_for_last_hp)], 1)
        return out_nxm

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
    
    def forecast_meany_per_bridge(self, Xm_nxp, ym_n, cranmer: bool = True, predm_n: np.array = None, pred0_mxn: np.array = None, self_normalized: bool = True):
        tXm_nxp = torch.from_numpy(Xm_nxp).to(device=self.device, dtype=self.dtype)
        ldr_nxm = self._get_ldr_nxm(tXm_nxp).cpu().detach().numpy()
        if not self.multiclass:
            # m = 0 corresponds to going to design distribution, so flip
            ldr_nxm = np.fliplr(np.cumsum(np.fliplr(ldr_nxm), axis=1))
        ldr_nxm = ldr_nxm[:, : self.n_ratio]  # don't care about waymarks between last HP and training distribution
        
        if self.cranmer and cranmer:
            calibrated_ldr_nxm = np.zeros(ldr_nxm.shape)
            for k in range(self.n_ratio):
                logdesign_n = self.num_kd_m[k].score_samples(ldr_nxm[:, k][:, None])
                logtrain_n = self.denom_kd_m[k].score_samples(ldr_nxm[:, k][:, None])
                calibrated_ldr_nxm[:, k] = logdesign_n - logtrain_n
            ldr_nxm = calibrated_ldr_nxm
        
        if self_normalized:
            c_1xm = np.max(ldr_nxm, axis=0, keepdims=True)
            normalization_1xm = c_1xm + np.log(np.sum(np.exp(ldr_nxm - c_1xm), axis=0, keepdims=True))
            normalizeddr_nxm = np.exp(ldr_nxm - normalization_1xm) * ym_n.size  # sum to n
            # dr_nxm = dr_nxm / np.sum(dr_nxm, axis=0, keepdims=True)  # equivalent to normalizeddr_nxm

            if predm_n is None or pred0_mxn is None:
                weightedym_nxm = normalizeddr_nxm * ym_n[:, None]
                forecast_m = np.mean(weightedym_nxm, axis=0, keepdims=False)
            else:
                weightedrectm_nxm = normalizeddr_nxm * (predm_n - ym_n)[:, None]
                forecast_m = np.mean(pred0_mxn, axis=1, keepdims=False) - np.mean(weightedrectm_nxm, axis=0, keepdims=False)
        else:
            normalizeddr_nxm = None
            dr_nxm = np.exp(ldr_nxm)
            print('Not returning normalized weights')
            if predm_n is None or pred0_mxn is None:
                weightedym_nxm = dr_nxm * ym_n[:, None]
                forecast_m = np.mean(weightedym_nxm, axis=0, keepdims=False)
            else:
                weightedrectm_nxm = dr_nxm *  (predm_n - ym_n)[:, None]
                forecast_m = np.mean(pred0_mxn, axis=1, keepdims=False) - np.mean(weightedrectm_nxm, axis=0, keepdims=False)

        return forecast_m, normalizeddr_nxm



