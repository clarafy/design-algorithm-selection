import copy
from time import time

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
            [X0samp_nxd[:, : k * chunk_sz], Xmsamp_nxd[:, k * chunk_sz :]]
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
    def __init__(self, model_class, in_size: int, n_ratio: int = 1, device = None, dtype = None):
        super().__init__()
        self.bridges = nn.ModuleList([model_class(in_size) for _ in range(n_ratio)])
        for b in self.bridges:
            b.to(device=device, dtype=dtype)
        self.in_size = in_size
        self.n_ratio = n_ratio
        self.device = device
        self.dtype = dtype

    def fit(self, X0_nxp: np.array, Xm_nxp: np.array, cfg):
        # generate waymarks
        if cfg['waymark_type'] == 'linear':
            a_m1 = get_alphas(self.n_ratio)
            X_m1xnxd = get_linear_waymarks(X0_nxp, Xm_nxp, a_m1)
        elif cfg['waymark_type'] == 'mixed_dimensions':
            X_m1xnxd = get_mixed_dim_waymarks(X0_nxp, Xm_nxp, self.n_ratio)
        print('Generated {} waymarks.'.format(X_m1xnxd.shape[0] - 1))

        # train each bridge
        train_dfs = []
        t0 = time()
        for i in range(self.n_ratio):  # TODO: parallelize
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

    def forward(self, tX_nxp: torch.Tensor):
        ldre_nxm = torch.cat([self.bridges[i](tX_nxp) for i in range(self.n_ratio)], 1)
        return torch.sum(ldre_nxm, 1, keepdim=True)

    def predict_log_dr(self, X_nxp: np.array):
        tX_nxp = torch.from_numpy(X_nxp).to(device=self.device, dtype=self.dtype)
        ldre_nx1 = self(tX_nxp).cpu().detach().numpy()
        return ldre_nx1.squeeze(-1)

