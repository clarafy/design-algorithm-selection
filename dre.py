from abc import ABC, abstractmethod
import copy

import numpy as np
import pandas as pd

import torch
from torch import nn


class DensityRatioEstimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, Xtrain_nxp: np.array, ytrain_n: np.array, Xtest_mxp: np.array, ytest_m: np.array):
        pass

    @abstractmethods
    def predict(self, Xnew_nxp):
        pass


def fit_dr_estimator(cfg, dr_estimator, Xsource_xp, Xtarget_xp, val_frac: float = 0):
    tXsource_xp = torch.Tensor(Xsource_xp)
    tXtarget_xp = torch.Tensor(Xtarget_xp)
    n_source = tXsource_xp.shape[0]
    n_target = tXtarget_xp.shape[0]

    zsource = torch.zeros((n_source), 1, device=dr_estimator.device)
    ztarget = torch.ones((n_target), 1, device=dr_estimator.device)
    cat_X = torch.cat([tXsource_xp, tXtarget_xp])
    cat_Z = torch.cat([zsource, ztarget])

    if val_frac:
        n_val = int(0.1 * cat_X.shape[-2])
        rand_perm = np.random.permutation(cat_X.shape[-2])
        Xval_xp = cat_X[rand_perm[:n_val]]
        Zval_x1 = cat_Z[rand_perm[:n_val]]
        Xtrain_nxp = cat_X[rand_perm[n_val:]]
        Ztrain_nx1 = cat_Z[rand_perm[n_val:]]
    else:
        n_val = 0
        Xtrain_nxp = cat_X
        Ztrain_nx1 = cat_Z
        Xval_xp = None
        Zval_x1 = None

    emp_ratio = n_source / n_target  # TODO: estimate on all data including validation?
    dr_estimator._ratio = emp_ratio  # TODO: this whole function should be class method
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([emp_ratio]))  # TODO: proper score?
    dr_estimator.requires_grad_(True)
    optimizer = torch.optim.Adam(
        dr_estimator.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_grad_steps'])

    best_loss = float('inf')
    losses = []
    val_loss = 0

    for _ in range(cfg['num_grad_steps']):
        if n_val:
            with torch.no_grad():
                val_logits = dr_estimator.classifier(Xval_xp)
                val_loss = loss_fn(val_logits, Zval_x1)

        dr_estimator.zero_grad()
        Xaug_nxp = Xtrain_nxp + cfg['noise_aug_scale'] * torch.randn_like(Xtrain_nxp)
        train_logits = dr_estimator.classifier(Xaug_nxp)
        train_loss = loss_fn(train_logits, Ztrain_nx1)

        if n_val and val_loss < best_loss:
            best_loss = val_loss.item()
            best_weights = copy.deepcopy(dr_estimator.state_dict())
            ckpt_train_loss = train_loss.item()
        elif train_loss < best_loss:
            best_loss = train_loss.item()
            best_weights = copy.deepcopy(dr_estimator.state_dict())
            ckpt_train_loss = train_loss.item()

        train_loss.backward()
        optimizer.step()
        lr_sched.step()
        losses.append([ckpt_train_loss, val_loss])

    dr_estimator.load_state_dict(best_weights)
    dr_estimator.requires_grad_(False)
    dr_estimator.update_target_network()

    metrics = dict(
        dre_best_loss=best_loss,
        dre_last_train_loss=train_loss.item()
    )
    df = pd.DataFrame(losses, columns=["train_loss", "val_loss"])

    if n_val:
        tgt_network = dr_estimator._target_network
        tgt_logits = tgt_network(Xval_xp)
        metrics['dre_tgt_loss'] = loss_fn(tgt_logits, Zval_x1).item()
    return metrics, df


class Quadratic(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn((in_size, in_size)))
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return torch.matmul(torch.matmul(x.T, self.W), x) + self.b



class RatioEstimator(nn.Module):

    def __init__(self, in_size, n_hidden: int = 16, device=None, dtype=None, ema_weight=1e-2, lr=1e-3, weight_decay=1e-4):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.in_size = in_size

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_size, n_hidden),
        #     nn.ReLU(),
        #     nn.Linear(n_hidden, n_hidden),
        #     nn.ReLU(),
        #     nn.Linear(n_hidden, 1),
        # ).to(device=device, dtype=dtype)

        self.classifier = nn.Sequential(
            Quadratic(in_size),
        ).to(device=device, dtype=dtype)

        # density ratio estimates are exactly 1 when untrained
        self._target_network = copy.deepcopy(self.classifier)
        self._target_network.requires_grad_(False)
        for tgt_p in self._target_network.parameters():
            tgt_p.data.fill_(0.)
        self._ema_weight = ema_weight
        self._ratio = None

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)

    def forward2(self, inputs):
        _p = torch.exp(self._target_network(inputs).squeeze(-1))
        return _p

    def forward(self, inputs):
        _p = self._target_network(inputs).squeeze(-1).sigmoid()
        return self._ratio * _p.clamp_max(1 - 1e-6) / (1 - _p).clamp_min(1e-6)

    def update_target_network(self):
        with torch.no_grad():
            for src_p, tgt_p in zip(
                    self.classifier.parameters(), self._target_network.parameters()
            ):
                tgt_p.mul_(1. - self._ema_weight)
                tgt_p.add_(self._ema_weight * src_p)

    def optimize_callback(self, xk):
        if isinstance(xk, np.ndarray):
            xk = torch.from_numpy(xk)
        xk = xk.reshape(-1, self.in_size)
        self._pos_samples.extend([x for x in xk])

        if self._neg_samples is None:
            return None

        num_negative = self._neg_samples.size(0)
        num_positive = len(self._pos_samples)

        if num_positive < num_negative:
            return None

        self.classifier.requires_grad_(True)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        neg_minibatch = self._neg_samples.to(device=self.device, dtype=self.dtype)
        pos_minibatch = torch.stack([
            self._pos_samples[idx] for idx in np.random.permutation(num_positive)[:num_negative]
        ]).to(device=self.device, dtype=self.dtype)
        minibatch_X = torch.cat([neg_minibatch, pos_minibatch])
        minibatch_Z = torch.cat(
            [torch.zeros(num_negative, 1), torch.ones(num_negative, 1)]
        ).to(device=self.device, dtype=self.dtype)

        self.optim.zero_grad()
        loss = loss_fn(self.classifier(minibatch_X), minibatch_Z)
        loss.backward()
        self.optim.step()
        self.update_target_network()

        self.classifier.eval()
        self.classifier.requires_grad_(False)