from time import time
from tqdm import tqdm
import copy
import os.path
from pathlib import Path

import numpy as np
import scipy as sc
from sklearn.linear_model import LogisticRegression, RidgeCV, LinearRegression
import torch
from torch import nn
from torch.utils.data import DataLoader

import utils

# ===== models for predicting enrichment from sequence =====

def type_check_and_one_hot_encode_sequences(seq_n, alphabet, verbose: bool = False):
    # TODO: check sequence lengths 
    if isinstance(seq_n[0], str):
        t0 = time()
        ohe_nxlxa = np.stack([utils.str2onehot(seq, alphabet) for seq in seq_n])
        if verbose:
            print('One-hot encoded sequences to shape = {} ({} sec)'.format(ohe_nxlxa.shape, int(time() - t0)))

    elif type(seq_n[0]) is np.ndarray:
        if len(seq_n.shape) == 3:
            if verbose:
                print('Sequences are already one-hot encoded.')
            # assume seq_n is already shaped like ohe_nxlxa
            ohe_nxlxa = seq_n.copy()
        else:
            raise ValueError('Input should be list of strings or 3D np.array. seq_n has shape {}.'.format(seq_n.shape))
        
    else:
        raise ValueError('Unrecognized seq_n type: {} is type {}'.format(seq_n[0], type(seq_n[0])))
    return ohe_nxlxa

class ExceedancePredictor():
    def __init__(self, model, threshold: float) -> None:
        self.model = model
        self.threshold = threshold
        self.lr = LogisticRegression(class_weight='balanced')
        self.lr_fitted = False

    def fit(self, ohe_nxlxa: np.array, binary_y_n: np.array):
        pred_n = self.model.predict(ohe_nxlxa)
        self.lr.fit(pred_n[:, None], binary_y_n)
        self.lr_fitted = True

    def predict(self, ohe_nxlxa: np.array):
        if not self.lr_fitted:
            # print('Warning: ExceedancePredictor has not been fit. Making predictions by thresholding.')
            return (self.model.predict(ohe_nxlxa) >= self.threshold).astype(float)
        return self.lr.predict_proba(self.model.predict(ohe_nxlxa)[:, None])[:, 1]

# TODO: subclass from TorchRegressorEnsemble
# unfortunately cannot load existing saved EnrichmentFeedForward parameters into a FeedForward model
class EnrichmentFeedForward(torch.nn.Module):  
    def __init__(
        self,
        seq_len: int,
        alphabet: str,
        n_hidden: int = 10,
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
        
        if len(y_nx2.shape) == 1:
            print('No fitness variance estimates provided. Using unweighted MSE loss.')
            y_nx2 = np.hstack([y_nx2[:, None], 0.5 * np.ones([len(seq_n), 1])])
        elif len(y_nx2.shape) == 2 and y_nx2.shape[1] == 1:
            print('No fitness variance estimates provided. Using unweighted MSE loss.')
            y_nx2 = np.hstack([y_nx2, 0.5 * np.ones([len(seq_n), 1])])

        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=True)
        dataset = [(ohe_lxa.flatten(), y_mean_var[0], y_mean_var[1]) for ohe_lxa, y_mean_var in zip(ohe_nxlxa, y_nx2)]

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
                tX_bxlxa, tymean_b, tyvar_b = data
                tX_bxlxa = tX_bxlxa.to(device=self.device, dtype=self.dtype)
                tX_bxla = torch.flatten(tX_bxlxa, start_dim=1)
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
                tX_bxlxa, tymean_b, tyvar_b = data
                tX_bxlxa = tX_bxlxa.to(device=self.device, dtype=self.dtype)
                tX_bxla = torch.flatten(tX_bxlxa, start_dim=1)
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
        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=verbose)
        tohe_nxlxa = torch.from_numpy(ohe_nxlxa).to(device=self.device, dtype=self.dtype)
        tohe_nxla = torch.flatten(tohe_nxlxa, start_dim=1)
        return self(tohe_nxla).cpu().detach().numpy()
    
    def save(self, save_fname_no_ftype, save_path: str = '/homefs/home/wongfanc/density-ratio-estimation/gb1-models'):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(save_path, save_fname_no_ftype + '.pt')
        torch.save(self.state_dict(), fname)
        print('Saved models to {}.'.format(fname))
    
    def load(self, save_fname_no_ftype, save_path: str = '/homefs/home/wongfanc/density-ratio-estimation/gb1-models'):
        fname = os.path.join(save_path, save_fname_no_ftype + '.pt')
        self.load_state_dict(torch.load(fname))


class TorchRegressorEnsemble(torch.nn.Module):
    def __init__(
        self,
        models,
        seq_len: int,
        alphabet: str,
        device = None,
        dtype = torch.float
    ):
        super().__init__()
        
        self.models = models
        self.models.to(device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        self.seq_len = seq_len
        self.alphabet = alphabet
        self.input_sz = seq_len * len(alphabet)

    def forward(self, tX_nxlxa):
        pred_nxm = torch.cat([model(tX_nxlxa) for model in self.models], dim=1)
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
        
        if len(y_nx2.shape) == 1:
            print('No fitness variance estimates provided. Using unweighted MSE loss.')
            y_nx2 = np.hstack([y_nx2[:, None], 0.5 * np.ones([len(seq_n), 1])])
        elif len(y_nx2.shape) == 2 and y_nx2.shape[1] == 1:
            print('No fitness variance estimates provided. Using unweighted MSE loss.')
            y_nx2 = np.hstack([y_nx2, 0.5 * np.ones([len(seq_n), 1])])

        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=True)
        dataset = [(ohe_lxa, y_mean_var[0], np.fmin(y_mean_var[1], 1e6)) for ohe_lxa, y_mean_var in zip(ohe_nxlxa, y_nx2)]

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
                tX_bxlxa, tymean_b, tyvar_b = data
                tX_bxlxa = tX_bxlxa.to(device=self.device, dtype=self.dtype)
                tymean_b = tymean_b.to(device=self.device, dtype=self.dtype)
                tyvar_b = tyvar_b.to(device=self.device, dtype=self.dtype)

                pred_b = self(tX_bxlxa)
                loss = self.weighted_mse_loss(tymean_b, pred_b, 1 / (2 * tyvar_b))
                total_val_loss += loss.item() * tX_bxlxa.shape[0]
            total_val_loss /= n_val

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_parameters = copy.deepcopy(self.state_dict())

            # gradient step on training loss
            self.requires_grad_(True)
            total_train_loss = 0.
            for _, data in enumerate(tqdm(train_loader)):
                tX_bxlxa, tymean_b, tyvar_b = data
                tX_bxlxa = tX_bxlxa.to(device=self.device, dtype=self.dtype)
                tymean_b = tymean_b.to(device=self.device, dtype=self.dtype)
                tyvar_b = tyvar_b.to(device=self.device, dtype=self.dtype)

                optimizer.zero_grad()

                pred_b = self(tX_bxlxa)

                loss = self.weighted_mse_loss(tymean_b, pred_b, 1 / (2 * tyvar_b))
                loss.backward()

                optimizer.step()
                total_train_loss += loss.item() * tX_bxlxa.shape[0]

            total_train_loss /= n_train
            loss_tx2[t] = total_train_loss, total_val_loss
            print('Epoch {}. Train loss: {:.2f}. Val loss: {:.2f}. {} sec.'.format(t, total_train_loss, total_val_loss, int(time() - t0)))
        
        self.load_state_dict(best_model_parameters)
        self.requires_grad_(False)
        return loss_tx2
    
    def predict(self, seq_n, verbose: bool = False):
        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=verbose)
        tohe_nxlxa = torch.from_numpy(ohe_nxlxa).to(device=self.device, dtype=self.dtype)
        return self(tohe_nxlxa).cpu().detach().numpy()
    
    def predict_prob_exceedance(self, seq_n, threshold: float, verbose: bool = False):
        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=verbose)
        tohe_nxlxa = torch.from_numpy(ohe_nxlxa).to(device=self.device, dtype=self.dtype)
        pred_nxm = torch.cat([model(tohe_nxlxa) for model in self.models], dim=1).detach().cpu().numpy()
        pred_n = np.mean(pred_nxm, axis=1)
        var_n = np.var(pred_nxm, axis=1)
        pexceed_n = sc.stats.norm.sf(threshold, loc=pred_n, scale=np.sqrt(var_n))
        return pexceed_n
    
    def save(self, save_fname):
        torch.save(self.state_dict(), save_fname)
        print('Saved models to {}.'.format(save_fname))
    
    def load(self, save_fname):
        self.load_state_dict(torch.load(save_fname))


class FeedForward(TorchRegressorEnsemble):
    def __init__(
        self,
        seq_len: int,
        alphabet: str,
        n_hidden: int,
        n_model: int = 3,
        device = None,
        dtype = torch.float
    ):
        models = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * len(alphabet), n_hidden),
                nn.ReLU(), 
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
            )
        for _ in range(n_model)])
        super().__init__(models, seq_len, alphabet, device=device, dtype=dtype)

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X_nxlxa):
        return torch.transpose(X_nxlxa, 1, 2)

class CNN(TorchRegressorEnsemble):
    def __init__(
        self,
        seq_len: int,
        alphabet: str,
        n_filters: int,
        n_hidden: int,
        pool_sz: int = 1,
        kernel_sz: int = 5,
        n_model: int = 3,
        device = None,
        dtype = torch.float
    ):
        models = nn.ModuleList([
            nn.Sequential(
                Transpose(),
                nn.Conv1d(len(alphabet), n_filters, kernel_sz),
                nn.ReLU(), 
                nn.Conv1d(n_filters, n_filters, kernel_sz), 
                nn.ReLU(),
                nn.MaxPool1d(pool_sz),
                nn.Conv1d(n_filters, n_filters, kernel_sz),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(n_filters * (int((seq_len - 2 * (kernel_sz - 1)) / pool_sz) - (kernel_sz - 1)), n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1)
            )
        for _ in range(n_model)])
        super().__init__(models, seq_len, alphabet, device=device, dtype=dtype)


class SklearnRegressor():
    def __init__(
        self,
        model,
        seq_len: int,
        alphabet: str
    ):
        self.model = model
        self.seq_len = seq_len
        self.alphabet = alphabet
        self.mse = None
        self.model_fitted = False

    def fit(self, seq_n, y_n, weight_n: np.array = None, val_frac: float = 0.1, verbose: bool = False):
        if weight_n is None:
            weight_n = np.ones([y_n.size])
            
        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=verbose)
        ohe_nxla = np.reshape(ohe_nxlxa, [len(seq_n), ohe_nxlxa.shape[1] * ohe_nxlxa.shape[2]])

        shuffle_idx = np.random.permutation(len(seq_n))
        n_val = int(val_frac * len(seq_n))
        n_train = len(seq_n) - n_val
        train_idx, val_idx = shuffle_idx[: n_train], shuffle_idx[n_train :]

        self.model.fit(ohe_nxla[train_idx], y_n[train_idx], sample_weight=weight_n[train_idx])
        predval_n = self.model.predict(ohe_nxla[val_idx])
        self.mse = np.mean(weight_n[val_idx] * np.square(y_n[val_idx] - predval_n))
        self.model_fitted = True

    def predict(self, seq_n, verbose: bool = False):
        if not self.model_fitted:
            raise ValueError(
                'This SklearnRegressor instance is not fitted yet. Call `fit` with appropriate arguments.'
            )
        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=verbose)
        ohe_nxla = np.reshape(ohe_nxlxa, [len(seq_n), ohe_nxlxa.shape[1] * ohe_nxlxa.shape[2]])
        return self.model.predict(ohe_nxla)
    
    def predict_prob_exceedance(self, seq_n, threshold: float, verbose: bool = False):
        if not self.model_fitted:
            raise ValueError(
                'This SklearnRegressor instance is not fitted yet. Call `fit` with appropriate arguments.'
            )
        pred_n = self.predict(seq_n, verbose=verbose)
        pexceed_n = sc.stats.norm.sf(threshold, loc=pred_n, scale=np.sqrt(self.mse))
        return pexceed_n

class RidgeRegressor(SklearnRegressor):
    def __init__(self, seq_len: int, alphabet: str, alphas = None):
        if alphas is None:
            alphas = np.logspace(-5, 5, 11)
        model = RidgeCV(alphas=alphas)
        super().__init__(model, seq_len, alphabet)


class LinearRegressor(SklearnRegressor):
    def __init__(self, seq_len: int, alphabet: str):
        model = LinearRegression()
        super().__init__(model, seq_len, alphabet)
    


     