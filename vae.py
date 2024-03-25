from time import time
import copy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from pandas import DataFrame
import numpy as np

from models import type_check_and_one_hot_encode_sequences
from utils import RNA_NUCLEOTIDES

class VAE(nn.Module):
    def __init__(
            self,
            seq_len: int = 50,
            alphabet: str = RNA_NUCLEOTIDES,
            latent_dim: int = 10,
            n_hidden: int = 20,
            device = None,
            dtype = torch.float
        ):
        super().__init__()

        self.seq_len = seq_len
        self.alphabet = alphabet
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.seq_len * len(self.alphabet), n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        # latent mean and variance 
        self.mean_layer = nn.Linear(n_hidden, latent_dim)
        self.logvar_layer = nn.Linear(n_hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.seq_len * len(self.alphabet)),
            nn.Unflatten(dim=1, unflattened_size=(self.seq_len, len(self.alphabet))),
        )
        self.encoder.to(device)
        self.mean_layer.to(device)
        self.logvar_layer.to(device)
        self.decoder.to(device)
        self.device = device
        self.dtype = dtype 

    def encode(self, X_nxlxa):
        h_nxh = self.encoder(X_nxlxa)
        mu_nxd, logvar_nxd = self.mean_layer(h_nxh), self.logvar_layer(h_nxh)
        return mu_nxd, logvar_nxd

    def reparameterization(self, mu_nxd, logvar_nxd):
        eps_nxd = torch.randn_like(logvar_nxd).to(self.device)      
        z_nxd = mu_nxd + torch.exp(0.5 * logvar_nxd) * eps_nxd
        return z_nxd

    def decode(self, z_nxd):
        return self.decoder(z_nxd)

    def forward(self, X_nxlxa):
        mu_nxd, logvar_nxd = self.encode(X_nxlxa)
        z_nxd = self.reparameterization(mu_nxd, logvar_nxd)
        logits_nxlxa = self.decode(z_nxd)
        return logits_nxlxa, mu_nxd, logvar_nxd
    
    def kl_loss(self, mu_nxd, logvar_nxd, weight_n):
        kl_n =  -0.5 * torch.sum(1 + logvar_nxd - torch.pow(mu_nxd, 2) - torch.exp(logvar_nxd), dim=1)
        return torch.sum(weight_n * kl_n)
    
    def decode_probabilities(self, z_nxd: np.array):
        tz_nxd = torch.from_numpy(z_nxd).to(self.device, self.dtype)
        logits_nxlxa =  self.decoder(tz_nxd)
        p_nxlxa = torch.softmax(logits_nxlxa, dim=2)
        p_nxlxa = p_nxlxa.detach().cpu().numpy()
        return p_nxlxa
    
    def generate(self, n_sample):
        z_nxd = np.random.randn(n_sample, self.latent_dim)
        p_nxlxa = self.decode_probabilities(z_nxd)

        if (
            np.isnan(p_nxlxa).any()
            or np.isinf(p_nxlxa).any()
        ):
            raise ValueError('NaN and/or inf in the reconstructed matrix.')

        seq_n = []
        for i in range(n_sample):
            seq = []
            for site in range(self.seq_len):
                seq.extend(random.choices(self.alphabet, p_nxlxa[i, site, :]))
            seq = ''.join(seq)
            seq_n.append(seq)

        return seq_n, p_nxlxa, z_nxd


    def fit(
        self,
        seq_n,
        weight_n = None,
        batch_size: int = 10,
        lr: float = 0.001,
        n_epoch: int = 10,
        val_frac: float = 0.1,
        n_data_workers: int = 1,
        verbose: bool = False
    ):
        if weight_n is None:
            weight_n = np.ones([len(seq_n)])
        assert(weight_n.size == len(seq_n))
        weight_n = weight_n * weight_n.size / np.sum(weight_n)
        ohe_nxlxa = type_check_and_one_hot_encode_sequences(seq_n, self.alphabet, verbose=verbose)
        dataset = [(ohe_lxa, w) for ohe_lxa, w in  zip(ohe_nxlxa, weight_n)]

        # split into training and validation
        shuffle_idx = np.random.permutation(len(dataset))
        n_val = int(val_frac * len(dataset))
        n_train = len(dataset) - n_val
        train_dataset = [dataset[i] for i in shuffle_idx[: n_train]]
        val_dataset = [dataset[i] for i in shuffle_idx[n_train :]]
        assert(len(val_dataset) == n_val)
        if verbose:
            print('{} training data points, {} validation data points.'.format(n_train, n_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_data_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_data_workers)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  
        loss_df_columns = []
        best_val_loss = np.inf
        best_model_parameters = None
        print_format_str = 'Epoch {}. Train loss: {:.4f}, KL: {:.4f}, CE: {:.4f}. Val loss: {:.4f}, KL: {:.4f}, CE: {:.4f}. ({} s)'
        for t in range(n_epoch):
            
            t0 = time()

            # validation loss
            self.requires_grad_(False)
            total_val_loss = 0.
            total_val_kl_loss = 0.
            total_val_recon_loss = 0.
            for _, data in enumerate(val_loader):
                X_bxlxa, w_b = data
                X_bxlxa = X_bxlxa.to(device=self.device, dtype=self.dtype)
                w_b = w_b.to(device=self.device, dtype=self.dtype)
                
                logits_bxlxa, mu_bxd, logvar_bxd = self(X_bxlxa)

                logits_bxaxl = torch.transpose(logits_bxlxa, dim0=1, dim1=2)
                assert(logits_bxaxl.shape[1 :] == (len(self.alphabet), self.seq_len))
                idx_bxl = torch.argmax(X_bxlxa, axis=2)
                reconstruction_loss_bxl = nn.functional.cross_entropy(logits_bxaxl, idx_bxl, reduction='none')
                assert(reconstruction_loss_bxl.shape[1] == self.seq_len)
                reconstruction_loss = torch.sum(w_b * torch.sum(reconstruction_loss_bxl, dim=1))

                kl_loss = self.kl_loss(mu_bxd, logvar_bxd, w_b)
                total_val_kl_loss += kl_loss.item()
                total_val_recon_loss += reconstruction_loss.item()
                total_val_loss += kl_loss.item() + reconstruction_loss.item()
            total_val_kl_loss /= n_val
            total_val_recon_loss /= n_val
            total_val_loss /= n_val

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_parameters = copy.deepcopy(self.state_dict())

            # gradient step on training loss
            self.requires_grad_(True)
            total_train_loss = 0.
            total_train_kl_loss = 0.
            total_train_recon_loss = 0.
            for _, data in enumerate(train_loader):
                X_bxlxa, w_b = data
                X_bxlxa = X_bxlxa.to(device=self.device, dtype=self.dtype)
                w_b = w_b.to(device=self.device, dtype=self.dtype)
                
                optimizer.zero_grad()

                logits_bxlxa, mu_bxd, logvar_bxd = self(X_bxlxa)

                logits_bxaxl = torch.transpose(logits_bxlxa, dim0=1, dim1=2)
                idx_bxl = torch.argmax(X_bxlxa, axis=2)
                reconstruction_loss_bxl = nn.functional.cross_entropy(logits_bxaxl, idx_bxl, reduction='none')
                reconstruction_loss = torch.sum(w_b * torch.sum(reconstruction_loss_bxl, dim=1))

                kl_loss = self.kl_loss(mu_bxd, logvar_bxd, w_b)
                loss = (kl_loss + reconstruction_loss) / X_bxlxa.shape[0]

                loss.backward()
                optimizer.step()

                total_train_kl_loss += kl_loss.item()
                total_train_recon_loss += reconstruction_loss.item()
                total_train_loss += kl_loss.item() + reconstruction_loss.item()

            total_train_kl_loss /= n_train
            total_train_recon_loss /= n_train
            total_train_loss /= n_train
            loss_df_columns.append(
                [total_train_loss, total_train_kl_loss, total_train_recon_loss, total_val_loss, total_val_kl_loss, total_val_recon_loss]
            )
            if verbose:
                print(
                    print_format_str.format(
                        t, total_train_loss, total_train_kl_loss, total_train_recon_loss,
                        total_val_loss, total_val_kl_loss, total_val_recon_loss, 
                        int(time() - t0)
                    )
                )
        
        self.load_state_dict(best_model_parameters)
        self.requires_grad_(False)
        loss_df = DataFrame(loss_df_columns, columns=['train_loss', 'train_kl', 'train_recon', 'val_loss', 'val_kl', 'val_recon'])
        return loss_df