from abc import ABC, abstractmethod
from time import time
from pathlib import Path
from parse import parse

import numpy as np
import scipy as sc
import pandas as pd

import flexs
import flexs.utils.sequence_utils as sutils

from shifts import DistributionShift, get_mutant

class FLEXSShift(object):
    def __init__(self, landscape_name, noise_sd: float = 0.0):
        if 'RNA' in landscape_name:
            self.problem = flexs.landscapes.rna.registry()[landscape_name]
            self.landscape = flexs.landscapes.RNABinding(**self.problem['params'])
            self.alphabet = sutils.RNAA
            self.noise_sd = noise_sd
        else:
            raise ValueError('Unrecognized landscape_name: {}'.format(landscape_name))
        # super().__init__(self.landscape.seq_length * len(self.alphabet))


    def get_data(self, n: int, model_class, explorer_class, explorer_kwarg_name2vals, model_kwargs = None,
                 save_fname_prefix: str = None, seed_idx: int = 1, avg_n_mut: int = 3):
        if model_kwargs is None:
            model_kwargs = {}
        if model_class == flexs.baselines.models.LinearRegression or model_class == flexs.baselines.models.RidgeCV:
            model = model_class(alphabet=self.alphabet, **model_kwargs)
        else:
            model = model_class(self.landscape.seq_length, alphabet=self.alphabet, **model_kwargs)

        seed = self.problem['starts'][seed_idx]
        yseed = self.landscape.get_fitness([seed])[0]

        # generate random mutants around WT
        p_mut = avg_n_mut / self.landscape.seq_length
        trainseqs_n = [get_mutant(seed, p_mut, self.alphabet) for _ in range(n)]
        ytrain_n = self.landscape.get_fitness(trainseqs_n)
        noise = sc.stats.norm.rvs(loc=0, scale=self.noise_sd, size=len(trainseqs_n)) # TODO HERE: clean up FLEXS
        ytrain_n = ytrain_n + noise
        calseqs_n = [get_mutant(seed, p_mut, self.alphabet) for _ in range(n)]
        ycal_n = self.landscape.get_fitness(calseqs_n)
        noise = sc.stats.norm.rvs(loc=0, scale=self.noise_sd, size=len(calseqs_n))
        ycal_n = ycal_n + noise
        train_data = pd.DataFrame(
            {
                "sequence": trainseqs_n,
                "model_score": np.nan,
                "true_score": ytrain_n,
                "round": 0,
                "model_cost": model.cost,
                "measurement_cost": 1,
            }
        )

        testseqs_list = []
        for name, vals in explorer_kwarg_name2vals.items():
            for val in vals:
                self.explorer = explorer_class(
                    model,
                    **{name: val},
                    starting_sequence=seed,
                    rounds=1,
                    sequences_batch_size=n,
                    model_queries_per_batch=4 * n,  # TODO: ensure get n designs
                    alphabet=self.alphabet
                )

                # train predictive model on data
                self.explorer.model.train(trainseqs_n, ytrain_n)
                predcal_n = self.explorer.model.get_fitness(calseqs_n)

                # run design algorithm
                testseqs_n, predtest_n = self.explorer.propose_sequences(train_data)
                ytest_n = self.landscape.get_fitness(testseqs_n)
                noise = sc.stats.norm.rvs(loc=0, scale=self.noise_sd, size=len(testseqs_n)) # TODO HERE: clean up FLEXS
                ytest_n = ytest_n + noise
                testseqs_list.append((testseqs_n, ytest_n, predtest_n))
        
                if save_fname_prefix is not None:
                    
                    fname = '{}-seed{}-n{}-{}{}.npz'.format(
                        save_fname_prefix, seed_idx, n, name, val
                    )
                    np.savez(
                        fname,
                        yseed=yseed,
                        trainseqs_n=trainseqs_n,
                        ytrain_n=ytrain_n,
                        calseqs_n=calseqs_n,
                        ycal_n=ycal_n,
                        predcal_n=predcal_n,
                        testseqs_n=testseqs_n,
                        ytest_n=ytest_n,
                        predtest_n=predtest_n
                    )
                    print('Saved training, calibration, and design data to {}'.format(fname))

        return trainseqs_n, ytrain_n, calseqs_n, ycal_n, predcal_n, testseqs_list, yseed

    def get_log_dr(self, X_nxd: np.array):
        pass


def generate_rna_data(model_class, explorer_class, explorer_kwarg_name2vals, save_fname_dir: str, noise_sd: float = .0,
                      model_kwargs = None, seq_len: int = 50, landscape_names = None, ns = None, avg_n_mut: int = 3):

    if landscape_names is None:
        landscape_names = flexs.landscapes.rna.registry().keys()
        landscape_names = [name for name in landscape_names if 'L{}'.format(seq_len) in name and '+' not in name]
    print('Generating data from the following landscapes:')
    print(landscape_names)

    if ns is None:
        ns = [1000, 10000]
    print('with the following amounts of training data:')
    print(ns)

    for landscape_name in landscape_names:

        save_fname_prefix = '/data/wongfanc/dre-data/data/{}/{}'.format(save_fname_dir, landscape_name)
        Path(save_fname_prefix).mkdir(parents=True, exist_ok=True)

        flexsshift = FLEXSShift(landscape_name, noise_sd=noise_sd)

        for n in ns:
            
            for seed_idx in flexsshift.problem['starts'].keys():

                t0 = time() 

                _ = flexsshift.get_data(
                    n,
                    model_class,
                    explorer_class,
                    explorer_kwarg_name2vals,
                    model_kwargs=model_kwargs,
                    seed_idx=seed_idx,
                    avg_n_mut=avg_n_mut,
                    save_fname_prefix=save_fname_prefix
                )    
                print('Generated and saved data for {}, n = {}, seed {} ({} s).'.format(
                    landscape_name, n, seed_idx, int(time() - t0)
                ))

def load_rna_data(landscape_name: str, seed_idx: int, n: int, explorer_kwarg_name2vals, save_fname_dir: str):
    
    assert(len(explorer_kwarg_name2vals.keys()) == 1)
    hp_name = list(explorer_kwarg_name2vals.keys())[0]
    hp_vals = explorer_kwarg_name2vals[hp_name]
    m = len(hp_vals)  # number of bridge ratios

    seq_length = int(parse('L{}_RNA{}', landscape_name)[0])
    print('Problem has sequence length {}'.format(seq_length))
    d = seq_length * len(sutils.RNAA)
    X_m1xnxd = np.zeros([m + 1, n, d])
    y_m1xn = np.zeros([m + 1, n])
    pred_mxn = np.zeros([m, n])  # slice i corresponds to i in X_m1xnxd. no predictions on training data

    Xcal_mxnxd = np.zeros([m, n, d])  # slice i corresponds to training data going from i + 1 to i in X_m1xnxd
    ycal_mxn = np.zeros([m, n])
    predcal_mxn = np.zeros([m, n])  # slice i corresponds to i in Xcal_mxnxd

    save_fname_prefix = '{}/{}'.format(save_fname_dir, landscape_name)
    print('Loading waymarks in the following order for k = 0, 1, ..., m where k = 0 is the target design distribution.')
    print(hp_vals)
    for k, val in enumerate(hp_vals):

        fname = '{}-seed{}-n{}-{}{}.npz'.format(save_fname_prefix, seed_idx, n, hp_name, val)
        loaded_dict = np.load(fname)

        # design data
        testseqs_n = loaded_dict['testseqs_n']
        Xk_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in testseqs_n])
        X_m1xnxd[k] = Xk_nxd
        y_m1xn[k] = loaded_dict['ytest_n']
        pred_mxn[k] = loaded_dict['predtest_n']  # no predictions on training data

        # calibration data
        calseqs_n = loaded_dict['calseqs_n']
        Xk1cal_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in calseqs_n])
        Xcal_mxnxd[k] = Xk1cal_nxd
        ycal_mxn[k] = loaded_dict['ycal_n']
        predcal_mxn[k] = loaded_dict['predcal_n']

        # print('After loading k = {}:'.format(k))
        # for i in range(m):  # sanity
        #     print(
        #         np.sum(X_m1xnxd[i]),
        #         np.sum(y_m1xn[i]),
        #         np.sum(pred_mxn[i]),
        #         np.sum(Xcal_mxnxd[i]),
        #         np.sum(ycal_mxn[i]),
        #         np.sum(predcal_mxn[i]),
        #     )
        # print(
        #         np.sum(X_m1xnxd[-1]),
        #         np.sum(y_m1xn[-1]),
        #     )


    # training sequences. could load from any hp_val, all the same training sequences
    fname = '{}-seed{}-n{}-{}{}.npz'.format(save_fname_prefix, seed_idx, n, hp_name, hp_vals[0])
    loaded_dict = np.load(fname)
    trainseqs_n = loaded_dict['trainseqs_n']
    Xm_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in trainseqs_n])
    X_m1xnxd[-1] = Xm_nxd
    y_m1xn[-1] = loaded_dict['ytrain_n']

    # print('After loading training data:')
    # for i in range(m):  # sanity
    #     print(
    #         np.sum(X_m1xnxd[i]),
    #         np.sum(y_m1xn[i]),
    #         np.sum(pred_mxn[i]),
    #         np.sum(Xcal_mxnxd[i]),
    #         np.sum(ycal_mxn[i]),
    #         np.sum(predcal_mxn[i]),
    #     )
    # print(
    #     np.sum(X_m1xnxd[-1]),
    #     np.sum(y_m1xn[-1]),
    # )

    return X_m1xnxd, y_m1xn, pred_mxn, Xcal_mxnxd, ycal_mxn, predcal_mxn