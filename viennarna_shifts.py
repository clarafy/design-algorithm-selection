from abc import ABC, abstractmethod
from time import time
from pathlib import Path
import os.path

from parse import parse

import numpy as np
import scipy as sc
import pandas as pd

import flexs
import flexs.utils.sequence_utils as sutils
from flexs.baselines.explorers import CbAS

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


    def get_data(self, n: int, model_class, explorer: str, explorer_kwarg_name2vals, model_kwargs = None,
                 N: int = None, save_fname_prefix: str = None, seed_idx: int = 1, n_cal: int = 1000, avg_n_mut: int = 3):
        if model_kwargs is None:
            model_kwargs = {}
        if model_class == flexs.baselines.models.LinearRegression or model_class == flexs.baselines.models.RidgeCV:
            model = model_class(alphabet=self.alphabet, **model_kwargs)
        else:
            model = model_class(self.landscape.seq_length, alphabet=self.alphabet, **model_kwargs)
        if N is None:
            N = n

        seed = self.problem['starts'][seed_idx]
        yseed = self.landscape.get_fitness([seed])[0]

        # generate random mutants around WT
        p_mut = avg_n_mut / self.landscape.seq_length
        trainseqs_n = [get_mutant(seed, p_mut, self.alphabet) for _ in range(n)]
        ytrain_n = self.landscape.get_fitness(trainseqs_n)
        noise = sc.stats.norm.rvs(loc=0, scale=self.noise_sd, size=len(trainseqs_n)) # TODO: clean up FLEXS
        ytrain_n = ytrain_n + noise

        trainseqs_n, calseqs_n = trainseqs_n[: n - n_cal], trainseqs_n[n - n_cal :]
        ytrain_n, ycal_n = ytrain_n[: n - n_cal], ytrain_n[n - n_cal :]
        assert(ytrain_n.size + ycal_n.size == n)        

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
                if explorer == 'adalead':
                    self.explorer = flexs.baselines.explorers.Adalead(
                        model,
                        **{name: val},
                        starting_sequence=seed,
                        rounds=1,
                        sequences_batch_size=N,
                        model_queries_per_batch=5 * N,  # TODO: ensure get n designs
                        alphabet=self.alphabet,
                        eval_batch_size=1000
                    )
                elif explorer == 'cbas':
                    vae = flexs.baselines.explorers.VAE(len(seed), alphabet=self.alphabet, epochs=10, verbose=True)
                    self.explorer = CbAS(
                        model=model,
                        generator=vae,
                        cycle_batch_size=100,
                        rounds=1,
                        starting_sequence=seed,
                        sequences_batch_size=N,
                        model_queries_per_batch=5 * N,
                        alphabet=self.alphabet,
                        algo= "cbas",
                        **{name: val},
                    )
                else:
                    raise ValueError('Unknown explorer: {}'.format(explorer))

                # train predictive model on data
                self.explorer.model.train(trainseqs_n, ytrain_n)
                print('Regression model trained for {} = {}'.format(name, val))
                predcal_n = self.explorer.model.get_fitness(calseqs_n)

                # run design algorithm
                testseqs_n, predtest_n = self.explorer.propose_sequences(train_data)
                print('Designed sequences for {} = {}'.format(name, val))
                assert(len(testseqs_n) == N)

                ytest_n = self.landscape.get_fitness(testseqs_n)
                noise = sc.stats.norm.rvs(loc=0, scale=self.noise_sd, size=N) # TODO: clean up FLEXS
                ytest_n = ytest_n + noise
                testseqs_list.append((testseqs_n, ytest_n, predtest_n))
        
                if save_fname_prefix is not None:
                    fname = os.path.join(save_fname_prefix, 'seed{}-n{}-nmut{}-{}{}.npz'.format(
                        seed_idx, n, avg_n_mut, name, val))
                    
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


def generate_rna_data(model_class,
                      explorer: str,
                      explorer_kwarg_name2vals,
                      save_fname_dir: str,
                      noise_sd: float = .0,
                      trial_idxs = None,
                      model_kwargs = None,
                      seq_len: int = 50,
                      seed_idxs = None,
                      landscape_names = None,
                      ns = None,
                      N: int = None,
                      avg_n_muts = None):

    if landscape_names is None:
        landscape_names = flexs.landscapes.rna.registry().keys()
        landscape_names = [name for name in landscape_names if 'L{}'.format(seq_len) in name and '+' not in name]
    print('Generating data from the following landscapes:')
    print(landscape_names)
    if trial_idxs is None:
        trial_idxs = range(10)
        print('Using trial_idxs {}'.format(trial_idxs))

    if ns is None:
        ns = [1000, 10000]
    print('with the following amounts of training data:')
    print(ns)

    if seed_idxs is None:
        seed_idxs = flexsshift.problem['starts'].keys()

    if avg_n_muts is None:
        avg_n_muts = [3]

    for landscape_name in landscape_names:

        flexsshift = FLEXSShift(landscape_name, noise_sd=noise_sd)

        for n in ns:
            
            for seed_idx in seed_idxs:

                for avg_n_mut in avg_n_muts:

                    for t in trial_idxs:

                        save_fname_prefix = os.path.join(
                            '/data/wongfanc/dre-data/data', save_fname_dir, landscape_name, 'trial{}'.format(t)
                        )
                        Path(save_fname_prefix).mkdir(parents=True, exist_ok=True)
                        
                        t0 = time()

                        _ = flexsshift.get_data(
                            n,
                            model_class,
                            explorer,
                            explorer_kwarg_name2vals,
                            N=N,
                            model_kwargs=model_kwargs,
                            seed_idx=seed_idx,
                            avg_n_mut=avg_n_mut,
                            save_fname_prefix=save_fname_prefix
                        )    
                        print('Generated and saved data for {}, n = {}, seed {}, avg. # train mutations {} ({} s).'.format(
                            landscape_name, n, seed_idx, avg_n_mut, int(time() - t0)
                        ))

def process_getted_data(trainseqs_n, ytrain_n, calseqs_n, testseqs_list):
    m = len(testseqs_list)
    n = len(trainseqs_n)
    seq_len = len(trainseqs_n[0])
    d = seq_len * len(sutils.RNAA)

    X_m1xnxd = np.zeros([m + 1, n, d])
    y_m1xn = np.zeros([m + 1, n])
    # slice i corresponds to i in X_m1xnxd. no predictions on training data
    pred_mxn = np.zeros([m, n])

    for k in range(m): 

        testseqs_n, ytest_n, predtest_n = testseqs_list[k]

        Xk_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in testseqs_n])
        X_m1xnxd[k] = Xk_nxd
        y_m1xn[k] = ytest_n
        pred_mxn[k] = predtest_n  # no predictions on training data

    # training data
    Xm_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in trainseqs_n])
    X_m1xnxd[-1] = Xm_nxd
    y_m1xn[-1] = ytrain_n

    # calibration data
    Xcal_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in calseqs_n])

    return X_m1xnxd, y_m1xn, pred_mxn, Xcal_nxd

def load_rna_data(landscape_name: str, seed_idx: int, n: int, N: int, explorer_kwarg_name2vals, save_fname_dir: str, avg_n_mut: int, trial_idx: int):
    
    assert(len(explorer_kwarg_name2vals.keys()) == 1)
    hp_name = list(explorer_kwarg_name2vals.keys())[0]
    hp_vals = explorer_kwarg_name2vals[hp_name]
    m = len(hp_vals)  # number of bridge ratios

    seq_length = int(parse('L{}_RNA{}', landscape_name)[0])
    print('Problem has sequence length {}'.format(seq_length))
    d = seq_length * len(sutils.RNAA)
    X_mxnxd = np.zeros([m, N, d])
    y_mxn = np.zeros([m, N])
    pred_mxn = np.zeros([m, N])  # slice i corresponds to i in X_mxnxd. no predictions on training data

    print('Loading waymarks in the following order for k = 0, 1, ... where k = 0 is the target design distribution.')
    print(hp_vals)
    
    for k, val in enumerate(hp_vals):

        fname = 'seed{}-n{}-nmut{}-{}{}.npz'.format(seed_idx, n, avg_n_mut, hp_name, val)
        save_fname = os.path.join(
            '/data/wongfanc/dre-data/data', save_fname_dir, landscape_name, 'trial{}'.format(trial_idx), fname
        )
        print('Loading from {}'.format(save_fname))
        loaded_dict = np.load(save_fname)

        # design data
        testseqs_n = loaded_dict['testseqs_n']
        Xk_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in testseqs_n])
        X_mxnxd[k] = Xk_nxd
        y_mxn[k] = loaded_dict['ytest_n']
        pred_mxn[k] = loaded_dict['predtest_n']  # no predictions on training data

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


    # training and calibration sequences. could load from any hp_val, all the same training sequences
    loaded_dict = np.load(save_fname)
    trainseqs_n = loaded_dict['trainseqs_n']
    Xtr_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in trainseqs_n])
    calseqs_n = loaded_dict['calseqs_n']
    assert(len(trainseqs_n) + len(calseqs_n) == n)
    Xcal_nxd = np.array([sutils.string_to_one_hot(seq, sutils.RNAA).flatten() for seq in calseqs_n])
    Xtrcal_nxd = np.concatenate([Xtr_nxd, Xcal_nxd], axis=0)
    
    # ytrain_n = loaded_dict['ytrain_n']
    ycal_n = loaded_dict['ycal_n']
    predcal_n = loaded_dict['predcal_n']

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

    return X_mxnxd, y_mxn, pred_mxn, Xtrcal_nxd, Xcal_nxd, ycal_n, predcal_n