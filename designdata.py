from time import time
from pathlib import Path
import os.path
from parse import parse

import numpy as np
import scipy as sc
import pandas as pd
import editdistance

from keras import backend as K

import flexs
import flexs.utils.sequence_utils as sutils
from flexs.baselines.explorers.cbas_dbas import CbAS

from shifts import get_mutant

def get_keras_embeddings(functors, seq_n, alphabet, layer_idx: int = 6):
        X_nxp = np.stack([sutils.string_to_one_hot(seq, alphabet) for seq in seq_n])
        embeddings = []
        for i in range(len(functors)):
            embeddings.append(functors[i]([X_nxp])[layer_idx])
        emb_nxd = np.concatenate(embeddings, axis=1)
        return emb_nxd

class DesignDataFactory(object):
    def __init__(self, landscape_name: str, noise_sd: float = 0.0, seed = None):
        
        if 'RNA' in landscape_name:
            problem = flexs.landscapes.rna.registry()[landscape_name]
            self.landscape = flexs.landscapes.RNABinding(**problem['params'])
            self.alphabet = sutils.RNAA
            self.seq_len = self.landscape.seq_length
            self.noise_sd = noise_sd
            if seed is None:
                self.seed = problem['starts'][0]
                print('Using seed #0.')
            elif type(seed) is int:
                self.seed = problem['starts'][seed]
                print('Using seed #{}'.format(seed))
            elif type(seed) is str:
                self.seed = seed
                print('Using provided seed: {}'.format(self.seed))
            
        elif landscape_name == 'gfp':
            self.landscape = flexs.landscapes.BertGFPBrightness()
            self.alphabet = sutils.AAS
            self.seq_len = len(self.landscape.gfp_wt_sequence)
            self.noise_sd = noise_sd
            self.seed = self.landscape.starts['ed_18_wt']
            print('Using seed ed_18_wt.')
            
        else:
            raise ValueError('Unrecognized landscape_name: {}'.format(landscape_name))


    def get_data(self, n: int, model_name: str, model_kwargs, explorer_name: str, hp_name: str, hp_vals, 
                 N: int = None, save_path: str = None, n_cal: int = 1000, n_mut: int = 3,
                 save_keras_weights: bool = False, c: float = 0.01, keras_layer_idx: int = 6, heteroskedastic: bool = False):
        if model_kwargs is None:
            model_kwargs = {}

        if model_name == 'linear':  # TODO: should this go down in for loop?
            model = flexs.baselines.models.LinearRegression(alphabet=self.alphabet, **model_kwargs)
        elif model_name == 'ridge':
            model = flexs.baselines.models.RidgeCV(alphabet=self.alphabet, **model_kwargs)
        elif model_name == 'ff':
            model = flexs.Ensemble([
                flexs.baselines.models.MLP(self.seq_len, alphabet=self.alphabet, **model_kwargs) for _ in range(3)
            ])
        elif model_name == 'cnn':
            model = flexs.Ensemble([
                flexs.baselines.models.CNN(self.seq_len, alphabet=self.alphabet, **model_kwargs) for _ in range(3)
            ])
        else:
            raise ValueError('Unrecognized model_name: {}'.format(model_name))
        
        if N is None:
            N = n

        # generate training sequences around seed
        p_mut = n_mut / self.seq_len
        trainseqs_n = [get_mutant(self.seed, p_mut, self.alphabet) for _ in range(n)]
        print('Generating {} training and calibration data...'.format(n))
        t0 = time()
        ytrain_n = self.landscape.get_fitness(trainseqs_n)
        print('Done. ({} s)'.format(int(time() - t0)))
        if heteroskedastic:
            # noise_n = np.array([sc.stats.norm.rvs(loc=0, scale=self.noise_sd / y) for y in ytrain_n])
            noise_n = -np.array([c * editdistance.eval(seq, self.landscape.gfp_wt_sequence) for seq in trainseqs_n])
        else:
            noise_n = sc.stats.norm.rvs(loc=0, scale=self.noise_sd, size=len(trainseqs_n))
        ytrain_n = ytrain_n + noise_n

        # split into training and calibration 
        trainseqs_n, calseqs_n = trainseqs_n[: n - n_cal], trainseqs_n[n - n_cal :]
        ytrain_n, ycal_n = ytrain_n[: n - n_cal], ytrain_n[n - n_cal :]
        assert(ytrain_n.size + ycal_n.size == n)        

        train_data = pd.DataFrame(
            {
                "sequence": trainseqs_n,
                "model_score": np.nan,
                "true_score": ytrain_n,
                "round": 1,
                "model_cost": model.cost,
                "measurement_cost": 1,
            }
        )

        testseqs_list = []
        testembs_list = []

        if explorer_name == 'adalead':
            self.explorer = flexs.baselines.explorers.Adalead(
                model,
                **{hp_name: hp_vals[0]},
                starting_sequence=self.seed,
                rounds=1,
                sequences_batch_size=N,
                model_queries_per_batch=10 * N,  # TODO: ensure get n designs
                alphabet=self.alphabet,
                eval_batch_size=1000
            )
        elif explorer_name in ['cbas', 'dbas']:
            vae = flexs.baselines.explorers.VAE(
                len(self.seed),
                alphabet=self.alphabet,
                batch_size=10,
                latent_dim=20,
                intermediate_dim=50,
                epochs=20,
                lr=1e-3,
                verbose=True
            )
            self.explorer = CbAS(
                model=model,
                generator=vae,
                cycle_batch_size=2000,
                proposal_update_rounds=20,
                epochs_per_proposal_update=10,
                rounds=1,
                starting_sequence=self.seed,
                sequences_batch_size=N,
                model_queries_per_batch=10 * N,
                alphabet=self.alphabet,
                algo=explorer_name,
                **{hp_name: hp_vals[0]},
            )
        elif explorer_name == 'pex':
            raise NotImplementedError
            # self.explorer = PEX()  # TODO
        else:
            raise ValueError('Unknown explorer_name: {}'.format(explorer_name))
        
        # train predictive model
        print('Training regression model...')
        t0 = time()
        self.explorer.model.train(trainseqs_n, ytrain_n)
        print('Done ({} s).'.format(int(time() - t0)))
        predcal_n = self.explorer.model.get_fitness(calseqs_n)

        # get embeddings for training and calibration sequences
        embtr_nxd, embcal_nxd = None, None
        if save_keras_weights:
            functors = []
            for i in range(len(model.models)):
                inp = model.models[i].model.input                                       # input placeholder
                outputs = [layer.output for layer in model.models[i].model.layers]      # all layer outputs
                functor = K.function([inp], outputs)   # evaluation function
                functors.append(functor)
            embtr_nxd = get_keras_embeddings(functors, trainseqs_n, self.alphabet, layer_idx=keras_layer_idx)
            embcal_nxd = get_keras_embeddings(functors, calseqs_n, self.alphabet, layer_idx=keras_layer_idx)

        for val in hp_vals:
            self.explorer.__dict__[hp_name] = val

            # run design algorithm
            print('Designing sequences for {} = {}...'.format(hp_name, self.explorer.__dict__[hp_name]))
            t0 = time()
            testseqs_n, predtest_n = self.explorer.propose_sequences(train_data)
            print('Done ({} s).'.format(int(time() - t0)))
            assert(len(testseqs_n) == N)

            # get embeddings for design sequences
            embtest_nxd = None
            if save_keras_weights:
                embtest_nxd = get_keras_embeddings(functors, testseqs_n, self.alphabet, layer_idx=keras_layer_idx)

            # get ground truth for designs
            ytest_n = self.landscape.get_fitness(testseqs_n)
            if heteroskedastic: 
                # noise_n = np.array([sc.stats.norm.rvs(loc=0, scale=self.noise_sd / y) for y in ytest_n])
                noise_n = -np.array([c * editdistance.eval(seq, self.landscape.gfp_wt_sequence) for seq in testseqs_n])
            else:
                noise_n = sc.stats.norm.rvs(loc=0, scale=self.noise_sd, size=N)
            ytest_n = ytest_n + noise_n
            testseqs_list.append((testseqs_n, ytest_n, predtest_n))
            testembs_list.append(embtest_nxd)
    
            if save_path is not None:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                fname = os.path.join(save_path, '{}-n{}-nmut{}-nsd{}-{}{}.npz'.format(
                    model_name, n - n_cal, n_mut, self.noise_sd, hp_name, val))
                np.savez(
                    fname,
                    trainseqs_n=trainseqs_n,
                    ytrain_n=ytrain_n,
                    embtr_nxd=embtr_nxd,
                    calseqs_n=calseqs_n,
                    ycal_n=ycal_n,
                    predcal_n=predcal_n,
                    embcal_nxd=embcal_nxd,
                    testseqs_n=testseqs_n,
                    ytest_n=ytest_n,
                    predtest_n=predtest_n,
                    embtest_nxd=embtest_nxd,
                )
                print('Saved training, calibration, and design data to {}'.format(fname))

        return trainseqs_n, ytrain_n, embtr_nxd, calseqs_n, ycal_n, predcal_n, embcal_nxd, testseqs_list, testembs_list


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
                      avg_n_muts = None):  # HERE: edit, then start running trials for noise_sd = 0.5 and 0.7

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


    if avg_n_muts is None:
        avg_n_muts = [3]

    for landscape_name in landscape_names:

        flexsshift = FLEXSShift(landscape_name, noise_sd=noise_sd)
        if landscape_name.lower() == 'gfp':
                seed_idxs = [0]
        else:
            if seed_idxs is None:
                seed_idxs = flexsshift.problem['starts'].keys()

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


def process_getted_data(trainseqs_n, calseqs_n, testseqs_list, alphabet):
    m = len(testseqs_list)  # number of distributions
    N = len(testseqs_list[0][0])  # number of designs
    seq_len = len(trainseqs_n[0])
    d = seq_len * len(alphabet)

    X_mxNxd = np.zeros([m, N, d])
    y_mxN = np.zeros([m, N])
    pred_mxN = np.zeros([m, N])

    for k in range(m): 
        testseqs_n, ytest_n, predtest_n = testseqs_list[k]
        Xk_Nxd = np.array([sutils.string_to_one_hot(seq, alphabet).flatten() for seq in testseqs_n])
        X_mxNxd[k] = Xk_Nxd
        y_mxN[k] = ytest_n
        pred_mxN[k] = predtest_n

    # training data
    Xtr_nxd = np.array([sutils.string_to_one_hot(seq, alphabet).flatten() for seq in trainseqs_n])
    Xcal_nxd = np.array([sutils.string_to_one_hot(seq, alphabet).flatten() for seq in calseqs_n])
    Xtrcal_nxd = np.vstack([Xtr_nxd, Xcal_nxd]) 

    return X_mxNxd, y_mxN, pred_mxN, Xtrcal_nxd

def load_rna_data(landscape_name: str, seed_idx: int, n: int, N: int, explorer_kwarg_name2vals, save_fname_dir: str, avg_n_mut: int, trial_idx: int):
    
    assert(len(explorer_kwarg_name2vals.keys()) == 1)
    hp_name = list(explorer_kwarg_name2vals.keys())[0]
    hp_vals = explorer_kwarg_name2vals[hp_name]
    m = len(hp_vals)  # number of bridge ratios

    if "RNA" in landscape_name:
        seq_length = int(parse('L{}_RNA{}', landscape_name)[0])
        alphabet = sutils.RNAA
        d = seq_length * len(alphabet)
    elif landscape_name.lower() == 'gfp':
        seq_length = 238
        alphabet = sutils.AAS
        d = seq_length * len(alphabet)
    print('Problem has sequence length {}'.format(seq_length))
    
    X_mxnxd = np.zeros([m, N, d])
    y_mxn = np.zeros([m, N])
    pred_mxn = np.zeros([m, N])  # slice i corresponds to i in X_mxnxd. no predictions on training data
    print('Loading waymarks in the following order for k = 0, 1, ... where k = 0 is the target design distribution.')
    print(hp_vals)
    
    for k, val in enumerate(hp_vals):

        # fname = 'seed{}-n{}-nmut{}-{}{}.npz'.format(seed_idx, n, avg_n_mut, hp_name, val)
        fname = 'n{}-nmut{}-{}{}.npz'.format(seed_idx, n, avg_n_mut, hp_name, val)
        save_fname = os.path.join(
            '/data/wongfanc/dre-data/data', save_fname_dir, landscape_name, 'trial{}'.format(trial_idx), fname
        )
        print('Loading from {}'.format(save_fname))
        loaded_dict = np.load(save_fname)

        # design data
        testseqs_n = loaded_dict['testseqs_n']
        Xk_nxd = np.array([sutils.string_to_one_hot(seq, alphabet).flatten() for seq in testseqs_n])
        X_mxnxd[k] = Xk_nxd
        y_mxn[k] = loaded_dict['ytest_n']
        pred_mxn[k] = loaded_dict['predtest_n']  # no predictions on training data


    # training and calibration sequences. could load from any hp_val, all the same training sequences
    loaded_dict = np.load(save_fname)
    trainseqs_n = loaded_dict['trainseqs_n']
    Xtr_nxd = np.array([sutils.string_to_one_hot(seq, alphabet).flatten() for seq in trainseqs_n])
    calseqs_n = loaded_dict['calseqs_n']
    assert(len(trainseqs_n) + len(calseqs_n) == n)
    Xcal_nxd = np.array([sutils.string_to_one_hot(seq, alphabet).flatten() for seq in calseqs_n])
    Xtrcal_nxd = np.concatenate([Xtr_nxd, Xcal_nxd], axis=0)
    
    # ytrain_n = loaded_dict['ytrain_n']
    ycal_n = loaded_dict['ycal_n']
    predcal_n = loaded_dict['predcal_n']

    return X_mxnxd, y_mxn, pred_mxn, Xtrcal_nxd, Xcal_nxd, ycal_n, predcal_n