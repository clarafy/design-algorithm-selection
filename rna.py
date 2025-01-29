import os
import pickle
from time import time

import numpy as np
import scipy as sc
from sklearn.isotonic import IsotonicRegression
from pandas import DataFrame
from statsmodels.stats.weightstats import _zstat_generic

try:
    import RNA
except ImportError:
    pass

import models
import designers
from dre import MultiMDRE, prepare_name2designdata, is_intermediate_iteration_name
import utils
from utils import RNA_NUCLEOTIDES, RNANUC2COMPLEMENT, get_mutant, get_conformal_prediction_lower_bound, editdistance, wheelock_mean_forecast
from utils import rectified_p_value


# ===== ViennaRNA binding landscape =====


class RNABinding():
    """
    RNA binding landscape using ViennaRNA duplexfold.
    
    Adapted from the FLEXS package.
    Original source: https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/rna.py.
    """

    BINDING_TARGETS = [
        "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACCCCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",  
        "GAGGCACAUUCCGGCUCGCCCCCGUCCGCGCGGGGGCCCCGCGCGGACGGGGUCCGGCCCGCGCGGGGCCCCCGCGCGGGAGCCGGAAUGUGCCUCGUUC",  
        "CCGGUGAUACUGUUAGUGGUCACGGUGCAUUUAUAGCGCUAAAGUACAGUCUUCCCCUGUUGAACGGCGCCAUUGCAUACAGGGCCAGCCGCGUAACGCC", 
        "UAAGAGAGCGUAAAAAUAGAGAUAUGUUCUUGGGUCAGGGCUAUGCGUACCCCAUGAGAGUAAAUCAUACCCCCAAUGGGCUUCGGCGGAAAUUCACUUA",
    ]

    SEEDS = [
        "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACC",
        "CCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",
        "AUGUUUCUUUUAUUUAUCUGAGCAUGGGCGGGGCAUUUGCCCAUGCAAUU",
        "UAAACGAUGCUUUUGCGCCUGCAUGUGGGUUAGCCGAGUAUCAUGGCAAU",
        "AGGGAAGAUUAGAUUACUCUUAUAUGACGUAGGAGAGAGUGCGGUUAAGA",
    ]

    def __init__(
        self,
        seq_len: int = 50,
        binding_target_idx = 0,
        noise_sd: float = 0.02,
    ):
        """
        Create an RNABinding landscape.

        Args:
            seq_len: length of sequence domain of this landscape (not necessarily equal to length of binding target).
            binding_target_idx: index or list of indices into the list of possible binding targets, BINDING_TARGETS.
            noise_sd: standard deviation of measurement noise that will be added when returning binding energy
                of a sequence.
        """
        # ViennaRNA is not available through pip, so give a warning message
        # if not installed.
        try:
            RNA
        except NameError as e:
            raise ImportError(
                f"{e}.\n"
                "Hint: ViennaRNA not installed.\n"
                "      Source and binary installations available at "
                "https://www.tbi.univie.ac.at/RNA/#download.\n"
                "      Conda installation available at "
                "https://anaconda.org/bioconda/viennarna."
            ) from e
        
        if isinstance(binding_target_idx, int):
            binding_target_idx = [binding_target_idx]
        self.targets = [self.BINDING_TARGETS[i] for i in binding_target_idx]

        self.seq_len = seq_len
        self.norm_values = self.compute_min_binding_energies()
        self.noise_sd = noise_sd

    def compute_min_binding_energies(self):
        """Compute the lowest possible binding energies for the target sequences."""
        
        energy_t = []
        for target in self.targets:
            complement = "".join(RNANUC2COMPLEMENT[nuc] for nuc in target)[::-1]
            energy = RNA.duplexfold(complement, target).energy
            energy_t.append(energy * self.seq_len / len(target))

        return np.array(energy_t)

    def get_fitness(self, sequences, geometric_mean: bool = True, noiseless: bool = False):
        fitness_n = []

        for seq in sequences:

            if len(seq) != self.seq_len:
                raise ValueError('All sequences in `sequences` must be of length {self.seq_len}.')
            
            energy_t = np.array([RNA.duplexfold(target, seq).energy for target in self.targets])

            if geometric_mean:
                fitness = sc.stats.mstats.gmean(energy_t / self.norm_values)
            else:
                fitness = np.mean(energy_t / self.norm_values)

            fitness_n.append(fitness)

        # add homoskedastic measurement noise
        fitness_n = np.array(fitness_n)
        if self.noise_sd > 0 and not noiseless:
            fitness_n = sc.stats.norm.rvs(loc=fitness_n, scale=self.noise_sd)
            fitness_n = np.fmin(np.fmax(fitness_n, 0), 1)

        return fitness_n
    
    def get_training_data(
            self,
            n_train: int,
            p_mut: float = 0.08,
            seed_idx: int = 3,
        ):
        """
        Generates labeled training sequences, each of which is a mutant of a seed with
        probability p_mut of a mutation at each nucleotide site.
        """

        trainseqs_n = [get_mutant(self.SEEDS[seed_idx], p_mut, RNA_NUCLEOTIDES) for _ in range(n_train)]
        print('Generating {} labeled sequences...'.format(n_train))
        t0 = time()
        ytrain_n = self.get_fitness(trainseqs_n)
        print('Done. ({} s)'.format(int(time() - t0)))
        return trainseqs_n, ytrain_n


# ===== functions for training models and designing sequences =====s


def train_models(
        n_train: int,
        binding_target_idx = 0,
        seed_idx = 3,
        p_mutation: float = 0.08,
        noise_sd: float = 0.02,
        n_hidden: int = 100,
        n_epoch: int = 5,
        lr: float = 0.001,
        n_filters: int = 32,
        save_path: str = '/data/wongfanc/rna-models',
        save_fname_no_ftype: str = None,
    ):
    """
    Generates training data and trains ridge regression, ensemble of CNNs, and ensemble of feedforward models.
    """

    # generate training and test data
    landscape = RNABinding(seq_len=50, binding_target_idx=binding_target_idx, noise_sd=noise_sd)
    trainseq_n, ytrain_n = landscape.get_training_data(
        n_train,
        p_mutation,
        seed_idx=seed_idx,
    )
    testseq_n, ytest_n = landscape.get_training_data(
        n_train,
        p_mutation,
        seed_idx=seed_idx,
    )

    # train models
    ridge = models.RidgeRegressor(seq_len=landscape.seq_len, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)
    print(f'CV-selected alpha for ridge: {ridge.model.alpha_}.')

    ff = models.FeedForward(landscape.seq_len, RNA_NUCLEOTIDES, n_hidden)
    _ = ff.fit(
        trainseq_n,
        ytrain_n,
        n_epoch=n_epoch,
        lr=lr,
    )

    cnn = models.CNN(landscape.seq_len, RNA_NUCLEOTIDES, n_filters, n_hidden)
    _ = cnn.fit(
        trainseq_n,
        ytrain_n,
        n_epoch=n_epoch,
        lr=lr,
    )
    
    # save models and training data
    if save_fname_no_ftype is not None:
        ff_fname = os.path.join(save_path, 'ff-' + save_fname_no_ftype + '.pt')
        ff.save(ff_fname)
        print(f'Saved FF model to {ff_fname}.')

        cnn_fname = os.path.join(save_path, 'cnn-' + save_fname_no_ftype + '.pt')
        cnn.save(cnn_fname)
        print(f'Saved CNN model to {cnn_fname}.')

        data_fname = os.path.join(save_path, 'traindata-' + save_fname_no_ftype + '.npz')
        np.savez(data_fname, trainseq_n=trainseq_n, ytrain_n=ytrain_n)
        print(f'Saved training data to {data_fname}.')
    else:
        print('`save_fname_no_ftype` not provided, not saving models or data.')
    
    return ridge, ff, cnn, trainseq_n, ytrain_n, testseq_n, ytest_n


def label_design_sequences(
    design_pkl_fname: str,
    binding_target_idx: int = 0,
):  
    print('Loading/saving labeled name2designdata to {}'.format(design_pkl_fname))
    landscape = RNABinding(binding_target_idx=[binding_target_idx], noise_sd=0)
    print('Labeling with noiseless landscape with binding target {}.'.format(binding_target_idx))

    with open(design_pkl_fname, 'rb') as f:
        name2designdata = pickle.load(f)

    t0 = time()
    for name, data in name2designdata.items():
        if name == 'train':
            continue

        # no need to label sequences from intermediate C/DbAS iterations,
        # only used to facilitate density ratio estimation
        if is_intermediate_iteration_name(name):
            print(f'Skipping labels for {name}.')
            continue

        designseq_n, ydesign_n, preddesign_n = data

        if ydesign_n is None:
            print(f'Getting noiseless labels for {name}...')
            ydesign_n = landscape.get_fitness(designseq_n, noiseless=True)
            print('  Mean prediction: {:.3f}, mean noiseless label: {:.3f}. ({} s)\n'.format(
                np.mean(preddesign_n), np.mean(ydesign_n), int(time() - t0)
            ))
            # save labels
            name2designdata[name] = (designseq_n, ydesign_n, preddesign_n)
            with open(design_pkl_fname, 'wb') as f:
                pickle.dump(name2designdata, f)

        # already labeled
        else:
            print('{} already labeled. Mean prediction: {:.3f}, mean label: {:.3f}.'.format(
                name, np.mean(preddesign_n), np.mean(ydesign_n)
            ))


def sample_design_sequences(
    n_design: int,
    adalead_thresholds,
    biswas_temperatures,
    cbas_dbas_quantiles,
    dbas_ridge_quantiles,
    model_and_data_fname_no_ftype: str,
    model_and_data_path: str = '/data/wongfanc/rna-models',
    seed_idx: int = 3,
    design_pkl_fname: str = None,
    intermediate_iter = None,
    n_hidden: int = 100,
    n_filters: int = 32,
    n_recomb_partner: int = 1,
    recomb_rate: float = 0.2,
    max_model_queries: int = None,
    max_mu: float = 2,
    n_trust_radius_mutations: int = 5,
    n_step: int = 2000,
    latent_dim: int = 10,
    n_vae_hidden: int = 20,
    p_mutation_pex: float = 0.04,
):
    """
    Samples design sequences using Adalead, Biswas, CbAS, DbAS, and PEX.
    """

    # load training data
    data_fname = os.path.join(model_and_data_path, 'traindata-' + model_and_data_fname_no_ftype + '.npz')
    d = np.load(data_fname)
    trainseq_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']
    print(f'Loaded {ytrain_n.size} training points from {data_fname}.\n')

    # train ridge regression, load trained FF and CNN models
    ridge = models.RidgeRegressor(seq_len=50, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)

    ff_fname = os.path.join(model_and_data_path, 'ff-' + model_and_data_fname_no_ftype + '.pt')
    ff = models.FeedForward(50, RNA_NUCLEOTIDES, n_hidden)
    ff.load(ff_fname)

    cnn_fname = os.path.join(model_and_data_path, 'cnn-' + model_and_data_fname_no_ftype + '.pt')
    cnn = models.CNN(50, RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)

    name2model = {
        'ridge': ridge,
        'ff': ff,
        'cnn': cnn
    }

    # design sequences
    name2designs = {}
    if design_pkl_fname is not None:
        print(f'Saving all results to {design_pkl_fname}.\n')
    else:
        print('`design_pkl_fname` not provided, not saving design results.\n')

    # ===== CbAS ridge with intermediate iterations =====
    if intermediate_iter is None:
        intermediate_iter = range(20)
    
    cbas = designers.CbAS(
        ridge,
        trainseq_n,
        latent_dim=latent_dim,
        n_hidden=n_vae_hidden,
        weight_type='cbas',
        device='cuda'
    )
    for quantile in cbas_dbas_quantiles:
        quantile = round(quantile, 2)

        # design sequences
        print('Designing CbAS {} ridge sequences from intermediate iterations {}...'.format(quantile, intermediate_iter))
        t0 = time()
        cbas_iter2designseq = cbas.design_sequences_from_intermediate_iterations(
            n_design,
            intermediate_iter,
            quantile=quantile
        )
        print(f'  Done. ({int(time() - t0)} s)')
        
        # store
        for it, cbas_n in cbas_iter2designseq.items():
            predcbas_n = ridge.predict(cbas_n)
            print('  Mean prediction for iteration {}: {:.3f}'.format(it, np.mean(predcbas_n)))
            name2designs[f'cbas-ridge-{quantile}t{it}'] = (cbas_n, None, predcbas_n)
            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} CbAS {quantile} t = {it} ridge sequences.')
            print()
    
    # ===== sample from VAE fit to training sequences =====
    print(f'Designing sequences from VAE fit to training data...')
    t0 = time()
    trainvae_n = cbas.sample_from_train_distribution(n_design)
    print(f'  Done. ({int(time() - t0)} s)')
    # store
    predtrainvae_n = ridge.predict(trainvae_n)
    print('  Mean prediction: {:.3f}'.format(np.mean(predtrainvae_n)))
    name2designs['vae-ridge'] = (trainvae_n, None, predtrainvae_n)
    # save
    with open(design_pkl_fname, 'wb') as f:
        pickle.dump(name2designs, f)
    print(f'  Saved {n_design} training VAE sequences to {design_pkl_fname}.')

    # ===== DbAS ridge with intermediate iterations =====
    dbas = designers.CbAS(
        ridge,
        trainseq_n,
        latent_dim=latent_dim,
        n_hidden=n_vae_hidden,
        weight_type='dbas',
        device='cuda'
    )
    for quantile in dbas_ridge_quantiles:
        quantile = round(quantile, 2)

        # design sequences
        print('Designing DbAS {} ridge sequences from intermediate iterations {}...'.format(quantile, intermediate_iter))
        t0 = time()
        dbas_iter2designseq = dbas.design_sequences_from_intermediate_iterations(
            n_design,
            intermediate_iter,
            quantile=quantile
        )
        print(f'  Done. ({int(time() - t0)} s)')

        # store
        for it, dbas_n in dbas_iter2designseq.items():
            preddbas_n = ridge.predict(dbas_n)
            print('  Mean prediction for iteration {}: {:.3f}'.format(it, np.mean(preddbas_n)))
            name2designs[f'dbas-ridge-{quantile}t{it}'] = (dbas_n, None, preddbas_n)
            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} DbAS {quantile} t = {it} ridge sequences.')
            print()
    

    # all other configurations
    if max_model_queries is None:
        max_model_queries = 5 * n_design
    for model_name, model in name2model.items():

        # ----- C/DbAS with other models, no intermediate iterations -----
        if model_name != 'ridge':
            cbas = designers.CbAS(
                model,
                trainseq_n,
                latent_dim=latent_dim,
                n_hidden=n_vae_hidden,
                weight_type='cbas',
                device='cuda'
            )
            for quantile in cbas_dbas_quantiles:
                quantile = round(quantile, 2)

                # design sequences
                print(f'Designing CbAS {quantile} {model_name} sequences...')
                t0 = time()
                cbas_n = dbas.design_sequences(
                    n_design,
                    quantile=quantile
                )
                print(f'  Done. ({int(time() - t0)} s)')
                # store
                predcbas_n = model.predict(cbas_n)
                print('  Mean prediction: {:.3f}'.format(np.mean(predcbas_n)))
                name2designs[f'cbas-{model_name}-{quantile}'] = (cbas_n, None, predcbas_n)
                # save
                if design_pkl_fname is not None:
                    with open(design_pkl_fname, 'wb') as f:
                        pickle.dump(name2designs, f)
                    print(f'  Saved {n_design} CbAS {quantile} {model_name} sequences.')
                print()

            dbas = designers.CbAS(
                model,
                trainseq_n,
                latent_dim=latent_dim,
                n_hidden=n_vae_hidden,
                weight_type='dbas',
                device='cuda'
            )
            for quantile in cbas_dbas_quantiles:
                quantile = round(quantile, 2)

                # design sequences
                print(f'Designing DbAS {quantile} {model_name} sequences...')
                t0 = time()
                dbas_n = dbas.design_sequences(
                    n_design,
                    quantile=quantile
                )
                print(f'  Done. ({int(time() - t0)} s)')
                # store
                preddbas_n = model.predict(dbas_n)
                print('  Mean prediction: {:.3f}'.format(np.mean(preddbas_n)))
                name2designs[f'dbas-{model_name}-{quantile}'] = (dbas_n, None, preddbas_n)
                # save
                if design_pkl_fname is not None:
                    with open(design_pkl_fname, 'wb') as f:
                        pickle.dump(name2designs, f)
                    print(f'  Saved {n_design} DbAS {quantile} {model_name} sequences.')
                print()

        # ===== AdaLead =====
        adalead = designers.AdaLead(model, trainseq_n, ytrain_n)
        for threshold in adalead_thresholds:

            threshold = round(threshold, 4)
            print(f'Designing AdaLead threshold = {threshold} {model_name} sequences...')
            
            # design sequences
            t0 = time()
            adalead_n = adalead.design_sequences(
                n_design=n_design,
                threshold=threshold,
                n_recomb_partner=n_recomb_partner,
                recomb_rate=recomb_rate,
                max_model_queries=max_model_queries
            )
            print(f'  Done. ({int(time() - t0)} s)')
            
            # store
            predadalead_n = model.predict(adalead_n)
            # yadalead_n = landscape.get_fitness(adalead_n)
            print('  Mean prediction: {:.3f}'.format(np.mean(predadalead_n)))
            name2designs[f'adalead-{model_name}-{threshold}'] = (adalead_n, None, predadalead_n)
            
            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} AdaLead threshold = {threshold} {model_name} sequences.')
            print()

        # ===== PEX =====
        pex = designers.PEX(
            model,
            trainseq_n,
            ytrain_n,
            RNABinding.SEEDS[seed_idx],
        )
        # design sequences
        print(f'Designing PEX {model_name} sequences...')
        t0 = time()
        pex_n = pex.design_sequences(
            n_design,
            p_mutation_pex,
        )
        print(f'  Done. ({int(time() - t0)} s)')
        # store
        predpex_n = model.predict(pex_n)
        # ypex_n = landscape.get_fitness(pex_n)
        print('  Mean prediction: {:.3f}'.format(np.mean(predpex_n)))
        name2designs[f'pex-{model_name}'] = (pex_n, None, predpex_n)
        # save
        if design_pkl_fname is not None:
            with open(design_pkl_fname, 'wb') as f:
                pickle.dump(name2designs, f)
            print(f'  Saved {n_design} PEX {model_name} sequences.')
        print()

        # ===== Biswas =====
        biswas = designers.Biswas(model, trainseq_n)
        for temp in biswas_temperatures:

            temp = round(temp, 4)
            print(f'Designing Biswas temperature = {temp} {model_name} sequences...')

            # design sequences
            t0 = time()
            biswas_n, _, _ = biswas.design_sequences(
                n_design,
                RNABinding.SEEDS[seed_idx],
                max_mu,
                temp,
                n_trust_radius_mutations,
                n_step,
                print_every=500
            )
            print(f'  Done. ({int(time() - t0)} s)')

            # store
            predbiswas_n = model.predict(biswas_n)
            # ybiswas_n = landscape.get_fitness(biswas_n)
            print('  Mean prediction: {:.3f}'.format(np.mean(predbiswas_n)))
            name2designs[f'biswas-{model_name}-{temp}'] = (biswas_n, None, predbiswas_n)

            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} Biswas temperature = {temp} {model_name} sequences.')
            print()

    return name2designs


def sample_design_sequences_2(
    n_design: int,
    adalead_thresholds,
    biswas_temperatures,
    cbas_dbas_quantiles,
    model_and_data_fname_no_ftype: str,
    model_and_data_path: str = '/data/wongfanc/rna-models',
    seed_idx: int = 3,
    design_pkl_fname: str = None,
    n_hidden: int = 100,
    n_filters: int = 32,
    n_recomb_partner: int = 1,
    recomb_rate: float = 0.2,
    max_model_queries: int = None,
    max_mu: float = 2,
    n_trust_radius_mutations: int = 5,
    n_step: int = 2000,
    latent_dim: int = 10,
    n_vae_hidden: int = 20,
    p_mutation_pex: float = 0.04,
):
    """
    Samples design sequences using Adalead, Biswas, CbAS, DbAS, and PEX.
    """

    # ===== load training data and predictive models =====
    data_fname = os.path.join(model_and_data_path, 'traindata-' + model_and_data_fname_no_ftype + '.npz')
    d = np.load(data_fname)
    trainseq_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']
    print(f'Loaded {ytrain_n.size} training points from {data_fname}.\n')

    # train ridge regression, load trained FF and CNN models
    ridge = models.RidgeRegressor(seq_len=50, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)

    ff_fname = os.path.join(model_and_data_path, 'ff-' + model_and_data_fname_no_ftype + '.pt')
    ff = models.FeedForward(50, RNA_NUCLEOTIDES, n_hidden)
    ff.load(ff_fname)

    cnn_fname = os.path.join(model_and_data_path, 'cnn-' + model_and_data_fname_no_ftype + '.pt')
    cnn = models.CNN(50, RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)

    name2model = {
        'ridge': ridge,
        'ff': ff,
        'cnn': cnn
    }

    # ===== design sequences =====
    name2designs = {}
    if design_pkl_fname is not None:
        print(f'Saving all results to {design_pkl_fname}.\n')
    else:
        print('`design_pkl_fname` not provided, not saving design results.\n')

    if max_model_queries is None:
        max_model_queries = 5 * n_design
    for model_name, model in name2model.items():

        # ----- CbAS -----
        cbas = designers.CbAS(
            model,
            trainseq_n,
            latent_dim=latent_dim,
            n_hidden=n_vae_hidden,
            weight_type='cbas',
            device='cuda'
        )
        for quantile in cbas_dbas_quantiles:
            quantile = round(quantile, 2)

            # design sequences
            print(f'Designing CbAS {quantile} {model_name} sequences...')
            t0 = time()
            cbas_n = cbas.design_sequences(
                n_design,
                quantile=quantile
            )
            print(f'  Done. ({int(time() - t0)} s)')
            # store
            predcbas_n = model.predict(cbas_n)
            print('  Mean prediction: {:.3f}'.format(np.mean(predcbas_n)))
            name2designs[f'cbas-{model_name}-{quantile}'] = (cbas_n, None, predcbas_n)
            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} CbAS {quantile} {model_name} sequences.')
            print()

        # ----- DbAS -----
        dbas = designers.CbAS(
            model,
            trainseq_n,
            latent_dim=latent_dim,
            n_hidden=n_vae_hidden,
            weight_type='dbas',
            device='cuda'
        )
        for quantile in cbas_dbas_quantiles:
            quantile = round(quantile, 2)
            if model_name == 'ridge' and quantile > 0.25:
                continue

            # design sequences
            print(f'Designing DbAS {quantile} {model_name} sequences...')
            t0 = time()
            dbas_n = dbas.design_sequences(
                n_design,
                quantile=quantile
            )
            print(f'  Done. ({int(time() - t0)} s)')
            # store
            preddbas_n = model.predict(dbas_n)
            print('  Mean prediction: {:.3f}'.format(np.mean(preddbas_n)))
            name2designs[f'dbas-{model_name}-{quantile}'] = (dbas_n, None, preddbas_n)
            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} DbAS {quantile} {model_name} sequences.')
            print()

        # ===== AdaLead =====
        adalead = designers.AdaLead(model, trainseq_n, ytrain_n)
        for threshold in adalead_thresholds:

            threshold = round(threshold, 4)
            print(f'Designing AdaLead threshold = {threshold} {model_name} sequences...')
            
            # design sequences
            t0 = time()
            adalead_n = adalead.design_sequences(
                n_design=n_design,
                threshold=threshold,
                n_recomb_partner=n_recomb_partner,
                recomb_rate=recomb_rate,
                max_model_queries=max_model_queries
            )
            print(f'  Done. ({int(time() - t0)} s)')
            
            # store
            predadalead_n = model.predict(adalead_n)
            # yadalead_n = landscape.get_fitness(adalead_n)
            print('  Mean prediction: {:.3f}'.format(np.mean(predadalead_n)))
            name2designs[f'adalead-{model_name}-{threshold}'] = (adalead_n, None, predadalead_n)
            
            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} AdaLead threshold = {threshold} {model_name} sequences.')
            print()

        # ===== PEX =====
        pex = designers.PEX(
            model,
            trainseq_n,
            ytrain_n,
            RNABinding.SEEDS[seed_idx],
        )
        # design sequences
        print(f'Designing PEX {model_name} sequences...')
        t0 = time()
        pex_n = pex.design_sequences(
            n_design,
            p_mutation_pex,
        )
        print(f'  Done. ({int(time() - t0)} s)')
        # store
        predpex_n = model.predict(pex_n)
        # ypex_n = landscape.get_fitness(pex_n)
        print('  Mean prediction: {:.3f}'.format(np.mean(predpex_n)))
        name2designs[f'pex-{model_name}'] = (pex_n, None, predpex_n)
        # save
        if design_pkl_fname is not None:
            with open(design_pkl_fname, 'wb') as f:
                pickle.dump(name2designs, f)
            print(f'  Saved {n_design} PEX {model_name} sequences.')
        print()

        # ===== Biswas =====
        biswas = designers.Biswas(model, trainseq_n)
        for temp in biswas_temperatures:

            temp = round(temp, 4)
            print(f'Designing Biswas temperature = {temp} {model_name} sequences...')

            # design sequences
            t0 = time()
            biswas_n, _, _ = biswas.design_sequences(
                n_design,
                RNABinding.SEEDS[seed_idx],
                max_mu,
                temp,
                n_trust_radius_mutations,
                n_step,
                print_every=500
            )
            print(f'  Done. ({int(time() - t0)} s)')

            # store
            predbiswas_n = model.predict(biswas_n)
            # ybiswas_n = landscape.get_fitness(biswas_n)
            print('  Mean prediction: {:.3f}'.format(np.mean(predbiswas_n)))
            name2designs[f'biswas-{model_name}-{temp}'] = (biswas_n, None, predbiswas_n)

            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} Biswas temperature = {temp} {model_name} sequences.')
            print()
    
    # ===== sample from VAE fit to training sequences =====
    print(f'Designing sequences from VAE fit to training data...')
    t0 = time()
    cbas = designers.CbAS(
        ridge,
        trainseq_n,
        latent_dim=latent_dim,
        n_hidden=n_vae_hidden,
        weight_type='cbas',
        device='cuda'
    )
    trainvae_n = cbas.sample_from_train_distribution(n_design)
    print(f'  Done. ({int(time() - t0)} s)')
    # store
    predtrainvae_n = ridge.predict(trainvae_n)
    print('  Mean prediction: {:.3f}'.format(np.mean(predtrainvae_n)))
    name2designs['vae-ridge'] = (trainvae_n, None, predtrainvae_n)
    # save
    with open(design_pkl_fname, 'wb') as f:
        pickle.dump(name2designs, f)
    print(f'  Saved {n_design} training VAE sequences to {design_pkl_fname}.')

    return name2designs


def run_imputation_selection_experiments(
    design_names,
    n_trial: int,
    wheelock_forecast_qs: np.array,
    design_pkl_fname_no_trial: str,
    model_and_data_fname_no_ftype: str,
    target_values: np.array,
    seed: str = RNABinding.SEEDS[3],
    n_hidden: int = 100,
    n_filters: int = 32,
    intermediate_iter_threshold: float = 0.1,
    imp_csv_fname: str = None,
    wheelock_csv_fname: str = None,
    model_and_data_path: str = '/data/wongfanc/rna-models'  
):

    # ----- load training data and models -----
    train_fname = os.path.join(model_and_data_path, 'traindata-' + model_and_data_fname_no_ftype + '.npz')
    d = np.load(train_fname)
    trainseq_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']

    # get edit distances from seed for Wheelock forecasts
    t0 = time()
    trained_n = np.array([editdistance.eval(seed, seq) for seq in trainseq_n])
    print('Done computing edit distance of {} training sequences to the seed ({} s).'.format(
        len(trainseq_n), int(time() - t0)
    ))

    # fit ridge
    ridge = models.RidgeRegressor(seq_len=50, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)

    # load trained FF and CNN models
    ff_fname = os.path.join(model_and_data_path, 'ff-' + model_and_data_fname_no_ftype + '.pt')
    ff = models.FeedForward(50, RNA_NUCLEOTIDES, n_hidden)
    ff.load(ff_fname)
    cnn_fname = os.path.join(model_and_data_path, 'cnn-' + model_and_data_fname_no_ftype + '.pt')
    cnn = models.CNN(50, RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)
    name2model = {
        'ridge': ridge,
        'ff': ff,
        'cnn': cnn
    }

    # predictions on training data
    name2predtrain = {name: model.predict(trainseq_n) for name, model in name2model.items()}

    # dataframe to record selection experiment results
    imp_selected_column_names = ['tr{}_imp_pval_{}'.format(i, name) for i in range(n_trial) for name in design_names]
    target_values = [round(val, 4) for val in target_values]
    df = DataFrame(index=target_values, columns=imp_selected_column_names)

    # dataframe for Wheelock forecasting results
    wf_column_names = ['wf_mean_q{:.2f}_{}'.format(q, name) for q in wheelock_forecast_qs for name in design_names]
    wf_cs_column_names = ['wf_mean_q{:.2f}_cs_{}'.format(q, name) for q in wheelock_forecast_qs for name in design_names]
    wf_df = DataFrame(index=range(n_trial), columns=wf_column_names + wf_cs_column_names)

    t0 = time()
    name2truemeans = {design_name: [] for design_name in design_names}
    if imp_csv_fname is not None:
        assert(wheelock_csv_fname is not None)
        results_pkl_fname = imp_csv_fname[: -4] + '-truemeans.pkl'
        
    # ===== run selection experiments =====
    for i in range(n_trial):

        # ----- load design sequences -----
        design_pkl_fname = design_pkl_fname_no_trial + '-{}.pkl'.format(i)
        # make sure designs have labels, add training data, select intermediate C/DbAS ridge iterations
        name2designdata = prepare_name2designdata(  
            design_pkl_fname,
            train_fname,
            intermediate_iter_threshold=intermediate_iter_threshold,
            verbose=False
        )

        assert('train' in name2designdata)
        assert(name2designdata['train'][2] is None)

        # print('All design names in provided design data:')
        # for name in name2designdata:
        #     print(name)
        # print()
            
        # name2designdata may contain other distributions used to faciliate DRE,
        # but which we are not interested in designs from
        for name in design_names:
            if name not in name2designdata:
                raise ValueError(f'{name} not in design data at {design_pkl_fname}.')

        # ----- imputation selection experiments -----
        for design_name in design_names:
            (designseq_N, ydesign_n, preddesign_n) = name2designdata[design_name]
            imputed_mean = np.mean(preddesign_n)
            imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)
            name2truemeans[design_name].append(np.mean(ydesign_n))

            for target_val in target_values:
                # get imputation p-value
                imp_pval = _zstat_generic(
                    imputed_mean,
                    0,
                    imputed_se,
                    alternative='larger',
                    diff=target_val
                )[1]

                df.loc[target_val]['tr{}_imp_pval_{}'.format(i, design_name)] = imp_pval

            # ===== Wheelock forecast ===== 
            assigned_model = False
            for model_name in name2predtrain.keys():
                if model_name in design_name:
                    predtrain_n = name2predtrain[model_name]
                    assigned_model = True
                    break
            if not assigned_model:
                raise ValueError(f'Unclear which predictive model to use for {design_name} designs.')
        
            # get edit distances from seed
            designed_n = np.array([editdistance.eval(seed, seq) for seq in designseq_N])

            designp_N, designped_N, q2functionalmus, designmuneg_N = wheelock_mean_forecast(
                ytrain_n, predtrain_n, trained_n, preddesign_n, designed_n, qs=wheelock_forecast_qs
            )

            # record forecast
            for q, (designmutilde_N, designmued_N) in q2functionalmus.items():
                # w/o correction for covariate shift
                forecast_tilde = np.mean(designp_N * designmutilde_N + (1 - designp_N) * designmuneg_N)
                wf_df.loc[i]['wf_mean_q{:.2f}_{}'.format(q, design_name)] = forecast_tilde

                # w/ correction to p and \tilde{\mu} for covariate shift,
                # based on edit distance to the seed sequence
                forecast_ed = np.mean(designped_N * designmued_N + (1 - designped_N) * designmuneg_N)
                wf_df.loc[i]['wf_mean_q{:.2f}_cs_{}'.format(q, design_name)] = forecast_ed
                 
    if imp_csv_fname is not None:
        df.to_csv(imp_csv_fname, index_label='target_value')
        wf_df.to_csv(wheelock_csv_fname, index_label='trial')
        with open(results_pkl_fname, 'wb') as f:
            pickle.dump(name2truemeans, f)
        print('Saved to {}, {}, and {} ({} s).\n'.format(
            imp_csv_fname, wheelock_csv_fname, results_pkl_fname, int(time() - t0)
        ))
    
    format_tokens = '{:.4f} '
    format_str = '  {}: ' + ''.join(n_trial * [format_tokens])
    print('High-ish variance estimates of true mean labels for:')
    for name, truemean_t in name2truemeans.items():
        vmin, vmax = np.min(truemean_t), np.max(truemean_t)
        if (vmax - vmin) / vmax > 0.05 * vmax:
            print(format_str.format(name, *truemean_t))

    return df, wf_df, name2truemeans

def run_imputation_exceedance_selection_experiments(
    design_names,
    threshold: float,
    n_trial: int,
    design_pkl_fname_no_trial: str,
    model_and_data_fname_no_ftype: str,
    target_values: np.array,
    seed: str = RNABinding.SEEDS[3],
    n_hidden: int = 100,
    n_filters: int = 32,
    intermediate_iter_threshold: float = 0.1,
    imp_csv_fname: str = None,
    model_and_data_path: str = '/data/wongfanc/rna-models'  
):

    # ----- load training data and models -----
    train_fname = os.path.join(model_and_data_path, 'traindata-' + model_and_data_fname_no_ftype + '.npz')
    d = np.load(train_fname)
    trainseq_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']

    # get edit distances from seed for Wheelock forecasts
    # t0 = time()
    # trained_n = np.array([editdistance.eval(seed, seq) for seq in trainseq_n])
    # print('Done computing edit distance of {} training sequences to the seed ({} s).'.format(
    #     len(trainseq_n), int(time() - t0)
    # ))

    # fit ridge
    ridge = models.RidgeRegressor(seq_len=50, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)

    # load trained FF and CNN models
    ff_fname = os.path.join(model_and_data_path, 'ff-' + model_and_data_fname_no_ftype + '.pt')
    ff = models.FeedForward(50, RNA_NUCLEOTIDES, n_hidden)
    ff.load(ff_fname)
    cnn_fname = os.path.join(model_and_data_path, 'cnn-' + model_and_data_fname_no_ftype + '.pt')
    cnn = models.CNN(50, RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)
    name2model = {
        'ridge': ridge,
        'ff': ff,
        'cnn': cnn
    }

    # predictions on training data
    name2predtrain = {name: model.predict(trainseq_n) for name, model in name2model.items()}

    # dataframe to record selection experiment results
    imp_selected_column_names = ['tr{}_imp_pval_{}'.format(i, name) for i in range(n_trial) for name in design_names]
    target_values = [round(val, 4) for val in target_values]
    df = DataFrame(index=target_values, columns=imp_selected_column_names)

    # dataframe for Wheelock forecasting results
    # wf_column_names = ['wf_mean_q{:.2f}_{}'.format(q, name) for q in wheelock_forecast_qs for name in design_names]
    # wf_cs_column_names = ['wf_mean_q{:.2f}_cs_{}'.format(q, name) for q in wheelock_forecast_qs for name in design_names]
    # wf_df = DataFrame(index=range(n_trial), columns=wf_column_names + wf_cs_column_names)

    t0 = time()
    name2truemeans = {design_name: [] for design_name in design_names}
    if imp_csv_fname is not None:
        # assert(wheelock_csv_fname is not None)
        results_pkl_fname = imp_csv_fname[: -4] + '-truemeans.pkl'
        
    # ===== run selection experiments =====
    for i in range(n_trial):

        # ----- load design sequences -----
        design_pkl_fname = design_pkl_fname_no_trial + '-{}.pkl'.format(i)
        # make sure designs have labels, add training data, select intermediate C/DbAS ridge iterations
        name2designdata = prepare_name2designdata(  
            design_pkl_fname,
            train_fname,
            intermediate_iter_threshold=intermediate_iter_threshold,
            verbose=False
        )

        assert('train' in name2designdata)
        assert(name2designdata['train'][2] is None)
            
        # name2designdata may contain other distributions used to faciliate DRE,
        # but which we are not interested in designs from
        for name in design_names:
            if name not in name2designdata:
                raise ValueError(f'{name} not in design data at {design_pkl_fname}.')

        # ----- imputation selection experiments -----
        for design_name in design_names:
            (designseq_N, ydesign_n, preddesign_n) = name2designdata[design_name]
            ydesign_n = (ydesign_n >= threshold).astype(float)
            preddesign_n = (preddesign_n >= threshold).astype(float)
            imputed_mean = np.mean(preddesign_n)
            imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)
            name2truemeans[design_name].append(np.mean(ydesign_n))

            for target_val in target_values:
                # get imputation p-value
                imp_pval = _zstat_generic(
                    imputed_mean,
                    0,
                    imputed_se,
                    alternative='larger',
                    diff=target_val
                )[1]

                df.loc[target_val]['tr{}_imp_pval_{}'.format(i, design_name)] = imp_pval

            # ===== Wheelock forecast ===== 
            # assigned_model = False
            # for model_name in name2predtrain.keys():
            #     if model_name in design_name:
            #         predtrain_n = name2predtrain[model_name]
            #         assigned_model = True
            #         break
            # if not assigned_model:
            #     raise ValueError(f'Unclear which predictive model to use for {design_name} designs.')
        
            # # get edit distances from seed
            # designed_n = np.array([editdistance.eval(seed, seq) for seq in designseq_N])

            # designp_N, designped_N, q2functionalmus, designmuneg_N = wheelock_mean_forecast(
            #     ytrain_n, predtrain_n, trained_n, preddesign_n, designed_n, qs=wheelock_forecast_qs
            # )

            # record forecast
            # for q, (designmutilde_N, designmued_N) in q2functionalmus.items():
            #     # w/o correction for covariate shift
            #     forecast_tilde = np.mean(designp_N * designmutilde_N + (1 - designp_N) * designmuneg_N)
            #     wf_df.loc[i]['wf_mean_q{:.2f}_{}'.format(q, design_name)] = forecast_tilde

            #     # w/ correction to p and \tilde{\mu} for covariate shift,
            #     # based on edit distance to the seed sequence
            #     forecast_ed = np.mean(designped_N * designmued_N + (1 - designped_N) * designmuneg_N)
            #     wf_df.loc[i]['wf_mean_q{:.2f}_cs_{}'.format(q, design_name)] = forecast_ed
                 
    if imp_csv_fname is not None:
        df.to_csv(imp_csv_fname, index_label='target_value')
        # wf_df.to_csv(wheelock_csv_fname, index_label='trial')
        with open(results_pkl_fname, 'wb') as f:
            pickle.dump(name2truemeans, f)
        print('Saved to {}, {}, and {} ({} s).\n'.format(
            imp_csv_fname, 'wheelock_csv_fname', results_pkl_fname, int(time() - t0)
        ))
    
    format_tokens = '{:.4f} '
    format_str = '  {}: ' + ''.join(n_trial * [format_tokens])
    print('High-ish variance estimates of true mean labels for:')
    for name, truemean_t in name2truemeans.items():
        vmin, vmax = np.min(truemean_t), np.max(truemean_t)
        if (vmax - vmin) / vmax > 0.05 * vmax:
            print(format_str.format(name, *truemean_t))

    return df, name2truemeans  # wf_df


def run_pp_selection_experiments(
    design_names,
    design_pkl_fname: str,
    model_and_data_fname_no_ftype: str,
    calibration_pkl_fname: str,
    mdre_group_regex_strs,
    target_values: np.array,
    n_trial: int,
    alpha: float = 0.1,
    intermediate_iter_threshold: float = 0.1,
    n_hidden: int = 100,
    n_filters: int = 32,
    use_quadratic_layer_mdre: bool = False,
    n_mdre_hidden: int = 500,
    n_mdre_epoch: int = 100,
    n_forecast_designs: int = 10000,
    n_cal: int = 5000,
    pp_csv_fname: str = None,
    cp_csv_fname: str = None,
    model_and_data_path: str = '/data/wongfanc/rna-models',
    device = None,
):
    # load design sequences
    # and prepare intermediate iterations for C/DbAS
    train_fname = os.path.join(model_and_data_path, 'traindata-' + model_and_data_fname_no_ftype + '.npz')
    name2designdata = prepare_name2designdata(
        design_pkl_fname,
        train_fname,
        intermediate_iter_threshold=intermediate_iter_threshold,
        verbose=False
    )

    assert('train' in name2designdata)
    (trainseqs_n, ytrain_n, predtrain_n) = name2designdata['train']
    assert(predtrain_n is None)

    print('All design names in provided design data:')
    for name in name2designdata:
        print(name)
    print()
        
    # name2designdata may contain other distributions used to faciliate DRE,
    # but which we are not interested in designs from
    n_configuration = len(design_names)
    for name in design_names:
        assert(name in name2designdata)
    
    # compute and save true mean label for each design distribution
    if pp_csv_fname is not None:
        assert(cp_csv_fname is not None)
        name2truemean = {name: np.mean(data[1]) for name, data in name2designdata.items()}
        truemeans_pkl_fname = pp_csv_fname[: -4] + '-truemeans.pkl'
        with open(truemeans_pkl_fname, 'wb') as f:
            pickle.dump(name2truemean, f)
        print(f'Saved true means to {truemeans_pkl_fname}.')

    # ----- load models -----
    # load training data and fit ridge regression
    d = np.load(train_fname)
    trainseqs_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']
    ridge = models.RidgeRegressor(seq_len=50, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseqs_n, ytrain_n)

    # load trained FF and CNN models
    ff_fname = os.path.join(model_and_data_path, 'ff-' + model_and_data_fname_no_ftype + '.pt')
    ff = models.FeedForward(50, RNA_NUCLEOTIDES, n_hidden)
    ff.load(ff_fname)
    cnn_fname = os.path.join(model_and_data_path, 'cnn-' + model_and_data_fname_no_ftype + '.pt')
    cnn = models.CNN(50, RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)
    name2model = {
        'ridge': ridge,
        'ff': ff,
        'cnn': cnn
    }

    # dataframe to record selection experiment results
    pp_column_names = ['tr{}_pp_pval_{}'.format(i, name) for i in range(n_trial) for name in design_names]
    cp_column_names = ['cp_lb_{}'.format(name) for name in design_names]
    cp_nobonf_column_names = ['cp_nobonf_lb_{}'.format(name) for name in design_names]
    qc_column_names = ['qc_forecast_mean_{}'.format(name) for name in design_names]
    target_values = [round(val, 4) for val in target_values]
    df = DataFrame(index=target_values, columns=pp_column_names)
    cp_df = DataFrame(index=range(n_trial), columns=cp_column_names + cp_nobonf_column_names + qc_column_names)

    mdre = MultiMDRE(
        mdre_group_regex_strs,
        device=device
    )

    # load calibration data
    with open(calibration_pkl_fname, 'rb') as f:
        caldata_t = pickle.load(f)

    # fit density ratio estimator (DRE) for all design algorithms
    mdre.fit(
        design_names,
        name2designdata,
        quadratic_final_layer=use_quadratic_layer_mdre,
        n_hidden=n_mdre_hidden,
        n_epoch=n_mdre_epoch,
        verbose=True
    ), # verbose=(t == 0))

    # get (unnormalized) density ratio weights for all N unlabeled examples
    # takes 1s out of the 3s to run get_cp_mean_lb (for one configuration)
    name2dr = {}
    t0 = time()
    name2designmusigma = {}
    for design_name in design_names:
        (designseq_N, _, designpred_n) = name2designdata[design_name]
        designdr_N = mdre.get_dr(designseq_N, design_name, self_normalize=True, verbose=False)
        name2dr[design_name] = designdr_N

        if 'ridge' in design_name:
            name2designmusigma[design_name] = (designpred_n, np.sqrt(ridge.mse) * np.ones(len(designseq_N)))
        else:
            if 'ff' in design_name:
                designpred_nxm = ff.ensemble_predict(designseq_N)
            elif 'cnn' in design_name:
                designpred_nxm = cnn.ensemble_predict(designseq_N)
                
            designmu_n = np.mean(designpred_nxm, axis=1)
            assert(np.max(np.abs(designmu_n - designpred_n)) < 1e-16)
            designstd_n = np.std(designpred_nxm, axis=1)
            name2designmusigma[design_name] = (designmu_n, designstd_n)

    print('Done estimating density ratios for all design sequences. ({} s)'.format(int(time() - t0)))

    # ----- selection experiments -----
    t0 = time()
    for t in range(n_trial):

        # load labeled calibration sequences from training distribution
        calseqs_n, ycal_n = caldata_t[t]
        assert(len(calseqs_n) == n_cal)
        # name2designdata['train'] = (trainseqs_n + calseqs_n, np.hstack([ytrain_n, ycal_n]), None)  # TODO: necessary if only fitting MDRE once?

        # ----- (non-weighted) quantile-calibrated forecast method -----
        # get parameters of forecasted CDF of p(Y | x) for each calibration sequence
        predcalff_nxm = ff.ensemble_predict(calseqs_n)
        predcalcnn_nxm = cnn.ensemble_predict(calseqs_n)
        name2calmusigma = {
            'ridge': (ridge.predict(calseqs_n), np.sqrt(ridge.mse) * np.ones([len(calseqs_n)])),
            'ff': (np.mean(predcalff_nxm, axis=1), np.std(predcalff_nxm, axis=1)),
            'cnn': (np.mean(predcalcnn_nxm, axis=1), np.std(predcalcnn_nxm, axis=1))
        }
        name2predcal = {name: mu_sigma[0] for name, mu_sigma in name2calmusigma.items()}

        # get CDF value for each calibration label
        name2calF_n = {
            name: sc.stats.norm.cdf(ycal_n, loc=mu_sigma[0], scale=mu_sigma[1]) for name, mu_sigma in name2calmusigma.items()
        }

        # get empirical CDF corresponding to each forecasted CDF value
        name2calempF_n = {
            name: np.mean(calF_n[:, None] <= calF_n[None, :], axis=0, keepdims=False) for name, calF_n in name2calF_n.items()
        }

        # calibrate forecasts
        name2ir = {}
        for name in ['ridge', 'ff', 'cnn']:
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            calF_n = name2calF_n[name]
            calempF_n = name2calempF_n[name]
            ir.fit(calF_n, calempF_n)
            name2ir[name] = ir

        for design_name in design_names:
            (designseq_N, _, preddesign_N) = name2designdata[design_name]

            # ----- quantities for prediction-powered test and CP-based baseline -----
            imputed_mean = np.mean(preddesign_N)
            imputed_se = np.std(preddesign_N) / np.sqrt(preddesign_N.size)

            # predictions for calibration sequences
            assigned_model = False
            for model_name in name2predcal.keys():
                if model_name in design_name:
                    predcal_n = name2predcal[model_name]
                    assigned_model = True
                    break
            if not assigned_model:
                raise ValueError(f'Unclear which predictive model to use for {design_name} designs.')

            # DRs for calibration sequences
            caldr_n = mdre.get_dr(calseqs_n, design_name, self_normalize=True, verbose=False)

            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal_n)
            rectifier_mean = np.mean(rect_n)
            rectifier_se = np.std(rect_n) / np.sqrt(rect_n.size)
            
            for target_val in target_values:

                # get prediction-powered p-value
                pp_pval = rectified_p_value(
                    rectifier_mean,
                    rectifier_se,
                    imputed_mean,
                    imputed_se,
                    null=target_val,
                    alternative='larger'
                )
                df.loc[target_val]['tr{}_pp_pval_{}'.format(t, design_name)] = pp_pval
            
            # subsample designs for CP and calibration baselines
            forecast_idx = np.random.choice(len(designseq_N), size=n_forecast_designs, replace=False)


            # ===== quantile-calibrated forecasts =====
            predmu_n, predsigma_n = name2designmusigma[design_name]
            predmu_n = predmu_n[forecast_idx]
            predsigma_n = predsigma_n[forecast_idx]

            ir_assigned = False
            for model_name in ['ridge', 'ff', 'cnn']:
                if model_name in design_name:
                    ir = name2ir[model_name]
                    ir_assigned = True
                    break
            assert(ir_assigned)
            
            qcmu_N = utils.get_mean_from_cdf(
                predmu_n,
                predsigma_n,
                ir,
                (0, 0.6),
                (-0.1, 0),
                quad_limit=200,
                err_norm='max'
            )
            cp_df.loc[t, f'qc_forecast_mean_{design_name}'] = np.mean(qcmu_N)


            # ===== CP =====
            # conformal prediction-based lower bound
            designdr_N = name2dr[design_name][forecast_idx]
            lb, lb_nobonf = get_conformal_prediction_lower_bound(
                ycal_n, predcal_n, caldr_n, preddesign_N[forecast_idx], designdr_N, alpha=alpha / n_configuration
            )
            # test outcomes (reject/fail to reject)
            cp_df.loc[t, f'cp_lb_{design_name}'] = lb
            cp_df.loc[t, f'cp_nobonf_lb_{design_name}'] = lb_nobonf
            if lb_nobonf > -np.inf:
                print('{} has LBs {:.4f}, {:.4f}'.format(design_name, lb, lb_nobonf))


        print('Done running {} / {} trials ({} s).'.format(t + 1, n_trial, int(time() - t0)))
        if pp_csv_fname is not None:
            df.to_csv(pp_csv_fname, index_label='target_value')  # time sink
            cp_df.to_csv(cp_csv_fname, index_label='trial')
            print('Saved to {} and {} ({} s).\n'.format(pp_csv_fname, cp_csv_fname, int(time() - t0)))

    return df

# TODO
def run_pp_exceedance_selection_experiments(
    threshold: float,
    design_names,
    design_pkl_fname: str,
    model_and_data_fname_no_ftype: str,
    calibration_pkl_fname: str,
    mdre_group_regex_strs,
    target_values: np.array,
    n_trial: int,
    alpha: float = 0.1,
    intermediate_iter_threshold: float = 0.1,
    n_hidden: int = 100,
    n_filters: int = 32,
    use_quadratic_layer_mdre: bool = False,
    n_mdre_hidden: int = 500,
    n_mdre_epoch: int = 100,
    n_forecast_designs: int = 10000,
    n_cal: int = 5000,
    pp_csv_fname: str = None,
    cp_csv_fname: str = None,
    model_and_data_path: str = '/data/wongfanc/rna-models',
    device = None,
):
    # load design sequences
    # and prepare intermediate iterations for C/DbAS
    train_fname = os.path.join(model_and_data_path, 'traindata-' + model_and_data_fname_no_ftype + '.npz')
    name2designdata = prepare_name2designdata(
        design_pkl_fname,
        train_fname,
        intermediate_iter_threshold=intermediate_iter_threshold,
        verbose=False
    )

    assert('train' in name2designdata)
    (trainseqs_n, ytrain_n, predtrain_n) = name2designdata['train']
    assert(predtrain_n is None)

    print('All design names in provided design data:')
    for name in name2designdata:
        print(name)
    print()
        
    # name2designdata may contain other distributions used to faciliate DRE,
    # but which we are not interested in designs from
    n_configuration = len(design_names)
    for name in design_names:
        assert(name in name2designdata)
    
    # compute and save true mean label for each design distribution
    if pp_csv_fname is not None:
        assert(cp_csv_fname is not None)
        name2truemean = {name: np.mean(data[1]) for name, data in name2designdata.items()}
        truemeans_pkl_fname = pp_csv_fname[: -4] + '-truemeans.pkl'
        with open(truemeans_pkl_fname, 'wb') as f:
            pickle.dump(name2truemean, f)
        print(f'Saved true means to {truemeans_pkl_fname}.')

    # ----- load models -----
    # load training data and fit ridge regression
    d = np.load(train_fname)
    trainseqs_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']
    ridge = models.RidgeRegressor(seq_len=50, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseqs_n, ytrain_n)

    # load trained FF and CNN models
    ff_fname = os.path.join(model_and_data_path, 'ff-' + model_and_data_fname_no_ftype + '.pt')
    ff = models.FeedForward(50, RNA_NUCLEOTIDES, n_hidden)
    ff.load(ff_fname)
    cnn_fname = os.path.join(model_and_data_path, 'cnn-' + model_and_data_fname_no_ftype + '.pt')
    cnn = models.CNN(50, RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)
    name2model = {
        'ridge': ridge,
        'ff': ff,
        'cnn': cnn
    }

    # dataframe to record selection experiment results
    pp_column_names = ['tr{}_pp_pval_{}'.format(i, name) for i in range(n_trial) for name in design_names]
    cp_column_names = ['cp_lb_{}'.format(name) for name in design_names]
    cp_nobonf_column_names = ['cp_nobonf_lb_{}'.format(name) for name in design_names]
    qc_column_names = ['qc_forecast_mean_{}'.format(name) for name in design_names]
    target_values = [round(val, 4) for val in target_values]
    df = DataFrame(index=target_values, columns=pp_column_names)
    cp_df = DataFrame(index=range(n_trial), columns=cp_column_names + cp_nobonf_column_names + qc_column_names)

    mdre = MultiMDRE(
        mdre_group_regex_strs,
        device=device
    )

    # load calibration data
    with open(calibration_pkl_fname, 'rb') as f:
        caldata_t = pickle.load(f)

    # fit density ratio estimator (DRE) for all design algorithms
    mdre.fit(
        design_names,
        name2designdata,
        quadratic_final_layer=use_quadratic_layer_mdre,
        n_hidden=n_mdre_hidden,
        n_epoch=n_mdre_epoch,
        verbose=True
    ), # verbose=(t == 0))

    # get (unnormalized) density ratio weights for all N unlabeled examples
    # takes 1s out of the 3s to run get_cp_mean_lb (for one configuration)
    name2dr = {}
    t0 = time()
    name2designmusigma = {}
    for design_name in design_names:
        (designseq_N, _, designpred_n) = name2designdata[design_name]
        designdr_N = mdre.get_dr(designseq_N, design_name, self_normalize=True, verbose=False)
        name2dr[design_name] = designdr_N

        if 'ridge' in design_name:
            name2designmusigma[design_name] = (designpred_n, np.sqrt(ridge.mse) * np.ones(len(designseq_N)))
        else:
            if 'ff' in design_name:
                designpred_nxm = ff.ensemble_predict(designseq_N)
            elif 'cnn' in design_name:
                designpred_nxm = cnn.ensemble_predict(designseq_N)
                
            designmu_n = np.mean(designpred_nxm, axis=1)
            assert(np.max(np.abs(designmu_n - designpred_n)) < 1e-16)
            designstd_n = np.std(designpred_nxm, axis=1)
            name2designmusigma[design_name] = (designmu_n, designstd_n)

    print('Done estimating density ratios for all design sequences. ({} s)'.format(int(time() - t0)))

    # ----- selection experiments -----
    t0 = time()
    for t in range(n_trial):

        # load labeled calibration sequences from training distribution
        calseqs_n, ycal_n = caldata_t[t]
        assert(len(calseqs_n) == n_cal)
        # name2designdata['train'] = (trainseqs_n + calseqs_n, np.hstack([ytrain_n, ycal_n]), None)  # TODO: necessary if only fitting MDRE once?

        # ----- (non-weighted) quantile-calibrated forecast method -----
        # get parameters of forecasted CDF of p(Y | x) for each calibration sequence
        predcalff_nxm = ff.ensemble_predict(calseqs_n)
        predcalcnn_nxm = cnn.ensemble_predict(calseqs_n)
        name2calmusigma = {
            'ridge': (ridge.predict(calseqs_n), np.sqrt(ridge.mse) * np.ones([len(calseqs_n)])),
            'ff': (np.mean(predcalff_nxm, axis=1), np.std(predcalff_nxm, axis=1)),
            'cnn': (np.mean(predcalcnn_nxm, axis=1), np.std(predcalcnn_nxm, axis=1))
        }
        name2predcal = {name: mu_sigma[0] for name, mu_sigma in name2calmusigma.items()}

        # get CDF value for each calibration label
        name2calF_n = {
            name: sc.stats.norm.cdf(ycal_n, loc=mu_sigma[0], scale=mu_sigma[1]) for name, mu_sigma in name2calmusigma.items()
        }

        # get empirical CDF corresponding to each forecasted CDF value
        name2calempF_n = {
            name: np.mean(calF_n[:, None] <= calF_n[None, :], axis=0, keepdims=False) for name, calF_n in name2calF_n.items()
        }

        # calibrate forecasts
        name2ir = {}
        for name in ['ridge', 'ff', 'cnn']:
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            calF_n = name2calF_n[name]
            calempF_n = name2calempF_n[name]
            ir.fit(calF_n, calempF_n)
            name2ir[name] = ir

        for design_name in design_names:
            (designseq_N, _, preddesign_N) = name2designdata[design_name]

            # ----- quantities for prediction-powered test and CP-based baseline -----
            imputed_mean = np.mean(preddesign_N)
            imputed_se = np.std(preddesign_N) / np.sqrt(preddesign_N.size)

            # predictions for calibration sequences
            assigned_model = False
            for model_name in name2predcal.keys():
                if model_name in design_name:
                    predcal_n = name2predcal[model_name]
                    assigned_model = True
                    break
            if not assigned_model:
                raise ValueError(f'Unclear which predictive model to use for {design_name} designs.')

            # DRs for calibration sequences
            caldr_n = mdre.get_dr(calseqs_n, design_name, self_normalize=True, verbose=False)

            # rectifier sample mean and standard error
            rect_n = caldr_n * (ycal_n - predcal_n)
            rectifier_mean = np.mean(rect_n)
            rectifier_se = np.std(rect_n) / np.sqrt(rect_n.size)
            
            for target_val in target_values:

                # get prediction-powered p-value
                pp_pval = rectified_p_value(
                    rectifier_mean,
                    rectifier_se,
                    imputed_mean,
                    imputed_se,
                    null=target_val,
                    alternative='larger'
                )
                df.loc[target_val]['tr{}_pp_pval_{}'.format(t, design_name)] = pp_pval
            
            # subsample designs for CP and calibration baselines
            forecast_idx = np.random.choice(len(designseq_N), size=n_forecast_designs, replace=False)


            # ===== quantile-calibrated forecasts =====
            predmu_n, predsigma_n = name2designmusigma[design_name]
            predmu_n = predmu_n[forecast_idx]
            predsigma_n = predsigma_n[forecast_idx]

            ir_assigned = False
            for model_name in ['ridge', 'ff', 'cnn']:
                if model_name in design_name:
                    ir = name2ir[model_name]
                    ir_assigned = True
                    break
            assert(ir_assigned)
            
            qcmu_N = utils.get_mean_from_cdf(
                predmu_n,
                predsigma_n,
                ir,
                (0, 0.6),
                (-0.1, 0),
                quad_limit=200,
                err_norm='max'
            )
            cp_df.loc[t, f'qc_forecast_mean_{design_name}'] = np.mean(qcmu_N)


            # ===== CP =====
            # conformal prediction-based lower bound
            designdr_N = name2dr[design_name][forecast_idx]
            lb, lb_nobonf = get_conformal_prediction_lower_bound(
                ycal_n, predcal_n, caldr_n, preddesign_N[forecast_idx], designdr_N, alpha=alpha / n_configuration
            )
            # test outcomes (reject/fail to reject)
            cp_df.loc[t, f'cp_lb_{design_name}'] = lb
            cp_df.loc[t, f'cp_nobonf_lb_{design_name}'] = lb_nobonf
            if lb_nobonf > -np.inf:
                print('{} has LBs {:.4f}, {:.4f}'.format(design_name, lb, lb_nobonf))


        print('Done running {} / {} trials ({} s).'.format(t + 1, n_trial, int(time() - t0)))
        if pp_csv_fname is not None:
            df.to_csv(pp_csv_fname, index_label='target_value')  # time sink
            cp_df.to_csv(cp_csv_fname, index_label='trial')
            print('Saved to {} and {} ({} s).\n'.format(pp_csv_fname, cp_csv_fname, int(time() - t0)))

    return df