import os
import pickle
from time import time

import numpy as np
import scipy as sc
from pandas import DataFrame, Series, read_csv
from statsmodels.stats.weightstats import _zstat_generic

try:
    import RNA
except ImportError:
    pass

import models
import designers
from dre import MultiMDRE
import utils
from utils import RNA_NUCLEOTIDES, RNANUC2COMPLEMENT, get_mutant, get_conformal_prediction_lower_bound, editdistance, gmm_mean_forecast


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


# no intermediate iteration weirdness
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


def sample_design_sequences_icml_cdbas(
    n_design: int,
    cbas_dbas_quantiles,
    latent_dims,
    n_vae_hiddens,
    model_and_data_fname_no_ftype: str,
    model_and_data_path: str = '/data/wongfanc/rna-models',
    design_pkl_fname: str = None,
    n_hidden: int = 100,
    n_filters: int = 32,
    max_model_queries: int = None,
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
        for latent_dim in latent_dims:
            for n_vae_hidden in n_vae_hiddens:

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
                    print(f'Designing CbAS {quantile} {model_name} {latent_dim} {n_vae_hidden} sequences...')
                    t0 = time()
                    cbas_n = cbas.design_sequences(
                        n_design,
                        quantile=quantile
                    )
                    print(f'  Done. ({int(time() - t0)} s)')
                    # store
                    predcbas_n = model.predict(cbas_n)
                    print('  Mean prediction: {:.3f}'.format(np.mean(predcbas_n)))
                    name2designs[f'cbas-{model_name}-{quantile}-{latent_dim}-{n_vae_hidden}'] = (cbas_n, None, predcbas_n)
                    # save
                    if design_pkl_fname is not None:
                        with open(design_pkl_fname, 'wb') as f:
                            pickle.dump(name2designs, f)
                        print(f'  Saved {n_design} CbAS {quantile} {model_name} latent {latent_dim} hidden {n_vae_hidden} sequences.')
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
                    print(f'Designing DbAS {quantile} {model_name} {latent_dim} {n_vae_hidden} sequences...')
                    t0 = time()
                    dbas_n = dbas.design_sequences(
                        n_design,
                        quantile=quantile
                    )
                    print(f'  Done. ({int(time() - t0)} s)')
                    # store
                    preddbas_n = model.predict(dbas_n)
                    print('  Mean prediction: {:.3f}'.format(np.mean(preddbas_n)))
                    name2designs[f'dbas-{model_name}-{quantile}-{latent_dim}-{n_vae_hidden}'] = (dbas_n, None, preddbas_n)
                    # save
                    if design_pkl_fname is not None:
                        with open(design_pkl_fname, 'wb') as f:
                            pickle.dump(name2designs, f)
                        print(f'  Saved {n_design} DbAS {quantile} {model_name} latent {latent_dim} hidden {n_vae_hidden} sequences.')
                    print()


    return name2designs

def sample_design_sequences_icml_adalead(
    n_design: int,
    adalead_thresholds,
    recomb_rates,
    mutation_rates,
    model_and_data_fname_no_ftype: str,
    model_and_data_path: str = '/data/wongfanc/rna-models',
    design_pkl_fname: str = None,
    n_hidden: int = 100,
    n_filters: int = 32,
    max_model_queries: int = None,
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
        if model_name != 'cnn':
            continue
        
        # ===== AdaLead =====
        adalead = designers.AdaLead(model, trainseq_n, ytrain_n)
        for threshold in adalead_thresholds:

            for recomb_rate in recomb_rates:

                for mutation_rate in mutation_rates:

                    threshold = round(threshold, 4)
                    print(f'Designing AdaLead threshold = {threshold}, recomb = {recomb_rate}, mutation = {mutation_rate}, {model_name} sequences...')
                    
                    # design sequences
                    t0 = time()
                    adalead_n = adalead.design_sequences(
                        n_design=n_design,
                        threshold=threshold,
                        recomb_rate=recomb_rate,
                        mutation_rate=mutation_rate,
                        max_model_queries=max_model_queries
                    )
                    print(f'  Done. ({int(time() - t0)} s)')
                    
                    # store
                    predadalead_n = model.predict(adalead_n)
                    # yadalead_n = landscape.get_fitness(adalead_n)
                    print('  Mean prediction: {:.3f}'.format(np.mean(predadalead_n)))
                    name2designs[f'adalead-{model_name}-{threshold}-{recomb_rate}-{mutation_rate}'] = (adalead_n, None, predadalead_n)
                    
                    # save
                    if design_pkl_fname is not None:
                        with open(design_pkl_fname, 'wb') as f:
                            pickle.dump(name2designs, f)
                        print(f'  Saved {n_design} AdaLead threshold = {threshold}, recomb = {recomb_rate}, mutation = {mutation_rate}, {model_name} sequences.')
                    print()

    return name2designs


def select_for_mean_without_labeled_data(
    design_names,
    design_pkl_fnames,
    desired_values: np.array,
    seed: str = RNABinding.SEEDS[3],
    n_hidden: int = 100,
    n_filters: int = 32,
    po_csv_fname: str = None,
    gmm_csv_fname: str = None,
    gmm_qs = None,
    train_fname: str = 'rna-models/rna-train-data-10k.npz',
    fc_fname: str = 'rna-models/fc-10k.pt',
    cnn_fname: str = 'rna-models/cnn-10k.pt',
):
    
    if po_csv_fname is not None:
        assert(gmm_csv_fname is not None)
    if gmm_csv_fname is not None:
        assert(po_csv_fname is not None)

    if gmm_qs is None:
        gmm_qs = utils.DEFAULT_GMM_QS

    # ===== load training data and models =====
    d = np.load(train_fname)
    trainseq_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']

    # get ridge regression, ensemble of FC, and ensemble of CNN models
    ridge = models.RidgeRegressor(seq_len=50, alphabet=utils.RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)
    fc = models.FeedForward(50, utils.RNA_NUCLEOTIDES, n_hidden)
    fc.load(fc_fname)
    cnn = models.CNN(50, utils.RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)
    name2model = {
        'ridge': ridge,
        'fc': fc,
        'cnn': cnn
    }

    # get predictions on training sequences
    name2predtrain = {name: model.predict(trainseq_n) for name, model in name2model.items()}

    # get training sequences' edit distances from seed for GMMForecasts
    trained_n = np.array([editdistance.eval(seed, seq) for seq in trainseq_n])

    # ===== initialize dataframes for results =====

    # TODO: copy over existing results, clean those column names
    # dataframe for PO results
    n_trial = len(design_pkl_fnames)
    po_column_names = ['tr{}_po_pval_{}'.format(i, name) for i in range(n_trial) for name in design_names]
    desired_values = [round(val, 4) for val in desired_values]
    po_df = DataFrame(index=desired_values, columns=po_column_names)

    # dataframe for Wheelock forecasting results
    gmm_column_names = ['gmm-q{:.2f}_mean_{}'.format(q, name) for q in gmm_qs for name in design_names]
    gmm_cs_column_names = ['gmm-cs-q{:.2f}_mean_{}'.format(q, name) for q in gmm_qs for name in design_names]
    gmm_df = DataFrame(index=range(n_trial), columns=gmm_column_names + gmm_cs_column_names)
            
    # ===== selection experiments =====
    t0 = time()
    for i, design_pkl_fname in enumerate(design_pkl_fnames):

        # load design sequences from all configurations
        with open(design_pkl_fname, 'rb') as f:
            name2designdata = pickle.load(f)
        assert(name2designdata.keys() == set(design_names))

        for design_name in design_names:
            (designseq_N, ydesign_n, preddesign_n) = name2designdata[design_name]
            imputed_mean = np.mean(preddesign_n)
            imputed_se = np.std(preddesign_n) / np.sqrt(preddesign_n.size)
            
            # ===== PO p-value =====
            for tau in desired_values:
                po_pval = _zstat_generic(
                    imputed_mean,
                    0,
                    imputed_se,
                    alternative='larger',
                    diff=tau
                )[1]
                po_df.loc[tau]['tr{}_po_pval_{}'.format(i, design_name)] = po_pval

            # ===== GMMForecasts =====
            # get predictive model used by this configuration,
            # and predictions for the training sequences
            got_model = False
            for model_name in name2predtrain.keys():
                if model_name in design_name:
                    predtrain_n = name2predtrain[model_name]
                    got_model = True
                    break
            if not got_model:
                raise ValueError(f'Unclear which predictive model used by {design_name}.')
        
            # get design sequences' edit distances from seed sequence
            designed_N = np.array([editdistance.eval(seed, seq) for seq in designseq_N])

            # get GMM forecasts for each designed sequence
            designp_N, designped_N, q2functionalmus, designmuneg_N = gmm_mean_forecast(
                ytrain_n, predtrain_n, trained_n, preddesign_n, designed_N, qs=gmm_qs
            )

            # vary GMMForecasts hyperparameter q
            for q, (designmutilde_N, designmued_N) in q2functionalmus.items():
                # w/o correction for covariate shift
                forecast_tilde = np.mean(designp_N * designmutilde_N + (1 - designp_N) * designmuneg_N)
                gmm_df.loc[i]['gmm-q{:.2f}_mean_{}'.format(q, design_name)] = forecast_tilde

                # w/ correction to p and \tilde{\mu} for covariate shift,
                # based on edit distance to the seed sequence
                forecast_ed = np.mean(designped_N * designmued_N + (1 - designped_N) * designmuneg_N)
                gmm_df.loc[i]['gmm-cs-q{:.2f}_mean_{}'.format(q, design_name)] = forecast_ed
            
        print('Done with trial {} / {} ({} s).'.format(i + 1, n_trial, int(time() - t0)))
    
    # save results
    if po_csv_fname is not None:
        po_df.to_csv(po_csv_fname, index_label='target_value')
        gmm_df.to_csv(gmm_csv_fname, index_label='trial')
        print('Saved to {} and {} ({} s).\n'.format(po_csv_fname, gmm_csv_fname, int(time() - t0)))

    return po_df, gmm_df


def select_for_mean_with_labeled_data(
    design_names,
    design_pkl_fname: str,
    desired_values: np.array,
    mdre_group_regex_strs,
    n_trial: int = 200,
    n_hidden: int = 100,
    n_filters: int = 32,
    use_quadratic_layer_mdre: bool = False,
    n_mdre_hidden: int = 500,
    n_mdre_epoch: int = 100,
    n_heldout: int = 5000,
    n_design_forecasts: int = 10000,
    quad_limit: int = 100,
    train_fname: str = 'rna-models/rna-train-data-5k.npz',
    heldout_pkl_fname: str = 'rna-models/rna-heldout-5k.pkl',
    fc_fname: str = 'rna-models/fc-5k.pt',
    cnn_fname: str = 'rna-models/cnn-5k.pt',
    pp_csv_fname: str = None,
    cf_csv_fname: str = None,
    device = None,
):
    if pp_csv_fname is not None:
        assert(cf_csv_fname is not None)
    if cf_csv_fname is not None:
        assert(pp_csv_fname is not None)

    # ===== load training data and models =====
    d = np.load(train_fname)
    trainseq_n = list(d['trainseq_n'])
    ytrain_n = d['ytrain_n']

    # load held-out labeled data
    with open(heldout_pkl_fname, 'rb') as f:
        heldout_t = pickle.load(f)

    # get ridge regression, ensemble of FC, and ensemble of CNN models
    ridge = models.RidgeRegressor(seq_len=50, alphabet=utils.RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)
    fc = models.FeedForward(50, utils.RNA_NUCLEOTIDES, n_hidden)
    fc.load(fc_fname)
    cnn = models.CNN(50, utils.RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)
    name2model = {
        'ridge': ridge,
        'fc': fc,
        'cnn': cnn
    }
    
    # compute and save true mean label for each design distribution
    # name2truemean = {name: np.mean(data[1]) for name, data in name2designdata.items()}
    # truemeans_pkl_fname = pp_csv_fname[: -4] + '-truemeans.pkl'
    # with open(truemeans_pkl_fname, 'wb') as f:
    #     pickle.dump(name2truemean, f)
    # print(f'Saved true means to {truemeans_pkl_fname}.')

    # ===== load designed sequences from all configurations =====
    with open(design_pkl_fname, 'rb') as f:
        name2designdata = pickle.load(f)
    assert(name2designdata.keys() == set(design_names))

    # ===== density ratio estimation =====
    # fit multinomial logistic regression-based density ratio estimation (MDRE) model for all configurations
    # mdre = MultiMDRE(mdre_group_regex_strs, device=device)
    # name2designdata['train'] = (trainseq_n, ytrain_n, None)
    # mdre.fit(
    #     design_names,
    #     name2designdata,
    #     quadratic_final_layer=use_quadratic_layer_mdre,
    #     n_hidden=n_mdre_hidden,
    #     n_epoch=n_mdre_epoch,
    #     verbose=True
    # )

    # ===== initialize dataframes to record selection experiment results =====
    desired_values = [round(val, 4) for val in desired_values]
    pp_column_names = ['tr{}_pp_pval_{}'.format(i, name) for i in range(n_trial) for name in design_names]
    pp_df = DataFrame(index=desired_values, columns=pp_column_names)
    cf_column_names = ['cf_mean_{}'.format(name) for name in design_names]
    cf_df = DataFrame(index=range(n_trial), columns=cf_column_names)


    # ===== run selection experiments =====
    t0 = time()
    for t in range(n_trial):

        # load held-out labeled sequences
        labseqs_n, ylab_n = heldout_t[t]
        assert(len(labseqs_n) == n_heldout)

        # ===== CalibratedForecasts method =====
        # get parameters of forecasted CDF of p(Y | x) for each held-out labeled sequence
        predlabfc_nxm = fc.ensemble_predict(labseqs_n)
        predlabcnn_nxm = cnn.ensemble_predict(labseqs_n)
        name2labmusigma = {
            'ridge': (ridge.predict(labseqs_n), np.sqrt(ridge.mse) * np.ones([len(labseqs_n)])),
            'fc': (np.mean(predlabfc_nxm, axis=1), np.std(predlabfc_nxm, axis=1)),
            'cnn': (np.mean(predlabcnn_nxm, axis=1), np.std(predlabcnn_nxm, axis=1))
        }
        name2predlab = {name: mu_sigma[0] for name, mu_sigma in name2labmusigma.items()}

        # evaluate forecast CDF for each held-out label
        name2calF_n = {
            name: sc.stats.norm.cdf(ylab_n, loc=mu_sigma[0], scale=mu_sigma[1]) for name, mu_sigma in name2labmusigma.items()
        }

        # evaluate empirical CDF over held-out labeled data
        name2calempF_n = {
            name: np.mean(calF_n[:, None] <= calF_n[None, :], axis=0, keepdims=False) for name, calF_n in name2calF_n.items()
        }

        # fit monotonic transformations to calibrate forecasts
        name2ir = {}
        for name in ['ridge', 'fc', 'cnn']:
            ir = utils.IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            calF_n = name2calF_n[name]
            calempF_n = name2calempF_n[name]
            ir.fit(calF_n, calempF_n)
            name2ir[name] = ir


        for design_name in design_names:
            (designseq_N, ydesign_N, preddesign_N, designsigma_N) = name2designdata[design_name]
            imputed_mean = np.mean(preddesign_N)
            imputed_se = np.std(preddesign_N) / np.sqrt(preddesign_N.size)

            # predictions for held-out labeled sequences
            got_model = False
            for model_name in name2predlab.keys():
                if model_name in design_name:
                    predlab_n = name2predlab[model_name]
                    got_model = True
                    break
            if not got_model:
                raise ValueError(f'Unclear which predictive model used by {design_name}.')
            
            # ===== our method =====
            # estimate DRs for held-out labeled sequences
            # labdr_n = mdre.get_dr(labseqs_n, design_name, self_normalize=True, verbose=False)

            # # rectifier sample mean and standard error
            # rect_n = labdr_n * (ylab_n - predlab_n)
            # rectifier_mean = np.mean(rect_n)
            # rectifier_se = np.std(rect_n) / np.sqrt(rect_n.size)
            
            # # get prediction-powered p-value
            # for val in desired_values:
            #     pp_pval = utils.rectified_p_value(
            #         rectifier_mean,
            #         rectifier_se,
            #         imputed_mean,
            #         imputed_se,
            #         null=val,
            #         alternative='larger'
            #     )
            #     pp_df.loc[val]['tr{}_pp_pval_{}'.format(t, design_name)] = pp_pval
            
            # ===== CalibratedForecasts method =====
            # subsample designs for speed
            forecast_idx = np.random.choice(len(designseq_N), size=n_design_forecasts, replace=False)
            designmu_N = preddesign_N[forecast_idx]
            designsigma_N = designsigma_N[forecast_idx]
            lower_int_limit = np.fmax(np.min(designmu_N - 3 * designsigma_N), 0)
            upper_int_limit = np.fmin(np.max(designmu_N + 3 * designsigma_N), 1)

            got_ir = False
            for model_name in ['ridge', 'fc', 'cnn']:
                if model_name in design_name:
                    ir = name2ir[model_name]
                    got_ir = True
                    break
            assert(got_ir)

            qcmu_N, t1_err, t2_err = utils.get_mean_from_cdf(
                designmu_N,
                designsigma_N,
                ir,
                (0, 1), # (lower_int_limit, upper_int_limit),
                None,
                quad_limit=quad_limit,
                err_norm='max',
                # tol=tol,
            )
            cf_df.loc[t, f'cf_mean_{design_name}'] = np.mean(qcmu_N)
            print('done w/ {} ({:.3f}, true {:.3f}), lower_int_limit {:.3f}, upper_int_limit {:.3f}, t1_err {:.4f}, {} s'.format(
                design_name, np.mean(qcmu_N), np.mean(ydesign_N), lower_int_limit, upper_int_limit, t1_err, int(time() - t0)))
            
        print('Done running {} / {} trials ({} s).'.format(t + 1, n_trial, int(time() - t0)))
        if pp_csv_fname is not None:
            pp_df.to_csv(pp_csv_fname, index_label='target_value')
            cf_df.to_csv(cf_csv_fname, index_label='trial')
            print('Saved to {} and {} ({} s).\n'.format(pp_csv_fname, cf_csv_fname, int(time() - t0)))

    return pp_df, cf_df


def process_pvalues_for_plotting(
    df,
    desired_values,
    design_names,
    truemeans_fname: str,
    n_trial: int,
    method_name: str = 'po',  # 'po' or 'pp'
    alpha: float = 0.1
):

    n_config = len(design_names)
    alpha_bonferroni = alpha / n_config
    print('Processing {} results with {} configurations, {} desired values in [{:.2f}, {:.2f}], {} trials, and alpha = {:.1f}'.format(
        method_name, n_config, desired_values.shape[0], np.min(desired_values), np.max(desired_values),
        n_trial, alpha
    ))
    truemeans_df = read_csv(truemeans_fname, index_col=0)

    worst_v = []  # worst (i.e. lowest) mean design label achieved by any selected configuration, for each desired value (tau)
    err_v = []    # error rate, for each desired value
    sel_v = []    # selection rate, for each desired value

    t0 = time()
    for val in desired_values:
        val = round(val, 4)
            
        worst_t = []  # worst (i.e. lowest) mean label for trials where a configuration was selected
        
        for i in range(n_trial):
            selected = [name for name in design_names if df.loc[val]['tr{}_{}_pval_{}'.format(i, method_name, name)] < alpha_bonferroni]
            achieved = [truemeans_df.loc[name]['mean_design_label'] for name in selected]

            if len(selected):
                worst_t.append(np.min(achieved))
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        sel_v.append(len(worst_t) / n_trial)

    print('Done processing ({} s)'.format(int(time() - t0)))
    return err_v, sel_v, worst_v


def process_gmmforecasts_for_plotting(
    gmm_df,
    desired_values,
    design_names,
    truemeans_fname: str,
    n_trial: int,
    gmm_qs = None,
    covariate_shift: bool = True,
):
    if gmm_qs is None:
        gmm_qs = utils.DEFAULT_GMM_QS

    print('Processing GMMForecasts ({}) results with {} configurations, {} desired values in [{:.2f}, {:.2f}], and {} trials'.format(
        'w/ cov. shift' if covariate_shift else 'w/o cov. shift', len(design_names),
        desired_values.shape[0], np.min(desired_values), np.max(desired_values), n_trial
    ))

    q2results = {}
    t0 = time()
    for q in gmm_qs:
        method_name = 'gmm-{}q{:.2f}'.format('cs-' if covariate_shift else '', q)
        err_v, sel_v, worst_v = process_forecasts_for_plotting(
            gmm_df, desired_values, design_names, truemeans_fname, n_trial, method_name=method_name
        )
        q2results[q] = err_v, sel_v, worst_v
    print('Done processing ({} s)'.format(int(time() - t0)))
    return q2results


def process_forecasts_for_plotting(
    df,
    desired_values,
    design_names,
    truemeans_fname,
    n_trial: int,
    method_name: str,
):
    desired_values = [round(val, 4) for val in desired_values]
    val2selected = {val: [] for val in desired_values}
    truemeans_df = read_csv(truemeans_fname, index_col=0)

    # for each desired value, collect the selected configurations
    for i in range(n_trial):
        for val in desired_values:
            val2selected[val].append([])

        for name in design_names:
            forecast = df.loc[i]['{}_mean_{}'.format(method_name, name)]

            for val in desired_values:
                if forecast >= val:
                    val2selected[val][i].append(name)

    worst_v = []  # worst (i.e. lowest) mean design label achieved by any selected configuration, for each desired value (tau)
    err_v = []    # error rate, for each desired value
    sel_v = []    # selection rate, for each desired value
    for val in desired_values:                    
        worst_t = []    # worst (i.e. lowest) mean design label for each trial
        for i in range(n_trial):
            achieved = [truemeans_df.loc[name]['mean_design_label'] for name in val2selected[val][i]]
            if len(achieved):
                worst_t.append(np.min(achieved))
                
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        sel_v.append(len(worst_t) / n_trial)
                        
    return err_v, sel_v, worst_v