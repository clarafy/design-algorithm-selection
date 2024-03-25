import os
import pickle

from time import time
import numpy as np
import scipy as sc

try:
    import RNA
except ImportError:
    pass

import models
import designers

from utils import RNA_NUCLEOTIDES, get_mutant

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

    SEQ_LEN = 50

    def __init__(
        self,
        binding_target_idx: int = 0,
    ):
        """
        Create an RNABinding landscape.

        Args:
            binding_target_idx:

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
        
        self.target = self.BINDING_TARGETS[binding_target_idx]
        self.norm_value = self.compute_min_binding_energy()

    def compute_min_binding_energy(self):
        """Compute the lowest possible binding energy for the target."""
        complements = {"A": "U", "C": "G", "G": "C", "U": "A"}

        complement = "".join(complements[x] for x in self.target)[::-1]
        energy = RNA.duplexfold(complement, self.target).energy
        return energy * self.SEQ_LEN / len(self.target)

    def get_fitness(self, sequences):
        fitnesses = []

        for seq in sequences:

            if len(seq) != self.SEQ_LEN:
                raise ValueError('All sequences in `sequences` must be of length {self.SEQ_LEN}.')

            energy = RNA.duplexfold(self.target, seq).energy
            fitnesses.append(energy / self.norm_value)

        return np.array(fitnesses)
    
    def get_training_data(
            self,
            n_train: int,
            p_mut: float,
            seed_idx: int = 3,
            noise_sd: float = 0.02
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
        noise_n = sc.stats.norm.rvs(loc=0, scale=noise_sd, size=n_train)
        ytrain_n = ytrain_n + noise_n
        return trainseqs_n, ytrain_n


def train_models(
        n_train: int,
        p_mutation: float = 0.1,
        noise_sd: float = 0.02,
        n_hidden: int = 10,
        n_epoch: int = 5,
        lr: float = 0.001,
        n_filters: int = 32,
        save_path: str = '/homefs/home/wongfanc/density-ratio-estimation/rna-models',
        save_fname_no_ftype: str = None,
    ):
    """
    Trains a ridge regression model, ensemble of CNNs, and ensemble of feedforward models
    given training data.
    """

    # generate training and test data
    landscape = RNABinding()
    trainseq_n, ytrain_n = landscape.get_training_data(
        n_train,
        p_mutation,
        noise_sd=noise_sd
    )
    testseq_n, ytest_n = landscape.get_training_data(
        n_train,
        p_mutation,
        noise_sd=noise_sd
    )

    # train models
    ridge = models.RidgeRegressor(seq_len=landscape.SEQ_LEN, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)
    print(f'CV-selected alpha for ridge: {ridge.model.alpha_}.')

    ff = models.FeedForward(landscape.SEQ_LEN, RNA_NUCLEOTIDES, n_hidden)
    _ = ff.fit(
        trainseq_n,
        ytrain_n,
        n_epoch=n_epoch,
        lr=lr,
    )

    cnn = models.CNN(landscape.SEQ_LEN, RNA_NUCLEOTIDES, n_filters, n_hidden)
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
    
    return ridge, ff, cnn, testseq_n, ytest_n


def write_sequences(seq_n, fname):
    with open(fname, 'w') as f:
        for seq in seq_n:
            f.write(f"{seq}\n")


def sample_design_sequences(
    n_design: int,
    adalead_thresholds,
    biswas_temperatures,
    model_and_data_fname_no_ftype: str,
    model_and_data_path: str = '/homefs/home/wongfanc/density-ratio-estimation/rna-models',
    design_pkl_fname: str = None,
    n_hidden: int = 100,
    n_filters: int = 32,
    n_recomb_partner: int = 1,
    recomb_rate: float = 0.2,
    max_mu: float = 2,
    n_trust_radius_mutations: int = 10,
    n_step: int = 10000,
    latent_dim: int = 10,
    n_vae_hidden: int = 20,
    quantile: float = 0.95,
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
    ridge = ridge = models.RidgeRegressor(seq_len=RNABinding.SEQ_LEN, alphabet=RNA_NUCLEOTIDES)
    ridge.fit(trainseq_n, ytrain_n)

    ff_fname = os.path.join(model_and_data_path, 'ff-' + model_and_data_fname_no_ftype + '.pt')
    ff = models.FeedForward(RNABinding.SEQ_LEN, RNA_NUCLEOTIDES, n_hidden)
    ff.load(ff_fname)

    cnn_fname = os.path.join(model_and_data_path, 'cnn-' + model_and_data_fname_no_ftype + '.pt')
    cnn = models.CNN(RNABinding.SEQ_LEN, RNA_NUCLEOTIDES, n_filters, n_hidden)
    cnn.load(cnn_fname)

    name2model = {
        'ridge': ridge,
        'ff': ff,
        'cnn': cnn
    }

    # design sequences
    name2designs = {}
    landscape = RNABinding()
    if design_pkl_fname is not None:
        print(f'Saving all results to {design_pkl_fname}.\n')
    for model_name, model in name2model.items():

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
                print_every=20000
            )
            print(f'  Done. ({int(time() - t0)} s)')
            
            # store
            predadalead_n = model.predict(adalead_n)
            yadalead_n = landscape.get_fitness(adalead_n)
            print('  Mean label, prediction: {:.3f}, {:.3f}'.format(
                np.mean(yadalead_n), np.mean(predadalead_n)
            ))
            name2designs[f'adalead{threshold}-{model_name}'] = (adalead_n, yadalead_n, predadalead_n)
            
            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} AdaLead threshold = {threshold} {model_name} sequences.')
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
                RNABinding.SEEDS[3],
                max_mu,
                temp,
                n_trust_radius_mutations,
                n_step,
                print_every=2000
            )
            print(f'  Done. ({int(time() - t0)} s)')

            # store
            predbiswas_n = model.predict(biswas_n)
            ybiswas_n = landscape.get_fitness(biswas_n)
            print('  Mean label, prediction: {:.3f}, {:.3f}'.format(
                np.mean(ybiswas_n), np.mean(predbiswas_n)
            ))
            name2designs[f'biswas{temp}-{model_name}'] = (biswas_n, ybiswas_n, predbiswas_n)

            # save
            if design_pkl_fname is not None:
                with open(design_pkl_fname, 'wb') as f:
                    pickle.dump(name2designs, f)
                print(f'  Saved {n_design} Biswas temperature = {temp} {model_name} sequences.')
            print()

        # ===== CbAS =====
        cbas = designers.CbAS(
            model,
            trainseq_n,
            latent_dim=latent_dim,
            n_hidden=n_vae_hidden,
            weight_type='cbas',
            device='cpu'
        )
        # design sequences
        print(f'Designing CbAS {model_name} sequences...')
        t0 = time()
        cbas_n = cbas.design_sequences(
            n_design,
            quantile=quantile
        )
        print(f'  Done. ({int(time() - t0)} s)')
        # store
        predcbas_n = model.predict(cbas_n)
        ycbas_n = landscape.get_fitness(cbas_n)
        print('  Mean label, prediction: {:.3f}, {:.3f}'.format(
            np.mean(ycbas_n), np.mean(predcbas_n)
        ))
        name2designs[f'cbas-{model_name}'] = (cbas_n, ycbas_n, predcbas_n)
        # save
        if design_pkl_fname is not None:
            with open(design_pkl_fname, 'wb') as f:
                pickle.dump(name2designs, f)
            print(f'  Saved {n_design} CbAS {model_name} sequences.')
        print()


        # ===== DbAS =====
        dbas = designers.CbAS(
            model,
            trainseq_n,
            latent_dim=latent_dim,
            n_hidden=n_vae_hidden,
            weight_type='dbas',
            device='cpu'
        )
        # design sequences
        print(f'Designing DbAS {model_name} sequences...')
        t0 = time()
        dbas_n = dbas.design_sequences(
            n_design,
            quantile=quantile
        )
        print(f'  Done. ({int(time() - t0)} s)')
        # store
        preddbas_n = model.predict(dbas_n)
        ydbas_n = landscape.get_fitness(dbas_n)
        print('  Mean label, prediction: {:.3f}, {:.3f}'.format(
            np.mean(ydbas_n), np.mean(preddbas_n)
        ))
        name2designs[f'dbas-{model_name}'] = (dbas_n, ydbas_n, preddbas_n)
        # save
        if design_pkl_fname is not None:
            with open(design_pkl_fname, 'wb') as f:
                pickle.dump(name2designs, f)
            print(f'  Saved {n_design} DbAS {model_name} sequences.')
        print()


        # ===== PEX =====
        pex = designers.PEX(
            model,
            trainseq_n,
            ytrain_n,
            RNABinding.SEEDS[3],
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
        ypex_n = landscape.get_fitness(pex_n)
        print('  Mean label, prediction: {:.3f}, {:.3f}'.format(
            np.mean(ypex_n), np.mean(predpex_n)
        ))
        name2designs[f'pex-{model_name}'] = (pex_n, ypex_n, predpex_n)
        # save
        if design_pkl_fname is not None:
            with open(design_pkl_fname, 'wb') as f:
                pickle.dump(name2designs, f)
            print(f'  Saved {n_design} PEX {model_name} sequences.')
    
    return name2designs


     

