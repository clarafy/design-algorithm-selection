from abc import ABC, abstractmethod

import numpy as np
import scipy as sc
import pandas as pd

import flexs
import flexs.utils.sequence_utils as sutils

from shifts import DistributionShift, get_mutant

class FLEXSShift(object):
    def __init__(self, landscape_name):
        if 'RNA' in landscape_name:
            self.problem = flexs.landscapes.rna.registry()[landscape_name]
            self.landscape = flexs.landscapes.RNABinding(**self.problem['params'])
            self.alphabet = sutils.RNAA
        else:
            raise ValueError('Unrecognized landscape_name: {}'.format(landscape_name))
        # super().__init__(self.landscape.seq_length * len(self.alphabet))


    def get_data(self, n: int, model_class, explorer_class, explorer_kwargs, seed_idx: int = 1, avg_n_mut: int = 2):
        model = model_class(alphabet=self.alphabet)
        seed = self.problem['starts'][seed_idx]
        self.explorer = explorer_class(
            model,
            **explorer_kwargs,
            starting_sequence=seed,
            rounds=2,
            sequences_batch_size=n,
            model_queries_per_batch=3 * n,
            alphabet=self.alphabet
        )

        # generate random mutants around WT
        p_mut = avg_n_mut / self.landscape.seq_length
        trainseqs_n = [get_mutant(seed, p_mut, self.alphabet) for _ in range(n)]
        ytrain_n = self.landscape.get_fitness(trainseqs_n)
        train_data = pd.DataFrame(
            {
                "sequence": trainseqs_n,
                "model_score": np.nan,
                "true_score": ytrain_n,
                "round": 0,
                "model_cost": self.explorer.model.cost,
                "measurement_cost": 1,
            }
        )

        # train predictive model on data
        self.explorer.model.train(trainseqs_n, ytrain_n)

        # run design algorithm
        testseqs_m, predtest_m = self.explorer.propose_sequences(train_data)
        ytest_m = self.landscape.get_fitness(testseqs_m)

        return trainseqs_n, ytrain_n, testseqs_m, ytest_m, predtest_m


    def get_log_dr(self, X_nxd: np.array):
        pass