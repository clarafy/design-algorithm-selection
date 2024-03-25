import random
from time import time
import abc
from copy import deepcopy

import torch

from pandas import DataFrame
import numpy as np
import editdistance

from vae import VAE
from utils import RNA_NUCLEOTIDES, get_mutant
from models import type_check_and_one_hot_encode_sequences


class Designer(abc.ABC):
    def __init__(self, model, trainseq_n):
        self.model = model
        self.trainseq_n = trainseq_n
    
    @abc.abstractmethod
    def design_sequences(self, n_design):
        pass


class CbAS(Designer):
    def __init__(
            self,
            model,
            trainseq_n,
            seq_len: int = 50,
            alphabet: str = RNA_NUCLEOTIDES,
            latent_dim: int = 10,
            n_hidden: int = 20,
            n_init_epoch: int = 5,
            weight_type: str = 'cbas',
            device = None,
        ):
        super().__init__(model, trainseq_n)

        # fit distribution of training sequences
        for seq in trainseq_n:
            assert(len(seq) == seq_len)

        # train distribution
        self.train_vae = VAE(
            seq_len=seq_len,
            alphabet=alphabet,
            latent_dim=latent_dim,
            n_hidden=n_hidden,
            device=device,
        )
        print('Fitting training distribution:')
        self.train_vae.fit(trainseq_n, n_epoch=n_init_epoch, verbose=True)

        # design distribution
        self.design_vae = VAE(
            seq_len=seq_len,
            alphabet=alphabet,
            latent_dim=latent_dim,
            n_hidden=n_hidden,
            device=device,
        )
        self.design_distribution_fitted = False
        self.alphabet = alphabet
        self.fitted_quantile = None
        self.weight_type = weight_type

    def fit_design_distribution(
            self,
            n_iter: int,
            n_sample: int = 1000,
            quantile: float = 0.9,
            tol: float = 1e-6,
            n_epoch: int = 5
        ):
        assert(quantile >= 0 and quantile <= 1)

        # initialize design distribution to training distribution
        self.design_vae.load_state_dict(self.train_vae.state_dict())

        threshold = -np.inf
        df_data = []
        t0 = time()

        for t in range(n_iter):
            # sample from current design distribution
            sample_n, pdesign_nxlxa, z_nxd = self.design_vae.generate(n_sample)
            ohe_nxlxa = type_check_and_one_hot_encode_sequences(sample_n, self.alphabet)

            # get predictions
            pred_n = self.model.predict(ohe_nxlxa, verbose=False)

            # get weights for refitting design distribution
            candidate_threshold = np.quantile(pred_n, quantile)
            if candidate_threshold > threshold:
                threshold = candidate_threshold
            # DbAS weights
            weight_n = self.model.predict_prob_exceedance(sample_n, threshold, verbose=False)

            if self.weight_type == 'cbas': 
                logpdesign_n = np.sum(np.log(pdesign_nxlxa) * ohe_nxlxa, axis=(1, 2))
                ptrain_nxlxa = self.train_vae.decode_probabilities(z_nxd)
                if t == 0:
                    assert(np.linalg.norm(pdesign_nxlxa - ptrain_nxlxa) < 1e-6)
                logptrain_n = np.sum(np.log(ptrain_nxlxa) * ohe_nxlxa, axis=(1, 2))
                dr_n = np.exp(logptrain_n - logpdesign_n)
                weight_n = weight_n * dr_n

            # refit candidate distribution
            weight_n = n_sample * weight_n / np.sum(weight_n)
            keep_idx = np.where(weight_n > tol)[0]  # for numerical stability
            sample_n = [sample_n[i] for i in keep_idx]
            weight_n = weight_n[keep_idx]
            self.design_vae.fit(sample_n, weight_n, n_epoch=n_epoch, verbose=False)

            # record and print progress
            df_data.append([np.mean(pred_n), np.std(pred_n), np.max(pred_n), keep_idx.size, threshold])
            print('Iter {}. Mean, SD, max prediction: {:.3f}, {:.3f}, {:.3f}. {} valid samples for threshold {:.3f}. ({} s)'.format(
                t, np.mean(pred_n), np.std(pred_n), np.max(pred_n), keep_idx.size, threshold, int(time() - t0)
            ))

        df = DataFrame(df_data, columns=['mean_pred', 'sd_pred', 'max_pred', 'n_valid_samples', 'threshold'])
        self.design_distribution_fitted = True
        self.fitted_quantile = quantile
        return df
    
    def design_sequences(
            self,
            n_design,
            n_iter: int = 20,
            n_sample: int = 1000,
            quantile: float = 0.95,
            n_epoch: int = 5
        ):
        if not self.design_distribution_fitted or self.fitted_quantile != quantile:
            print('Fitting design distribution with quantile hyperparameter = {}:'.format(quantile))
            _ = self.fit_design_distribution(
                n_iter=n_iter,
                n_sample=n_sample,
                quantile=quantile,
                n_epoch=n_epoch
            )
            self.fitted_quantile = quantile
        designseq_n, _, _ = self.design_vae.generate(n_design)
        return designseq_n


class PEX(Designer):
    """
    Proximal Exploration, adapted from:
    https://github.com/HeliXonProtein/proximal-exploration/blob/main/algorithm/pex.py
    """
    def __init__(
            self,
            model,
            trainseq_n,
            ytrain_n,
            seedseq,
            frontier_neighborhood_size: int = 5,
            alphabet: str = RNA_NUCLEOTIDES,
        ):
        super().__init__(model, trainseq_n)
        self.ytrain_n = ytrain_n
        self.alphabet = alphabet
        self.seedseq = seedseq
        self.frontier_neighborhood_size = frontier_neighborhood_size 

    def design_sequences(
            self,
            n_design: int,
            p_mutation: float,
            n_candidates = None,
        ):  
    
        if n_candidates is None:
            n_candidates = 2 * n_design

        # order training sequences by distance to the seed
        dist2seqy = {}
        for seq, y in zip(self.trainseq_n, self.ytrain_n):
            dist = editdistance.eval(seq, self.seedseq)

            if dist not in dist2seqy.keys():
                dist2seqy[dist] = []

            dist2seqy[dist].append([seq, y])

        # highlight sequences near the proximal frontier
        frontier_neighbors, frontier_height = [], -np.inf
        for dist in sorted(dist2seqy.keys()):
            seqy_list = dist2seqy[dist]
            seqy_list.sort(reverse=True, key=lambda seqy:seqy[1])

            for seqy in seqy_list[: self.frontier_neighborhood_size]:
                if seqy[1] > frontier_height:
                    frontier_neighbors.append(seqy)

            frontier_height = max(frontier_height, seqy_list[0][1])

        # construct candidates by randomly mutating sequences (line 2 of Algorithm 2 in PEX paper)
        # heuristic: only mutating sequences near the proximal frontier
        candidates = []
        while len(candidates) < n_candidates:
            candidate_seq = get_mutant(random.choice(frontier_neighbors)[0], p_mutation, self.alphabet)

            if candidate_seq not in self.trainseq_n and candidate_seq not in candidates:
                candidates.append(candidate_seq)

        # sort the candidates by distance to the seed
        candidate_dist2seqpred = {}
        pred_n = self.model.predict(candidates)
        for candidate_seq, pred in zip(candidates, pred_n):
            dist = editdistance.eval(candidate_seq, self.seedseq)

            if dist not in candidate_dist2seqpred.keys():
                candidate_dist2seqpred[dist] = []

            candidate_dist2seqpred[dist].append([candidate_seq, pred])

        for dist in sorted(candidate_dist2seqpred.keys()):
            candidate_dist2seqpred[dist].sort(reverse=True, key=lambda seqy:seqy[1])
        
        # iteratively extract the proximal frontier. 
        designs = []
        while len(designs) < n_design:
            # compute the proximal frontier by Andrew's monotone chain convex hull algorithm
            # (line 5 of Algorithm 2 in PEX paper)
            # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
            stack = []
            for dist in sorted(candidate_dist2seqpred.keys()):
                if len(candidate_dist2seqpred[dist]) > 0:

                    seqpred = candidate_dist2seqpred[dist][0]
                    new_distpred = np.array([dist, seqpred[1]])
                    
                    while len(stack) > 1 and not self.check_convex_hull(stack[-2], stack[-1], new_distpred):
                        stack.pop(-1)

                    stack.append(new_distpred)

            while len(stack) >= 2 and stack[-1][1] < stack[-2][1]:
                stack.pop(-1)
            
            # update design and candidate pools
            # (line 6 of Algorithm 2 in PEX paper)
            for dist, pred in stack:
                if len(designs) < n_design:
                    designs.append(candidate_dist2seqpred[dist][0][0])
                    candidate_dist2seqpred[dist].pop(0)
            
        return designs
    
    def check_convex_hull(self, point_1, point_2, point_3):
        return np.cross(point_2 - point_1, point_3 - point_1) <= 0
    

def recombine_population(seq_n, recomb_rate):
    if len(seq_n) == 1:
        return seq_n

    random.shuffle(seq_n)
    ret = []
    for i in range(0, len(seq_n) - 1, 2):
        strA = []
        strB = []
        switch = False
        for ind in range(len(seq_n[i])):
            if random.random() < recomb_rate:
                switch = not switch

            # putting together recombinants
            if switch:
                strA.append(seq_n[i][ind])
                strB.append(seq_n[i + 1][ind])
            else:
                strB.append(seq_n[i][ind])
                strA.append(seq_n[i + 1][ind])

        ret.append(''.join(strA))
        ret.append(''.join(strB))
    return ret


class AdaLead(Designer):
    def __init__(
            self,
            model,
            trainseq_n,
            ytrain_n,
            alphabet: str = RNA_NUCLEOTIDES
        ):
        super().__init__(model, trainseq_n)
        self.ytrain_n = ytrain_n
        self.alphabet = alphabet

    def design_sequences(
            self,
            n_design: int,
            n_candidates: int = None,
            threshold: float = 0.05,
            n_recomb_partner: int = 0,
            recomb_rate: float = 0,
            mutation_rate: float = 1,
            print_every: int = None
        ):

        if n_candidates is None:
            n_candidates = 2 * n_design
        assert(n_candidates >= n_design)

        # extract all training sequences within percentile of the maximum fitness
        y_max = np.max(self.ytrain_n)
        top_idx = np.where(self.ytrain_n >= y_max * (1 - np.sign(y_max) * threshold))[0]

        parentseq_n = [self.trainseq_n[i] for i in top_idx]

        design_n = []
        t0 = time()
        while len(design_n) < n_candidates:
            
            # generate recombinant mutants as parents
            if n_recomb_partner:
                for _ in range(n_recomb_partner):
                    nodeseq_n = recombine_population(parentseq_n, recomb_rate)
            else:
                nodeseq_n = parentseq_n.copy()
            
            # get predictions for parents
            predparents_n = self.model.predict(nodeseq_n)

            # pair each node sequence with the index of its parent,
            # to be able to access parent predictions
            nodes_n = list(enumerate(nodeseq_n))

            while len(nodes_n) and len(design_n) < n_candidates:

                # generate random mutant child for each node,
                # keeping track of the index of the upstream parent
                child_n = [
                    (parent_idx, get_mutant(seq, mutation_rate / len(seq), self.alphabet))
                    for parent_idx, seq in nodes_n
                ]

                # get predictions for children
                predchild_n = self.model.predict([seq for _, seq in child_n])

                # select some children as designs and new nodes
                nodes_n = []
                for (parent_idx, child_seq), predchild in zip(child_n, predchild_n):

                    # select child sequences that have not been observed before as designs 
                    if child_seq not in self.trainseq_n and child_seq not in design_n:
                        design_n.append(child_seq)

                        if print_every is not None and len(design_n) % print_every == 0:
                            print('Designed {} / {} candidates ({} s).'.format(
                                len(design_n), n_candidates, int(time() - t0)
                            ))

                        # if child also has better prediction than upstream parent,
                        # add as new node
                        if predchild >= predparents_n[parent_idx]:
                            nodes_n.append([parent_idx, child_seq])
        
        preddesign_n = self.model.predict(design_n)
        idx = np.argsort(preddesign_n)[::-1][: n_design]
        design_n = [design_n[i] for i in idx]

        return design_n


def add_mutations(seq, n_mutation: int, alphabet: str = RNA_NUCLEOTIDES):

    n_mutation = min(n_mutation, len(seq))
    sites_to_mutate = np.random.choice(len(seq), n_mutation, replace=False)

    seq_as_list = list(seq)
    for site in sites_to_mutate:
        token = np.random.choice(list(set(alphabet) - set([seq[site]])))
        seq_as_list[site] = token
        
    return ''.join(seq_as_list)


def get_acceptance_probabilities(predproposal_n, predcurrent_n, temperature):
    acceptprob_n = np.exp((predproposal_n - predcurrent_n) / temperature)
    acceptprob_n[acceptprob_n > 1] = 1
    return acceptprob_n
    

class Biswas(Designer):
    def __init__(self, model, trainseq_n):
        super().__init__(model, trainseq_n)

    def design_sequences(
            self,
            n_design: int,
            seed_seq: str,
            max_mu: float,
            temperature: float,
            n_trust_radius_mutations: int,
            n_step: int,
            print_every: int
        ):

        # initialize sampling chains with random mutants of seed,d
        # following same initialization procedure as Biswas et al. (2021)
        n_mutation_n = np.random.poisson(2, size=n_design) + 1
        seq_n = [add_mutations(seed_seq, n_mut) for n_mut in n_mutation_n]
        # initializing with training sequences yields very similar results
        # init_idx = np.random.choice(len(self.trainseq_n), n_design, replace=True)
        # seq_n = [self.trainseq_n[i] for i in init_idx]
        predcurrent_n = self.model.predict(seq_n)

        # set mutation rates for each chain
        mu_n = np.random.uniform(low=1., high=max_mu, size=n_design)

        # record acceptance probabilities and predictions of the chains
        acceptprob_nxt = np.nan * np.ones([n_design, n_step])
        pred_nxt1 = np.nan * np.ones([n_design, n_step + 1])
        pred_nxt1[:, 0] = predcurrent_n
        if print_every is not None:
            print('Initialization. Mean, SD prediction: {:.3f}, {:.3f}.'.format(
                np.mean(predcurrent_n), np.std(predcurrent_n)
            ))

        t0 = time()
        for t in range(n_step):

            # generate proposal sequence per chain
            n_mutation_n = np.random.poisson(mu_n - 1) + 1
            proposalseq_n = [add_mutations(seq, n_mut) for seq, n_mut in zip(seq_n, n_mutation_n)]
            
            # compute acceptance probability
            predproposal_n = self.model.predict(proposalseq_n)
            acceptprob_n = get_acceptance_probabilities(predproposal_n, predcurrent_n, temperature)
            # enforce trust region
            n_mutation_from_seed_n = np.array([editdistance.eval(seed_seq, seq) for seq in proposalseq_n])
            acceptprob_n[n_mutation_from_seed_n > n_trust_radius_mutations] = 0
            acceptprob_nxt[:, t] = acceptprob_n

            # accept or reject proposal sequences 
            accept_n = np.random.rand(n_design) < acceptprob_n
            for i, accept in enumerate(accept_n):
                if accept:
                    # update current state and prediction of current state
                    seq_n[i] = deepcopy(proposalseq_n[i])
                    predcurrent_n[i] = predproposal_n[i] 
                # else do nothing (reject)
            
            # record predictions of current states
            pred_nxt1[:, t + 1] = predcurrent_n

            if print_every is not None and (t + 1) % print_every == 0:
                print('Step {}. Mean, SD acceptance probability: {:.3f}, {:.3f}. Mean, SD prediction: {:.3f}, {:.3f}. ({} s)'.format(
                    t + 1,
                    np.mean(acceptprob_n), np.std(acceptprob_n),
                    np.mean(pred_nxt1[:, t + 1]), np.std(pred_nxt1[:, t + 1]),
                    int(time() - t0)
                ))
        
        return seq_n, pred_nxt1, acceptprob_nxt
