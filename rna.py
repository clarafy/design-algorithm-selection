from time import time
import numpy as np
import scipy as sc

try:
    import RNA
except ImportError:
    pass

RNA_NUCLEOTIDES = 'UGCA'

# ===== ViennaRNA binding landscape =====

def get_mutant(seq, p_mut, alphabet):
    mutprobs_axa = (p_mut / (len(alphabet) - 1)) * np.ones((len(alphabet), len(alphabet)))
    mutprobs_axa[np.diag_indices(len(alphabet), ndim=2)] = 1 - p_mut
    token2mutprobs = {token: mutprobs_axa[i] for i, token in enumerate(alphabet)}
    alphabet_list = list(alphabet)
    return ''.join([np.random.choice(alphabet_list, p=token2mutprobs[token]) for token in seq])

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
        self.seq_len = 50
        self.norm_value = self.compute_min_binding_energy()

    def compute_min_binding_energy(self):
        """Compute the lowest possible binding energy for the target."""
        complements = {"A": "U", "C": "G", "G": "C", "U": "A"}

        complement = "".join(complements[x] for x in self.target)[::-1]
        energy = RNA.duplexfold(complement, self.target).energy
        return energy * self.seq_len / len(self.target)

    def get_fitness(self, sequences):
        fitnesses = []

        for seq in sequences:

            if len(seq) != self.seq_len:
                raise ValueError('All sequences in `sequences` must be of length {self.seq_len}.')

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


def train_models():
    """
    Trains a ridge regression model, ensemble of CNNs, and ensemble of feedforward models
    given training data.
    """

def sample_design_sequences():
    """
    Samples design sequences using Adalead, CbAS, DbAS, PEX, Biswas, and CMA-ES for many trials.
    """
    # HERE: need to save training sequences in order to use CbAS/DbAS/Adalead.
    # think through organization of Designer superclass.