import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import flexs.utils.sequence_utils as s_utils

import editdistance

def rmse(x, y, axis=None):
    return np.sqrt(np.mean(np.square(x - y), axis=axis))


def plot_xy(x, y, s: int = 50, linewidth: float = 2, alpha: float = 0.8):
    plt.scatter(x, y, s=s, alpha=alpha)
    minval = np.min([np.min(x), np.min(y)])
    maxval = np.max([np.max(x), np.max(y)])
    plt.plot([minval, maxval], [minval, maxval], '--', c='orange', linewidth=linewidth)

def process_flexs_outputs(df, alphabet, n_train: int):
    # TODO: split calibration data properly
    df1 = df[df['round'] == 1]
    df2 = df[df['round'] == 2]
    X0_nxd = np.array([s_utils.string_to_one_hot(seq, alphabet).flatten() for seq in df2.sequence[: n_train]])
    y0_n = df2.true_score.to_numpy()
    pred0_n = df2.model_score.to_numpy()
    Xm_nxd = np.array([s_utils.string_to_one_hot(seq, alphabet).flatten() for seq in df1.sequence[: n_train]])
    Xmcal_nxd = np.array([s_utils.string_to_one_hot(seq, alphabet).flatten() for seq in df1.sequence[n_train :]])
    ymcal_n = df1.true_score.to_numpy()[n_train :]
    predmcal_n = df1.model_score.to_numpy()[n_train:]
    return X0_nxd, y0_n, pred0_n, Xm_nxd, Xmcal_nxd, ymcal_n, predmcal_n

# ========== sequence distribution comparisons ==========

def pairwise_distances(seqs1, seqs2, normalize: bool = True, n_pairs: int = 1000):
    s1_n = np.random.choice(seqs1, size=n_pairs, replace=True)
    s12_n = np.random.choice(seqs1, size=n_pairs, replace=True)
    s13_n = np.random.choice(seqs1, size=n_pairs, replace=True)

    s2_n = np.random.choice(seqs2, size=n_pairs, replace=True)
    s22_n = np.random.choice(seqs2, size=n_pairs, replace=True)
    s23_n = np.random.choice(seqs2, size=n_pairs, replace=True)

    dists_between = np.array([editdistance.eval(s1, s2) for s1, s2 in zip(s1_n, s2_n)])
    dists_within1 = np.array([editdistance.eval(s12, s13) for s12, s13 in zip(s12_n, s13_n)])
    dists_within2 = np.array([editdistance.eval(s22, s23) for s22, s23 in zip(s22_n, s23_n)])
    
    if normalize:
        l = len(seqs1[0])
        print('Normalizing by sequence length {}'.format(l))
        dists_between = dists_between / l
        dists_within1 = dists_within1 / l
        dists_within2 = dists_within2 / l
    return dists_within1, dists_within2, dists_between
