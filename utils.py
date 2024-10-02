from time import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import flexs.utils.sequence_utils as s_utils

import editdistance

RNA_NUCLEOTIDES = 'UGCA'
RNANUC2COMPLEMENT = {"A": "U", "C": "G", "G": "C", "U": "A"}

# ==========

def get_mutant(seq, p_mut, alphabet):
    mutant = []
    for s in seq:
        if np.random.rand() < p_mut:
            mutant.append(np.random.choice(list(alphabet)))
        else:
            mutant.append(s)
    return "".join(mutant)

def str2onehot(sequence: str, alphabet: str) -> np.ndarray:
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out

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


# ========== plotting ==========

def process_gb1_selection_experiments(
    df,
    target_values,
    temperatures,
    temp2mean,
    n_trial: int,
    imp_or_pp: str = 'imp',
    alpha: float = 0.1
):

    n_temp = temperatures.shape[0]
    alpha_bonferroni = alpha / n_temp
    print('Processing {} results with {} temperatures in [{:.3f}, {:.3f}], {} target values in [{:.2f}, {:.2f}], {} trials, and alpha = {:.1f}'.format(
        imp_or_pp, n_temp, np.min(temperatures), np.max(temperatures),
        target_values.shape[0], np.min(target_values), np.max(target_values),
        n_trial, alpha
    ))

    worst_v = []  # worst (i.e. lowest) mean label achieved by any selected temperature
    err_v = []    # error rate
    disc_v = []   # discovery rate
    val2temprange = {}  # map from target value to range of selected temperatures

    t0 = time()
    for v, val in enumerate(target_values):
        val = round(val, 4)
            
        worst_t = []  # worst (i.e. lowest) mean label for each trial
        temprange_t = []  # range of selected temperatures for each trial
        
        for i in range(n_trial):
            selected = [temp for temp in temperatures if df.loc[val]['tr{}_{}_pval_temp{:.4f}'.format(i, imp_or_pp, temp)] < alpha_bonferroni]
            achieved = [temp2mean[round(t, 4)] for t in selected]

            if len(selected):
                worst_t.append(np.min(achieved))
                temprange_t.append([np.min(selected), np.max(selected)])
            # if no discovery/selection, no worst achieved value
            else:
                temprange_t.append([])
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        disc_v.append(len(worst_t) / n_trial)
        val2temprange[val] = temprange_t

    print('Done processing ({} s)'.format(int(time() - t0)))
    return worst_v, err_v, disc_v, val2temprange
    

def process_rna_selection_experiments(
    df,
    target_values,
    design_names,
    name2truemeans,
    n_trial: int,
    imp_or_pp: str,
    alpha: float = 0.1
):
    n_config = len(design_names)
    alpha_bonferroni = alpha / n_config
    assert(set(design_names) == set(name2truemeans.keys()))
    print('Processing {} results with the following {} configurations, {} target values in [{:.2f}, {:.2f}], {} trials, and alpha = {:.1f}:'.format(
        imp_or_pp, n_config, target_values.shape[0], np.min(target_values), np.max(target_values), n_trial, alpha
    ))
    for name in design_names:
        print(f'  {name}')

    format_tokens = '{:.4f} '
    format_str = '  {}: ' + ''.join(n_trial * [format_tokens])
    report_high_var = False
    for name, truemean_t in name2truemeans.items():
        vmin, vmax = np.min(truemean_t), np.max(truemean_t)
        if (vmax - vmin) / vmax > 0.05 * vmax:
            if not report_high_var:
                report_high_var = True
                print('High-ish variance estimates of true mean labels for the following. Using the average:')
            print(format_str.format(name, *truemean_t))

    worst_v = []  # worst (i.e. lowest) mean label achieved by any selected configuration
    err_v = []    # error rate
    disc_v = []   # discovery rate
    val2configs = {}  # map from target value to lists selected configurations (per trial)

    t0 = time()
    for v, val in enumerate(target_values):
        val = round(val, 4)
            
        worst_t = []    # worst (i.e. lowest) mean label for each trial
        configs_t = []  # selected configurations for each trial
        
        for i in range(n_trial):
            selected = [name for name in design_names if df.loc[val]['tr{}_{}_pval_{}'.format(i, imp_or_pp, name)] < alpha_bonferroni]
            achieved = [np.mean(name2truemeans[name]) for name in selected]

            if len(selected):
                worst_t.append(np.min(achieved))
                configs_t.append(selected)
            # if no discovery/selection, no worst achieved value
            else:
                configs_t.append([])
                
        worst_v.append(worst_t)
        err_v.append(np.sum(np.array(worst_t) < val) / n_trial) 
        disc_v.append(len(worst_t) / n_trial)
        val2configs[val] = configs_t
        
        if (v + 1) % 20 == 0:
            print('Done processing {} / {} ({} s)'.format(v + 1, target_values.size, int(time() - t0)))

    print('Done processing ({} s)'.format(int(time() - t0)))
    return worst_v, err_v, disc_v, val2configs
