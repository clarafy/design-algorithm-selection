import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_xy(x, y, s: int = 50, linewidth: float = 2, alpha: float = 0.8):
    plt.scatter(x, y, s=s, alpha=alpha)
    minval = np.min([np.min(x), np.min(y)])
    maxval = np.max([np.max(x), np.max(y)])
    plt.plot([minval, maxval], [minval, maxval], '--', c='orange', linewidth=linewidth)
