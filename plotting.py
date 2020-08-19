import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_similarities(data, title=None, vmin=None, vmax=None):
    plt.figure()
    sns.set()
    sns.heatmap(data, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_cumulative_correlations(seq, keys):
    plt.figure()
    sns.set()
    plt.plot(seq)
    plt.plot(keys, seq[keys], 'x')
    plt.xlabel("frames")
    plt.ylabel("Cumulative correlation")
    plt.title("Fig. 8 Cumulutative correlation function for identifying peak expressions")
