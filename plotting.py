import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_similarities(data, title=None, vmin=None, vmax=None):
    sns.set()
    sns.heatmap(data, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.show()