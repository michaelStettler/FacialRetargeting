import numpy as np
from compute_corr_coef import compute_corr_coef
from plotting import plot_similarities


def compute_trust_values(dsk, do_plot=False):
    """
    Compute trust values following formula 6

    k:= number of blendshapes

    :param dsk: delta_sk vector (k, num_features)
    :param do_plot: decide if we want to plot the between-correlation matrix
    :return: trust values vector (k)
    """
    # compute between-blendshape correlation
    ckl = compute_corr_coef(dsk, dsk)
    ckl = np.maximum(ckl, np.zeros(np.shape(ckl)))
    if do_plot:
        plot_similarities(ckl, "Between blendshapes correlation", vmin=0, vmax=1)

    # compute lower triangle
    num_k = np.shape(ckl)[0]
    low_trig = np.zeros(num_k)
    for k in range(num_k):
        val = 0
        for l in range(k):
            val += ckl[k, l]
        low_trig[k] = val
    max_low_trig = np.max(low_trig)

    # compute trust values  (formula 6)
    tk = np.zeros(num_k)
    for k in range(len(tk)):
        tk[k] = 1 - low_trig[k]/max_low_trig

    return tk


if __name__ == '__main__':
    """
    test compute_trust_values function
    
    run: python -m compute_trust_values
    """
    np.random.seed(0)
    from re_order_delta import re_order_delta

    # test compute trust values
    sk = np.random.rand(6, 3)  # (k, num_features)
    sorted_sk = re_order_delta(sk)

    tk = compute_trust_values(sorted_sk, do_plot=False)

    print("tk")
    print(tk)
