import numpy as np

from load_data import load_training_frames
from compute_delta import compute_delta
from re_order_delta import re_order_delta
from compute_corr_coef import compute_corr_coef
from compute_corr_coef import compute_tilda_corr_coef
from compute_trust_values import compute_trust_values
from plotting import plot_similarities

"""
run: python -m blendshape_transfer
"""

# define parameters
max_num_seq = 3  # set to None if we want to use all the sequences
do_plot = True

# load data
sk = np.load('data/louise_bs_vrts_pos.npy')  # sparse representation of the blend shapes (vk)
ref_sk = sk[-1]  # neutral pose is the last one
delta_sk = compute_delta(sk[:-1], ref_sk)
delta_sk = delta_sk[:25]  # todo remove here to train on all bs!
af, delta_af = load_training_frames('D:/MoCap_Data/David/NewSession_labeled/', num_markers=45, max_num_seq=max_num_seq)  # actor animation  # todo downsamples freq?
delta_af = np.delete(delta_af, (38, 39, 40, 44), 1)  # remove HEAD markers

print("Finished loading data")
print("shape af, sk:", np.shape(af), np.shape(sk))
print("shape delta_sk", np.shape(delta_sk))
print("shape delta af:", np.shape(delta_af))


# 1) Facial Motion Similarity
# reorder delta blendshapes
sorted_delta_sk = re_order_delta(np.reshape(delta_sk, (np.shape(delta_sk)[0], -1)))

# measure similarity between character blendshapes and actor's capture performance
ckf = compute_corr_coef(np.reshape(delta_af, (np.shape(delta_af)[0], -1)),
                        np.reshape(sorted_delta_sk, (np.shape(delta_sk)[0], -1)))
print("shape ckf", np.shape(ckf))
if do_plot:
    plot_similarities(ckf, "Fig. 7: Motion space similarity")

# contrast enhancement
tk = compute_trust_values(sorted_delta_sk, do_plot=do_plot)
tilda_ckf = compute_tilda_corr_coef(ckf, tk)

# 2) extract most important frames

# 3) build initial guess blend shape RBF wrap