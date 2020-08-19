import numpy as np

from load_data import load_training_frames
from compute_delta import compute_delta
from re_order_delta import re_order_delta
from compute_corr_coef import compute_corr_coef
from compute_corr_coef import compute_tilda_corr_coef
from compute_trust_values import compute_trust_values
from get_key_expressions import get_key_expressions
from get_soft_mask import get_soft_mask
from EMatch import EMatch
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
print()


# 1) Facial Motion Similarity
# reorder delta blendshapes
sorted_delta_sk = re_order_delta(np.reshape(delta_sk, (np.shape(delta_sk)[0], -1)))

# measure similarity between character blendshapes and actor's capture performance
ckf = compute_corr_coef(np.reshape(delta_af, (np.shape(delta_af)[0], -1)),
                        np.reshape(sorted_delta_sk, (np.shape(delta_sk)[0], -1)))
if do_plot:
    plot_similarities(ckf, "Fig. 7: Motion space similarity")

# contrast enhancement
tk = compute_trust_values(sorted_delta_sk, do_plot=do_plot)
tilda_ckf = compute_tilda_corr_coef(ckf, tk)
print("shape ckf", np.shape(ckf))
print("shape tk", np.shape(tk))
print("shape tilda_ckf", np.shape(tilda_ckf))
print()

# 2) Key Expression Extraction
key_expressions = get_key_expressions(tilda_ckf, ksize=3, theta=2, do_plot=do_plot)
print("shape key_expressions", np.shape(key_expressions))
print()

# 3) Manifold Alignment
# built uk
uk = get_soft_mask(delta_sk)
# get E_match function
e_match_fn = EMatch(tilda_ckf, uk, delta_af).get_ematch()
e_match_fn("coucou")

# 4) build initial guess blend shape RBF wrap