import numpy as np
import time as time

from utils.load_data import load_training_frames
from utils.compute_delta import compute_delta
from utils.re_order_delta import re_order_delta
from utils.get_key_expressions import get_key_expressions
from src.compute_corr_coef import compute_corr_coef
from src.compute_corr_coef import compute_tilda_corr_coef
from src.compute_trust_values import compute_trust_values
from src.get_soft_mask import get_soft_mask
from src.EAlign import EAlign
from src.RBF_warp import get_initial_actor_blendshapes
from plotting import plot_similarities


"""
run: python -m blendshape_transfer
"""

# define parameters
save_folder = 'data/'
file_name = "David_based_Louise_personalized_blendshapes.npy"
max_num_seq = 3  # set to None if we want to use all the sequences
do_plot = True

# load data
sk = np.load('data/louise_bs_vrts_pos.npy')  # sparse representation of the blend shapes (vk)
ref_sk = sk[-1]  # neutral pose is the last one
delta_sk = compute_delta(sk[:-1], ref_sk)
# get actor animation  # todo downsamples freq?
af, delta_af = load_training_frames('D:/MoCap_Data/David/NewSession_labeled/', num_markers=45, max_num_seq=max_num_seq)
af = np.delete(af, (38, 39, 40, 44), 1)  # remove HEAD markers
delta_af = np.delete(delta_af, (38, 39, 40, 44), 1)  # remove HEAD markers
print("[data] Finished loading data")
print("[data] shape af:", np.shape(af))
print("[data] shape delta_sk", np.shape(delta_sk))
print("[data] shape delta af:", np.shape(delta_af))

# get dimensions
K, M, n_dim = np.shape(delta_sk)
F = np.shape(delta_af)[0]
print("[data] num_blendshapes:", K)
print("[data] num_markers:", M)
print("[data] num_features (M*3):", M*n_dim)
print("[data] num_frames", F)
print()

# 1) Facial Motion Similarity
# reorder delta blendshapes
sorted_delta_sk = re_order_delta(np.reshape(delta_sk, (np.shape(delta_sk)[0], -1)))
print("[Pre-processing] shape sorted_delta_sk", np.shape(sorted_delta_sk))

# measure similarity between character blendshapes and actor's capture performance
ckf = compute_corr_coef(np.reshape(delta_af, (np.shape(delta_af)[0], -1)),
                        np.reshape(sorted_delta_sk, (np.shape(delta_sk)[0], -1)))
if do_plot:
    plot_similarities(ckf, "Fig. 7: Motion space similarity")

# contrast enhancement
tk = compute_trust_values(sorted_delta_sk, do_plot=do_plot)
tilda_ckf = compute_tilda_corr_coef(ckf, tk)
print("[Pre-processing] shape ckf", np.shape(ckf))
print("[Pre-processing] shape tk", np.shape(tk))
print("[Pre-processing] shape tilda_ckf", np.shape(tilda_ckf))
print()

# 2) Key Expression Extraction
key_expressions_idx = get_key_expressions(tilda_ckf, ksize=3, theta=2, do_plot=do_plot)
print("[Key Expr. Extract.] shape key_expressions", np.shape(key_expressions_idx))
F = len(key_expressions_idx)
print("[Key Expr. Extract.] Keep", F, " frames")
delta_af = delta_af[key_expressions_idx, :, :]
tilda_ckf = tilda_ckf[:, key_expressions_idx]
print("[Key Expr. Extract.] shape delta_af", np.shape(delta_af))
print("[Key Expr. Extract.] shape tilda_ckf", np.shape(tilda_ckf))
print()

# 3) Manifold Alignment
# built soft max vector
uk = get_soft_mask(delta_sk)
print("[SoftMax] shape uk", np.shape(uk))
print()

# 4) Geometric Constraint
# build initial guess blendshape using RBF wrap (in delta space)
delta_gk = get_initial_actor_blendshapes(ref_sk, af[0], delta_sk)
print("[RBF Wrap] shape delta_gk", np.shape(delta_gk))
print()

# 5) build personalized actor-specific blendshapes (delta_p)
# rehsape to match required dimensions
delta_af = np.reshape(delta_af, (F, M*n_dim))
delta_sk = np.reshape(delta_sk, (K, M*n_dim))
# print control of all shapes
print("[dp] shape tilda_ckf:", np.shape(tilda_ckf))
print("[dp] shape uk:", np.shape(uk))
print("[dp] shape delta_af:", np.shape(delta_af))
print("[dp] shape delta_gk:", np.shape(delta_gk))
print("[dp] shape delta_sk", np.shape(delta_sk))
# declare E_Align
e_align = EAlign(tilda_ckf, uk, delta_af, delta_gk, delta_sk)
# compute personalized actor-specific blendshapes
start = time.time()
delta_p = e_align.compute_actor_specific_blendshapes(vectorized=False)
print("[dp] Solved in:", time.time() - start)
print("[dp] shape delta_p", np.shape(delta_p))
# save
np.save(save_folder + file_name, delta_p)
