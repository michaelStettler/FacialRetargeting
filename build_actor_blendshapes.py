import numpy as np
import time as time
import os
import matplotlib.pyplot as plt

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
from utils.plotting import plot_similarities


"""
run: python -m blendshape_transfer
"""

# define parameters
actor_recording_data_folder = 'D:/MoCap_Data/David/NewSession_labeled/'
blendshape_mesh_list_name = "D:/Maya projects/DigitalLuise/scripts/mesh_name_list.npy"
load_folder = 'data/'
sparse_blendhsape_vertices_pos_name = "louise_to_David_markers_blendshape_vertices_pos.npy"
save_folder = 'data/'
save_file_name = "David_based_Louise_personalized_blendshapes.npy"
neutral_pose_name = 'Louise_Neutral'
max_num_seq = 3  # set to None if we want to use all the sequences
do_plot = False

# load data
mesh_list = np.load(blendshape_mesh_list_name).astype(str)
sk = np.load(os.path.join(load_folder, sparse_blendhsape_vertices_pos_name))  # sparse representation of the blendshapes (vk)
# get Neutral blendhsape pose
ref_index = None
bs_index = []
for m, mesh in enumerate(mesh_list):
    if mesh == neutral_pose_name:
        ref_index = m
    else:
        bs_index.append(m)
if ref_index is None:
    raise ValueError("No Neutral pose find!")
print("[data] Reference index:", ref_index)
ref_sk = sk[ref_index]
delta_sk = compute_delta(sk[bs_index, :, :], ref_sk)
# get actor animation  # todo downsamples freq?
af, delta_af = load_training_frames(actor_recording_data_folder, num_markers=45, max_num_seq=max_num_seq)
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
sorted_delta_sk, sorted_index = re_order_delta(delta_sk)
sorted_mesh_list = mesh_list[sorted_index]
print("[Pre-processing] shape sorted_delta_sk", np.shape(sorted_delta_sk))

# measure similarity between character blendshapes and actor's capture performance
ckf = compute_corr_coef(np.reshape(delta_af, (np.shape(delta_af)[0], -1)),
                        np.reshape(sorted_delta_sk, (np.shape(sorted_delta_sk)[0], -1)))
if do_plot:
    plot_similarities(ckf, "Fig. 7: Motion space similarity")

# contrast enhancement
tk = compute_trust_values(np.reshape(sorted_delta_sk, (np.shape(sorted_delta_sk)[0], -1)), do_plot=do_plot)
tilda_ckf = compute_tilda_corr_coef(ckf, tk)
print("[Pre-processing] shape ckf", np.shape(ckf))
print("[Pre-processing] shape tk", np.shape(tk))
print("[Pre-processing] shape tilda_ckf", np.shape(tilda_ckf))
print()

# 2) Key Expression Extraction
key_expressions_idx = get_key_expressions(tilda_ckf, ksize=3, theta=2, do_plot=do_plot)
print("[Key Expr. Extract.] shape key_expressions", np.shape(key_expressions_idx))
F = len(key_expressions_idx)
print("[Key Expr. Extract.] Keep", F, "frames")
delta_af = delta_af[key_expressions_idx, :, :]
tilda_ckf = tilda_ckf[:, key_expressions_idx]
print("[Key Expr. Extract.] shape delta_af", np.shape(delta_af))
print("[Key Expr. Extract.] shape tilda_ckf", np.shape(tilda_ckf))
print()

# 3) Manifold Alignment
# built soft max vector
uk = get_soft_mask(sorted_delta_sk)
print("[SoftMax] shape uk", np.shape(uk))
print()

# 4) Geometric Constraint
# build initial guess blendshape using RBF wrap (in delta space)
delta_gk = get_initial_actor_blendshapes(ref_sk, af[0], sorted_delta_sk)
print("[RBF Wrap] shape delta_gk", np.shape(delta_gk))
print()

# 5) build personalized actor-specific blendshapes (delta_p)
# rehsape to match required dimensions
delta_af = np.reshape(delta_af, (F, M*n_dim))
sorted_delta_sk = np.reshape(sorted_delta_sk, (K, M*n_dim))
# print control of all shapes
print("[dp] shape tilda_ckf:", np.shape(tilda_ckf))
print("[dp] shape uk:", np.shape(uk))
print("[dp] shape delta_af:", np.shape(delta_af))
print("[dp] shape delta_gk:", np.shape(delta_gk))
print("[dp] shape delta_sk", np.shape(sorted_delta_sk))
# declare E_Align
e_align = EAlign(tilda_ckf, uk, delta_af, delta_gk, sorted_delta_sk)
# compute personalized actor-specific blendshapes
start = time.time()
delta_p = e_align.compute_actor_specific_blendshapes(vectorized=False)
print("[dp] Solved in:", time.time() - start)
print("[dp] shape delta_p", np.shape(delta_p))
print()

# 6) save delta_p ans sorted_mesh_list
np.save(save_folder + save_file_name, delta_p)
np.save(save_folder + 'sorted_mesh_name_list', sorted_mesh_list)
print("[save] saved delta_pk (actor specifik blendshapes), shape:", np.shape(delta_p))
print("[save] saved sorted_mesh_list, shape:", np.shape(delta_p))

if do_plot:
    plt.show()
