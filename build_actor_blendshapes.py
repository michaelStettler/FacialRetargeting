import numpy as np
import time as time
import os
import matplotlib.pyplot as plt

from utils.load_data import load_training_frames
from utils.compute_delta import compute_delta
from utils.remove_neutral_blendshape import remove_neutral_blendshape
from utils.normalize_positions import normalize_positions
from utils.align_to_head_markers import align_to_head_markers
from utils.modify_axis import modify_axis
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
np.set_printoptions(precision=4, linewidth=200, suppress=True)

# define parameters
actor_recording_data_folder = 'D:/MoCap_Data/David/NewSession_labeled/'
blendshape_mesh_list_name = "C:/Users/Michael/PycharmProjects/FacialRetargeting/data/mesh_name_list.npy"
load_folder = 'data/'
sparse_blendhsape_vertices_pos_name = "louise_to_David_markers_blendshape_vertices_pos_v2.npy"
save_folder = 'data/'
save_file_name = "David_based_Louise_personalized_blendshapes_v2_NewEMesh_alpha_0.0001.npy"
neutral_pose_name = 'Louise_Neutral'
ref_actor_pose = 'data/David_neutral_pose.npy'
max_num_seq = None  # set to None if we want to use all the sequences
do_plot = True
save = True
load_pre_processed = True

# load data
mesh_list = np.load(blendshape_mesh_list_name).astype(str)
sk = np.load(os.path.join(load_folder, sparse_blendhsape_vertices_pos_name))  # sparse representation of the blendshapes (vk)
# get Neutral ref index and new cleaned mesh list
cleaned_mesh_list, bs_index, ref_index = remove_neutral_blendshape(mesh_list, neutral_pose_name)

# get neutral (reference) blendshape and normalize it
ref_sk, min_sk, max_sk = normalize_positions(np.copy(sk[ref_index]), return_min=True, return_max=True)

# normalize sk
sk = normalize_positions(sk, min_pos=min_sk, max_pos=max_sk)

if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(ref_sk[:, 0], ref_sk[:, 1], ref_sk[:, 2])
    ax.set_title("Ref sk")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

# compute delta sparse blendshape
delta_sk = compute_delta(sk[bs_index, :, :], ref_sk)

# test if delta_sk has no none-unique entry
test_unique = np.unique(delta_sk, axis=1)
if np.shape(test_unique)[0] != np.shape(delta_sk)[0]:
    raise ValueError("delta_sk contains non unique entry! Check your index dictionary to build the sparse blendshape "
                     "(maya_scripts.03_extract_blendshapes_pos_and_obj)")

# get actor animation
template_labels = ['LeftBrow1', 'LeftBrow2', 'LeftBrow3', 'LeftBrow4', 'RightBrow1', 'RightBrow2', 'RightBrow3',
                 'RightBrow4', 'Nose1', 'Nose2', 'Nose3', 'Nose4', 'Nose5', 'Nose6', 'Nose7', 'Nose8',
                 'UpperMouth1', 'UpperMouth2', 'UpperMouth3', 'UpperMouth4', 'UpperMouth5', 'LowerMouth1',
                 'LowerMouth2', 'LowerMouth3', 'LowerMouth4', 'LeftOrbi1', 'LeftOrbi2', 'RightOrbi1', 'RightOrbi2',
                 'LeftCheek1', 'LeftCheek2', 'LeftCheek3', 'RightCheek1', 'RightCheek2', 'RightCheek3',
                 'LeftJaw1', 'LeftJaw2', 'RightJaw1', 'RightJaw2', 'LeftEye1', 'RightEye1', 'Head1', 'Head2',
                 'Head3', 'Head4']

# load ref pose
ref_actor_pose = np.load(ref_actor_pose)
# align sequence with the head markers
head_markers = range(np.shape(ref_actor_pose)[0] - 4, np.shape(ref_actor_pose)[0] - 1)  # use only 3 markers
ref_actor_pose = align_to_head_markers(ref_actor_pose, ref_idx=head_markers)

if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ref_actor_pose[:, 0], ref_actor_pose[:, 1], ref_actor_pose[:, 2])
    ax.set_title("ref pose A0")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

ref_actor_pose = ref_actor_pose[:-4, :]  # remove HEAD markers
# modify axis from xzy to xyz to match the scatter blendshape axis orders
ref_actor_pose = modify_axis(ref_actor_pose, order='xzy2xyz', inverse_z=True)
# normalize reference (neutral) actor positions
ref_actor_pose, min_af, max_af = normalize_positions(ref_actor_pose, return_min=True, return_max=True)

if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(ref_actor_pose[:, 0], ref_actor_pose[:, 1], ref_actor_pose[:, 2])
    ax.set_title("ref pose A0 normalized")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

if load_pre_processed:
    delta_af = np.load("data/training_delta_af.npy")
    tilda_ckf = np.load("data/training_tilda_ckf.npy")
else:
    # load sequence
    af = load_training_frames(actor_recording_data_folder,
                              num_markers=45,
                              template_labels=template_labels,
                              max_num_seq=max_num_seq,
                              down_sample_factor=5)
    af = align_to_head_markers(af, ref_idx=head_markers)
    af = af[:, :-4, :]  # remove HEAD markers
    # modify axis from xyz to xzy to match the scatter blendshape axis orders
    af = modify_axis(af, order='xzy2xyz', inverse_z=True)
    af = normalize_positions(af, min_pos=min_af, max_pos=max_af)

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(ref_actor_pose[:, 0], ref_actor_pose[:, 1], ref_actor_pose[:, 2])
        ax.scatter(af[0, :, 0], af[0, :, 1], af[0, :, 2], c='RED')
        # ax.scatter(af[5, :, 0], af[5, :, 1], af[5, :, 2], c='RED')
        # ax.scatter(af[10, :, 0], af[10, :, 1], af[10, :, 2], c='RED')
        # ax.scatter(af[2575, :, 0], af[2575, :, 1], af[2575, :, 2], c='YELLOW')
        ax.set_title("ref_pose A0 vs. af[0]")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    delta_af = compute_delta(af, ref_actor_pose, norm_thresh=2)

    print("[data] shape af:", np.shape(af))

print("[data] Finished loading data")
print("[data] Neutral Blendshape index:", ref_index)
print("[data] shape ref_actor_pose", np.shape(ref_actor_pose))
print("[data] shape delta af:", np.shape(delta_af))
print("[data] shape sk", np.shape(sk))
print("[data] shape delta_sk", np.shape(delta_sk))
print("[data] cleaned_mesh_list:", len(cleaned_mesh_list))

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
sorted_mesh_list = np.array(cleaned_mesh_list)[sorted_index]
print("[Pre-processing] shape sorted_delta_sk", np.shape(sorted_delta_sk))
print("[Pre-processing] len sorted_mesh_list", len(sorted_mesh_list))

if not load_pre_processed:
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
    F = len(key_expressions_idx)
    delta_af = delta_af[key_expressions_idx, :, :]
    tilda_ckf = tilda_ckf[:, key_expressions_idx]
    print("[Key Expr. Extract.] Keep", F, "frames")
    print("[Key Expr. Extract.] shape key_expressions", np.shape(key_expressions_idx))
    print("[Key Expr. Extract.] shape delta_af", np.shape(delta_af))
    print("[Key Expr. Extract.] shape tilda_ckf", np.shape(tilda_ckf))
    print()
    np.save("data/training_delta_af", delta_af)
    np.save("data/training_tilda_ckf", tilda_ckf)

# 3) Manifold Alignment
# built soft max vector
uk = get_soft_mask(sorted_delta_sk)
print("[SoftMax] shape uk", np.shape(uk))
print()

# 4) Geometric Constraint
# build initial guess blendshape using RBF wrap (in delta space)
delta_gk = get_initial_actor_blendshapes(ref_sk, ref_actor_pose, sorted_delta_sk)
print("[RBF Wrap] shape delta_gk", np.shape(delta_gk))
print()

if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    bs_idx = 0
    sk0 = ref_sk + sorted_delta_sk[bs_idx]
    ax.plot_trisurf(sk0[:, 0], sk0[:, 1], sk0[:, 2], alpha=0.6)
    gk0 = ref_sk + delta_gk[bs_idx]
    ax.plot_trisurf(gk0[:, 0], gk0[:, 1], gk0[:, 2], alpha=0.6)
    ax.set_title("delta sk[{}] vs. initial actor blendshape gk[{}]".format(bs_idx, bs_idx))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

# 5) build personalized actor-specific blendshapes (delta_p)
# reshape to match required dimensions
delta_af = np.reshape(delta_af, (F, M*n_dim))
sorted_delta_sk = np.reshape(sorted_delta_sk, (K, M*n_dim))
# print control of all shapes
print("[dp] shape tilda_ckf:", np.shape(tilda_ckf))
print("[dp] shape uk:", np.shape(uk))
print("[dp] shape delta_af:", np.shape(delta_af))
print("[dp] shape delta_gk:", np.shape(delta_gk))
print("[dp] shape delta_sk", np.shape(sorted_delta_sk))
# declare E_Align
e_align = EAlign(tilda_ckf, uk, delta_af, delta_gk, sorted_delta_sk, alpha=0.0001)
# compute personalized actor-specific blendshapes
start = time.time()
delta_p = e_align.compute_actor_specific_blendshapes(vectorized=False)
print("[dp] Solved in:", time.time() - start)
print("[dp] shape delta_p", np.shape(delta_p))
print()

# 6) save delta_p ans sorted_mesh_list
if save:
    np.save(save_folder + save_file_name, delta_p)
    np.save(save_folder + 'sorted_mesh_name_list', sorted_mesh_list)
    print("[save] saved delta_pk (actor specifik blendshapes), shape:", np.shape(delta_p))
    print("[save] saved sorted_mesh_list, shape:", np.shape(delta_p))

if do_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    gk0 = ref_sk + delta_gk[bs_idx]
    ax.plot_trisurf(gk0[:, 0], gk0[:, 1], gk0[:, 2])
    delta_p = np.reshape(delta_p, (K, M, n_dim))
    pk0 = ref_sk + delta_p[bs_idx]
    ax.plot_trisurf(pk0[:, 0], pk0[:, 1], pk0[:, 2], alpha=0.6)
    ax.set_title("initial dgk[{}] vs. optimized dpk[{}]".format(bs_idx, bs_idx))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_trisurf(sk0[:, 0], sk0[:, 1], sk0[:, 2], alpha=0.6)
    ax.plot_trisurf(pk0[:, 0], pk0[:, 1], pk0[:, 2], alpha=1.0)
    ax.set_title("sk[{}] vs. optimized dpk[{}]".format(bs_idx, bs_idx))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

if do_plot:
    plt.show()
