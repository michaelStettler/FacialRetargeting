import os
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve

import multiprocessing as mp
from functools import partial

from utils.load_data import load_training_seq
from utils.compute_delta import compute_delta
from src.ERetarget import ERetarget


def get_w(i, eRetarget, delta_af, use_L2):
    eRetarget.set_af(delta_af[i])
    A, b = eRetarget.get_dERetarget(L2=use_L2)
    return solve(A, b)


if __name__ == '__main__':
    pool = mp.Pool(processes=4)
    #### missing: get blendshapes and marker data from maya here

    # define parameters
    ref_actor_pose = 'data/David_neutral_pose.npy'
    load_folder = "data/"
    delta_p_name = "David_based_Louise_personalized_blendshapes_v3_norm_EMatch.npy"
    LdV_name = "LdV_louise.npy"
    load_sequence_folder = "D:/MoCap_Data/David/NewSession_labeled/"
    # sequence_name = "AngerTrail05.c3d"
    sequence_name = "HappyTrail01.c3d"
    # sequence_name = "FearTrail03.c3d"
    # sequence_name = "NeutralTrail14.c3d"
    num_markers = 45
    save_folder = "data/"
    save_name = "weights_David2Louise_retarget_Happy_15000_L1_v3_norm_EMatch"
    # save_name = "weights_David2Louise_retarget_FearTrail"
    use_L2 = False

    # ----------------------- data -------------------------
    # load data
    delta_p = np.load(os.path.join(load_folder, delta_p_name))
    print("max delta_p", np.amax(delta_p))
    LdV = np.load(os.path.join(load_folder, LdV_name))
    # LdV /= np.linalg.norm(LdV)  # todo normalize dV and not LdV?
    print("max ldv", np.amax(LdV))
    # load sequence to retarget
    ref_actor_pose = np.load(ref_actor_pose)
    # norm_ref = np.linalg.norm(ref_actor_pose)
    min_af = np.amin(ref_actor_pose)
    max_af = np.amax(ref_actor_pose)
    # ref_actor_pose /= norm_ref
    ref_actor_pose -= min_af
    ref_actor_pose /= max_af
    af = load_training_seq(load_sequence_folder, sequence_name, num_markers)
    # af /= norm_ref
    af -= min_af
    af /= max_af
    delta_af = compute_delta(af, ref_actor_pose)
    print("delta_af")
    print(delta_af[0])
    ref_actor_pose = ref_actor_pose[:-4, :]  # remove HEAD markers
    af = af[:, :-4, :]  # remove HEAD markers
    delta_af = delta_af[:, :-4, :]  # remove HEAD markers
    delta_af = np.reshape(delta_af, (np.shape(delta_af)[0], -1))

    print("[data] Finish loading data")
    print("[data] shape delta_p", np.shape(delta_p))
    print("[data] shape LdV", np.shape(LdV))
    print("[data] shape delta_af", np.shape(delta_af))
    num_frames = np.shape(delta_af)[0]
    num_blendshapes = np.shape(delta_p)[0]
    num_markers = np.shape(delta_p)[1] / 3
    print("[data] num frames:", num_frames)
    print("[data] num blendshapes:", num_blendshapes)
    print("[data] num_markers:", num_markers)
    print()

    # ----------------------- ERetarget -------------------------
    eRetarget = ERetarget(delta_p, LdV)

    # weights = []
    # for i in tqdm(range(500)):
    # # for i in tqdm(range(1589, 1590)):
    # # for i in tqdm(range(num_frames)):
    #     eRetarget.set_af(delta_af[i])
    #     A, b = eRetarget.get_dERetarget(L2=use_L2)
    #     w = solve(A, b)
    #     weights.append(w)

    # multiprocessing
    p_get_w = partial(get_w, eRetarget=eRetarget, delta_af=delta_af, use_L2=use_L2)
    weights = pool.map(p_get_w, tqdm(range(15000)))
    pool.close()

    print("[Retarget] shape weights", np.shape(weights))

    # normalize weights
    weights = np.array(weights)
    print("shape weights", np.shape(weights))
    print(weights[:, 0])
    max_weights = np.amax(weights)
    min_weights = np.amin(weights)
    max_index = np.argmax(weights)
    min_index = np.argmin(weights)
    print("max weights", max_weights, "at", max_index)
    print("min weights", min_weights, "at", min_index)
    # weights /= np.amax(weights)
    # save
    np.save(os.path.join(save_folder, save_name), weights)
    print("weights save as:", os.path.join(save_folder, save_name))
    print("max weights", np.amax(weights), "at", max_index)
    print("min weights", np.amin(weights), "at", min_index)


