import os
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve
import json

import multiprocessing as mp
from functools import partial

from utils.load_data import load_training_seq
from utils.normalize_positions import normalize_positions
from utils.align_to_head_markers import align_to_head_markers
from utils.modify_axis import modify_axis
from utils.compute_delta import compute_delta
from src.ERetarget import ERetarget


def get_w(i, eRetarget, delta_af):
    eRetarget.set_af(delta_af[i])
    A, b = eRetarget.get_dERetarget()
    return solve(A, b)


if __name__ == '__main__':
    pool = mp.Pool(processes=4)
    #### missing: get blendshapes and marker data from maya here

    # load and define parameters
    with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
        config = json.load(f)

    alpha = int(config['alpha'])
    beta = int(config['beta'])
    print("[PARAMS] alpha:", alpha)
    print("[PARAMS] beta:", beta)
    mu = "0.3"
    nu = "0.6"
    start = 6060
    end = 7000

    delta_p_name = config['dp_name']+'_alpha_'+config['alpha']+'_beta_'+config['beta']+".npy"
    # sequence_name = "AngerTrail05.c3d"
    # sequence_name = "HappyTrail01.c3d"
    # sequence_name = "FearTrail03.c3d"
    sequence_name = "SadTrail01.c3d"
    # sequence_name = "SurpriseTrail02.c3d"
    # sequence_name = "DisgustTrail04.c3d"
    # sequence_name = "NeutralTrail14.c3d"
    if start is not None:
        save_name = "weights_David2Louise_retarget_"+sequence_name+"_s"+str(start)+"_alpha_"+alpha+"_beta_"+beta+"_mu_"+mu+"_nu_"+nu
        if end is not None:
            save_name = "weights_David2Louise_retarget_"+sequence_name+"_" + str(
                start) + "_e"+str(end)+"_alpha_" + alpha + "_beta_" + beta + "_mu_" + mu + "_nu_" + nu
    elif end is not None:
        save_name = "weights_David2Louise_retarget_"+sequence_name+"_e" + str(
            end) + "_alpha_" + alpha + "_beta_" + beta + "_mu_" + mu + "_nu_" + nu
    else:
        save_name = "weights_David2Louise_retarget_"+sequence_name+"_alpha_" + alpha + "_beta_" + beta + "_mu_" + mu + "_nu_" + nu

    # get actor animation
    # ----------------------- data -------------------------
    # load data
    delta_p = np.load(os.path.join(config['python_data_path'], delta_p_name))
    print("max delta_p", np.amax(delta_p))
    LdV = np.load(os.path.join(config['python_data_path'], config['LdV_name']+'.npy'))
    # LdV /= np.linalg.norm(LdV)  # todo normalize dV and not LdV?
    print("max ldv", np.amax(LdV))

    # load reference actor pose
    ref_actor_pose = np.load(os.path.join(config['python_data_path'], config['neutral_pose_positions']+'.npy'))
    # align sequence with the head markers
    head_markers = range(np.shape(ref_actor_pose)[0] - 4, np.shape(ref_actor_pose)[0] - 1)  # use only 3 markers
    ref_actor_pose = align_to_head_markers(ref_actor_pose, ref_idx=head_markers)
    ref_actor_pose = ref_actor_pose[:-4, :]  # remove HEAD markers
    # modify axis from xzy to xyz to match the scatter blendshape axis orders
    ref_actor_pose = modify_axis(ref_actor_pose, order='xzy2xyz', inverse_z=True)
    # normalize reference (neutral) actor positions
    ref_actor_pose, min_af, max_af = normalize_positions(ref_actor_pose, return_min=True, return_max=True)

    # load sequence to retarget
    af = load_training_seq(config['mocap_folder'], sequence_name, config['num_markers'], template_labels=config['template_labels'])
    af = align_to_head_markers(af, ref_idx=head_markers)
    af = af[:, :-4, :]  # remove HEAD markers
    # modify axis from xyz to xzy to match the scatter blendshape axis orders
    af = modify_axis(af, order='xzy2xyz', inverse_z=True)
    af = normalize_positions(af, min_pos=min_af, max_pos=max_af)

    # compute delta af
    delta_af = compute_delta(af, ref_actor_pose, norm_thresh=2)
    delta_af = np.reshape(delta_af, (np.shape(delta_af)[0], -1))

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(ref_actor_pose[:, 0], ref_actor_pose[:, 1], ref_actor_pose[:, 2])
    # ax.scatter(af[0, :, 0], af[0, :, 1], af[0, :, 2])
    # ax.set_title("ref pose A0 normalized")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    print("[data] Finish loading data")
    print("[data] shape delta_p", np.shape(delta_p))
    print("[data] shape LdV", np.shape(LdV))
    print("[data] shape delta_af", np.shape(delta_af))
    num_frames = np.shape(delta_af)[0]
    num_blendshapes = np.shape(delta_p)[0]
    num_markers = int(np.shape(delta_p)[1] / 3)
    print("[data] num frames:", num_frames)
    print("[data] num blendshapes:", num_blendshapes)
    print("[data] num_markers:", num_markers)
    print()

    # ----------------------- ERetarget -------------------------
    eRetarget = ERetarget(delta_p, LdV, mu=float(mu), nu=float(nu))

    if start is not None:
        delta_af = delta_af[start:]
        if end is not None:
            delta_af = delta_af[start:end]
    elif end is not None:
        delta_af = delta_af[:end]

    weights = []
    # # for i in tqdm(range(500)):
    # for i in tqdm(range(5800, 7000)):
    # # for i in tqdm(range(num_frames)):
    #     eRetarget.set_af(delta_af[i])
    #     A, b = eRetarget.get_dERetarget()
    #     w = solve(A, b)
    #     weights.append(w)

    # multiprocessing
    p_get_w = partial(get_w, eRetarget=eRetarget, delta_af=delta_af)
    weights = pool.map(p_get_w, tqdm(range(len(delta_af))))
    pool.close()

    print("[Retarget] shape weights", np.shape(weights))

    # get weights info
    weights = np.array(weights)
    max_weights = np.amax(weights)
    min_weights = np.amin(weights)
    max_index = np.argmax(weights)
    min_index = np.argmin(weights)
    print("max weights", max_weights, "at", max_index)
    print("min weights", min_weights, "at", min_index)

    # save
    np.save(os.path.join(config['python_data_path'], save_name), weights)
    print("weights save as:", os.path.join(config['python_data_path'], save_name))


