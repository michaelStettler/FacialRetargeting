import os
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve

from utils.load_data import load_training_seq
from utils.load_data import get_delta_af
from src.ERetarget import ERetarget

#### missing: get blendshapes and marker data from maya here

# define parameters
load_folder = "data/"
delta_p_name = "David_based_Louise_personalized_blendshapes.npy"
LdV_name = "LdV_louise.npy"
load_sequence_folder = "D:/MoCap_Data/David/NewSession_labeled/"
sequence_name = "AngerTrail05.c3d"
num_markers = 45
save_folder = "data/"
save_name = "weights_David2Louise_retarget_AngerTrail"

# ----------------------- data -------------------------
# load data
delta_p = np.load(os.path.join(load_folder, delta_p_name))
LdV = np.load(os.path.join(load_folder, LdV_name))
# load sequence to retarget
sequence = load_training_seq(load_sequence_folder, sequence_name, num_markers)
af, delta_af = get_delta_af([sequence])
delta_af = np.delete(delta_af, (38, 39, 40, 44), 1)  # remove HEAD markers
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

weights = []
for i in tqdm(range(200)):
    eRetarget.set_af(delta_af[i])
    A, b = eRetarget.get_dERetarget()
    w = solve(A, b)
    weights.append(w)

print("[Retarget] shape weights", np.shape(weights))

# normalize weights


# save
np.save(os.path.join(save_folder, save_name), weights)
print("weights save as:", os.path.join(save_folder, save_name))
print("max weights", np.amax(weights))


