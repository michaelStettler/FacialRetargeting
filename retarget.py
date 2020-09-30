import os
import numpy as np

from utils.load_data import load_training_seq
from ERetarget import ERetarget

#### missing: get blendshapes and marker data from maya here

# define parameters
load_folder = "data/"
delta_p_name = "David_based_Louise_personalized_blendshapes.npy"
LdV_name = "LdV_louise.npy"
load_sequence_folder = "D:/MoCap_Data/David/NewSession_labeled/"
sequence_name = "AngerTrail05.c3d"
num_markers = 41
save_folder = "data/"
save_name = "weights_David2Louise_retarget_AngerTrail"

# load data
delta_p = np.load(os.path.join(load_folder, delta_p_name))
LdV = np.load(os.path.join(load_folder, LdV_name))
# load sequence to retarget

# define ERetarget
af = load_training_seq(load_sequence_folder, sequence_name, num_markers)

print("[data] Finish loading data")
print("[data] shape delta_p", np.shape(delta_p))
print("[data] shape LdV", np.shape(LdV))
print("[data] shape af", np.shape(af))
num_frames = np.shape(af)[0]
num_blendshapes = np.shape(delta_p)[0]
print("[data] num frames:", num_frames)
print("[data] num blendshapes:", num_frames)
print()