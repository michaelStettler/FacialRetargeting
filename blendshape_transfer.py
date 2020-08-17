import numpy as np

from load_data import load_training_frames
from compute_delta import compute_delta
from compute_corr_coef import compute_corr_coef

"""
run: python -m blendshape_transfer
"""

# 1) Facial Motion Similarity
# measure similarity between character blendshapes and actor's capture performance
sk = np.load('data/louise_bs_vrts_pos.npy')  # sparse representation of the blend shapes (vk)
print("shape sk:", np.shape(sk))
ref_sk = sk[-1]
delta_sk = compute_delta(sk[:-1], ref_sk)
delta_sk = delta_sk[:5]
print("shape delta_sk", np.shape(delta_sk))
af, delta_af = load_training_frames('D:/MoCap_Data/David/NewSession_labeled/', num_markers=45, max_num_seq=5)  # actor animation frames
print("shape delta af:", np.shape(delta_af))
# todo remove the HEAD markers!
ckf = compute_corr_coef(delta_af, delta_sk)

# 2) extract most important frames

# 3) build initial guess blend shape RBF wrap