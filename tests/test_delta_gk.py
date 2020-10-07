import numpy as np
from src.EAlign import EMesh
from src.RBF_warp import get_initial_actor_blendshapes

np.set_printoptions(precision=3, suppress=True)

# load data
ref_sk = np.load("data/ref_sk.npy")
sorted_delta_sk = np.load("data/sorted_delta_sk.npy")
af0 = np.load("data/af_ref.npy")

print("af0")
print(af0)
print("ref_sk")
print(ref_sk)

# test delta_gk
delta_gk = get_initial_actor_blendshapes(ref_sk, af0, sorted_delta_sk)
print("shape delta_gp", np.shape(delta_gk))
print(delta_gk[0])
test_eMesh = EMesh(delta_gk)
