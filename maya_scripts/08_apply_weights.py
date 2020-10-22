import maya.cmds as cmds
import numpy as np
import os
import json

# define paremeters for the scenes
weights_name = "C:/Users/Michael/PycharmProjects/FacialRetargeting/data/weights_David2Louise_retarget_Happy_500_v3.npy"
# todo put this as a parser

# load and define parameters
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# load parameters
weights = np.load(weights_name)
mesh_list_name = np.load(os.path.join(config['python_data_path'], config['sorted_maya_bs_mesh_list'])).astype(str)
print("shape weights", np.shape(weights))
print("shape mesh_list_name", np.shape(mesh_list_name))
ref_w = weights[1]
print("shape ref_w", np.shape(ref_w))

# select node
cmds.select(config['blendshape_node'])

# apply weights for each frame
for f in range(np.shape(weights)[0]):
    print("frame", f)
    bs_idx = 0
    for bs in mesh_list_name:
        if bs != config['neutral_pose']:  # remove neutral pose
            w = max(-10, weights[f, bs_idx])
            w = min(w, 10)
            cmds.setAttr(config['blendshape_node'] + '.' + bs, w)
            bs_idx += 1
    cmds.setKeyframe(t=f)
