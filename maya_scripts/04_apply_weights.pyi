import maya.cmds as cmds
import numpy as np

# define paremeters for the scenes
weights_name = "D:/Maya projects/DigitalLuise/data/retarget/David/weights_David2Louise_retarget_AngerTrail.npy"
mesh_list_name = 'C:/Users/Michael/PycharmProjects/FacialRetargeting/data/sorted_mesh_name_list.npy'  # important to use the sorted mesh list
neutral_pose = "Louise_Neutral"

# load parameters
weights = np.load(weights_name)
mesh_list_name = np.load(mesh_list_name).astype(str)
print("shape weights", np.shape(weights))
print("shape mesh_list_name", np.shape(mesh_list_name))

# apply weights for each frame
for f in range(np.shape(weights)[0]):
    cmds.currentTime(f)
    bs_idx = 0
    for bs in mesh_list_name:
        if bs != neutral_pose:  # remove neutral pose
            w = max(-10, weights[f, bs_idx])
            w = min(w, 10)
            cmds.setAttr('bs_node.'+ bs, w)
            bs_idx += 1
    cmds.setKeyframe()
