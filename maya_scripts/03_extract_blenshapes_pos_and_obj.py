import maya.cmds as cmds
import numpy as np
import json
import os

# load parameters for the scenes
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# load mesh_list
mesh_list = np.load(os.path.join(config['python_data_path'], config['maya_bs_mesh_list']+'.npy')).astype(str)
print("mesh list")
print(mesh_list)

vtx_list = np.array(config['vrts_pos']).astype(int)
print("num_markers:", len(vtx_list))

# get positions of all the markers across each blendshapes
bs_vrts_pos = []
for mesh in mesh_list:
    print("mesh", mesh)
    vrts_pos = []
    for vtx in vtx_list:
        vrts_pos.append(cmds.xform(mesh+".pnts["+str(vtx)+"]", query=True,
                                  translation=True,
                                  worldSpace=True))
    bs_vrts_pos.append(vrts_pos)

    # select and save object
    cmds.select(mesh)
    cmds.file(os.path.join(config['python_save_path_bsObj'], mesh +".obj"), pr=1,
              typ="OBJexport",
              es=1,
              op="groups=0; ptgroups=0; materials=0; smoothing=0; normals=0;")

print("done processing vertices for (n_blendshapes, n_markers, pos):", np.shape(bs_vrts_pos))

# save vertices positions
np.save(os.path.join(config['python_data_path'], config['vertices_pos_name']), bs_vrts_pos)
