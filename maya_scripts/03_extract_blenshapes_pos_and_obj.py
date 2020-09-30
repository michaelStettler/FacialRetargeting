import maya.cmds as cmds
import numpy as np
import os

# define paremeters for the scenes
mesh_list = "D:/Maya projects/DigitalLuise/scripts/mesh_name_list.npy"
save_path = "C:/Users/Michael/PycharmProjects/FacialRetargeting/data/blendshapes_obj"
scene_name = "louise_bs_vrts_pos"
bs_groupe = "Louise_bs_GRP"


# this is a list of the vertices that matches the markers used in Viccon -> due to python 2.7 and maya I did not
# manage to use a dictionary, but the matching could be found in data/louise2david_mk2vtrs_dict.py
# todo use dictionary?
vtx_list = [2912, 2589, 2909, 2927, 399, 76, 396, 414, 1779, 155, 825, 2195, 3338, 2668, 23, 34, 1993, 2138, 2202, 4634,
            4489, 2054, 2209, 4550, 1805, 2770, 257, 831, 406, 84, 2919, 3572, 2597, 2845, 2772, 332, 259, 3393, 3344,
            880, 539]  # note that only 41 markers are present since I removed the 4 "Head" markers

# get positions of all the markers across each blendshapes
bs_vrts_pos = []
for mesh in mesh_list:
    vrts_pos = []
    for vtx in vtx_list:
        vrts_pos.append(cmds.xform(mesh+".pnts["+str(vtx)+"]", query=True,
                                  translation=True,
                                  worldSpace=True))
    bs_vrts_pos.append(vrts_pos)

    # select and save object
    cmds.select(mesh)
    cmds.file(os.path.join(save_path, mesh +".obj"), pr=1,
              typ="OBJexport",
              es=1,
              op="groups=0; ptgroups=0; materials=0; smoothing=0; normals=0;")

print("done processing vertices for (n_blendshapes, n_markers, pos):", np.shape(bs_vrts_pos))

# save vertices positions
path = 'C:/Users/Michael/PycharmProjects/FacialRetargeting/data/'
np.save(path + scene_name, bs_vrts_pos)
