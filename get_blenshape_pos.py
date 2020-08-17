import maya.cmds as cmds
import numpy as np

# define paremeters for the scenes
# scene_name = cmds.file(q=True, sn=True, shn=True)
scene_name = "louise_bs_vrts_pos"
bs_groupe = "Louise_bs_GRP"

# get all blendshapes' meshes
mesh_list = cmds.ls(bs_groupe, dag=1, type="mesh")  # get all blenshapes from blenshape group
rem_objects = 21  # louise blendshape group has 21 weird names at the end, but it seems to be nothing # todo ask Nick!!
mesh_list = mesh_list[:-rem_objects]

# this is a list of the vertices that matches the markers used in Viccon -> due to python 2.7 and maya I did not
# manage to use a dictionary, but the matching could be found in data/louise2david_mk2vtrs_dict.py
# todo use dictionary
vtx_list = [2912, 2589, 2909, 2927, 399, 76, 396, 414, 1779, 155, 825, 2195, 3338, 2668, 23, 34, 1993, 2138, 2202, 4634,
            4489, 2054, 2209, 4550, 1805, 2770, 257, 831, 406, 84, 2919, 3572, 2597, 2845, 2772, 332, 259, 3393, 3344,
            880, 539]  # note that only 41 markers are present since I removed the 4 "Head" markers

# get positions of all the markers across each blendshapes
bs_vrts_pos = []
for bs in mesh_list:
    remove_letters = 5  # somehow maya adds "Shape" at the end of the mesh
    if 'ShapeOrig' in bs:  # ... and sometimes "ShapeOrig"
        remove_letters = 9
    vrts_pos = []
    for vtx in vtx_list:
        vrts_pos.append(cmds.xform(str(bs[:-remove_letters])+".pnts["+str(vtx)+"]", query=True,
                                  translation=True,
                                  worldSpace=True))
    bs_vrts_pos.append(vrts_pos)

print("done processing vertices, found (n_bs, n_markers, pos):", np.shape(bs_vrts_pos))

# save vertices positions
path = 'C:/Users/Michael/PycharmProjects/FacialRetargeting/data/'
np.save(path + scene_name, bs_vrts_pos)
