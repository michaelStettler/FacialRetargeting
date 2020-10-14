import maya.cmds as cmds
import numpy as np
import os

# define paremeters for the scenes
mesh_list_name = "C:/Users/Michael/PycharmProjects/FacialRetargeting/data/mesh_name_list.npy"
save_path_pos = 'C:/Users/Michael/PycharmProjects/FacialRetargeting/data/'
save_path_obj = "C:/Users/Michael/PycharmProjects/FacialRetargeting/data/blendshapes_obj"
scene_name = "louise_to_David_markers_blendshape_vertices_pos_v2"
bs_groupe = "Louise_bs_GRP"

# load mesh_list
mesh_list = np.load(mesh_list_name).astype(str)
print("mesh list")
print(mesh_list)

# this is a list of the vertices that matches the markers used in Viccon -> due to python 2.7 and maya I did not
# manage to use a dictionary, but the matching could be found in data/louise2david_mk2vtrs_dict.py
# vtx_list = [2912, 2589, 2909, 2927, 399, 76, 396, 414, 1779, 155, 825, 2195, 3338, 2668, 23, 34, 1993, 2138, 2202, 4634,
#             4489, 2054, 2209, 4550, 1805, 2770, 257, 831, 406, 84, 2919, 3572, 2597, 2845, 2772, 332, 259, 3393, 3344,
#             880, 539]  # note that only 41 markers are present since I removed the 4 "Head" markers
# vtx_list = [414, 77, 205, 399, 1785, 2912, 2718, 2590, 2927, 880, 3393, 23, 34, 2195, 145, 2658, 826, 3339, 539, 3572,
#             1068, 3581, 97, 2610, 257, 2770, 830, 3343, 753, 3173, 742, 3255, 1988, 1980, 2202, 4477, 4484, 1963,
#             2209, 4460, 1806]
vtx_list = [2912, 2718, 2590, 2591, 399, 205, 77, 78, 1779, 401, 1804, 2208, 4304, 2914, 23, 33, 1988, 1821, 50, 4318,  # V2
            4484, 1813, 45, 4310, 1805, 2682, 2782, 169, 215, 2918, 2677, 2763, 405, 410, 250, 3266, 3255, 753, 742,
            880, 3393]
# vtx_list = [2912, 2718, 2590, 2591, 399, 205, 77, 78, 1779, 401, 1804, 2208, 4304, 2914, 23, 33, 1988, 1980, 2202, 4634,  # V3
#             4484, 1963, 2209, 4460, 1805, 2682, 2728, 169, 215, 2918, 2677, 2763, 405, 410, 250, 3266, 3255, 753, 742,
#             880, 3393]

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
    cmds.file(os.path.join(save_path_obj, mesh +".obj"), pr=1,
              typ="OBJexport",
              es=1,
              op="groups=0; ptgroups=0; materials=0; smoothing=0; normals=0;")

print("done processing vertices for (n_blendshapes, n_markers, pos):", np.shape(bs_vrts_pos))

# save vertices positions
np.save(save_path_pos + scene_name, bs_vrts_pos)
