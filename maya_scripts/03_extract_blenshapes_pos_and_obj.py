import maya.cmds as cmds
import numpy as np
import os

# define paremeters for the scenes
mesh_list_name = "D:/Maya projects/DigitalLuise/scripts/mesh_name_list.npy"
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
vtx_list = [78, 77, 205, 399, 1779, 2912, 2718, 2590, 2591, 880, 3393, 23, 33, 2208, 2914, 405, 2918, 410, 2677, 250,
            2763, 1804, 4304, 169, 2682, 215, 2782, 1988, 753, 3266, 742, 3255, 1988, 1821, 50, 4318, 4484, 1813, 45,
            4310, 1805]
print("num_markers:", len(vtx_list))


# # get all vertices
# sel = cmds.ls(sl=True, o=True)[0]
# sel_vtx = cmds.ls('{}.vtx[:]'.format(sel), fl=True)
# print("num vertices", len(sel_vtx))
# for vtx in range(len(sel_vtx)):
#     if vtx not in vtx_list:
#         cmds.delete("Louise.pnts["+str(vtx)+"]")

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
