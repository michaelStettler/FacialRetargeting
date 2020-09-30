import maya.cmds as cmds
import numpy as np
import os

save_path = "D:/Maya projects/DigitalLuise/scripts"
save_name = "mesh_name_list.npy"
# select the folder containing all the blendshapes
bs_group = "Louise_bs_GRP"
base_mesh = "Louise"

# get all blendshapes' meshes
mesh_list = cmds.ls(bs_group, dag=1, type="mesh")  # get all blenshapes from the blenshape group folder

# remove names issue and make a list of string instead of that "maya" list [u"", u""]
mesh_list_tuple = []
for mesh in mesh_list:
    remove_letters = 5  # somehow maya adds "Shape" at the end of the mesh
    if 'ShapeOrig' in mesh:  # ... and sometimes "ShapeOrig"
        remove_letters = 9
    # create blendshape string list
    mesh_list_tuple.append(str(mesh[:-remove_letters]))

print("mesh_list_tuple")
print(mesh_list_tuple)

# create a blendshape nodes for every blendshape mesh
cmds.blendShape(mesh_list_tuple, base_mesh, name="bs_node")

# save mesh names
np.save(os.path.join(save_path, save_name), mesh_list_tuple)
