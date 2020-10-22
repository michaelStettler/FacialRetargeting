import maya.cmds as cmds
import numpy as np
import os
import json

with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# get all blendshapes' meshes
mesh_list = cmds.ls(config['maya_bs_group'], dag=1, type="mesh")  # get all blenshapes from the blenshape group folder

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
cmds.blendShape(mesh_list_tuple, config['maya_base_mesh_name'], name=config['blendshape_node'])

# save mesh names
np.save(os.path.join(config['python_data_path'], config['maya_bs_mesh_list']), mesh_list_tuple)
