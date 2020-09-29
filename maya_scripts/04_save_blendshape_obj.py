import maya.cmds as cmds
import numpy as np

# define paremeters for the scenes
# bs_groupe = "Louise_bs_GRP"
bs_groupe = "group1"
save_path = '/Users/michaelstettler/Documents/maya/projects/default/'

# get all blendshapes' meshes
mesh_list = cmds.ls(bs_groupe, dag=1, type="mesh")  # get all blenshapes from blenshape group

# save all meshes into a .obj file
name_list = []
for mesh in mesh_list:
    #  get mesh name
    remove_letters = 5  # somehow maya adds "Shape" at the end of the mesh
    if 'ShapeOrig' in mesh:  # ... and sometimes "ShapeOrig"
        remove_letters = 9
    mesh_name = mesh[:-remove_letters]

    # save mesh name for order
    name_list.append(mesh_name)

    # select and save object
    cmds.select(mesh[:-5])
    cmds.file(save_path + mesh +".obj", pr=1,
              typ="OBJexport",
              es=1,
              op="groups=0; ptgroups=0; materials=0; smoothing=0; normals=0;")

# save name list
name_list = np.array(name_list)
np.save(save_path + "name_list.npy", name_list)
