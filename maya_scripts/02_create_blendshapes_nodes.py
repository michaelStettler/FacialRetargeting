import maya.cmds as cmds

# select the folder containing all the blendshapes
bs_groupe = "Louise_bs_GRP"

# get all blendshapes' meshes
mesh_list = cmds.ls(bs_groupe, dag=1, type="mesh")  # get all blenshapes from blenshape group

# remove names issue and make a list of string instead of that "maya" list [u"", u""]
mesh_list_tuple = []
for mesh in mesh_list:
    remove_letters = 5  # somehow maya adds "Shape" at the end of the mesh
    if 'ShapeOrig' in mesh:  # ... and sometimes "ShapeOrig"
        remove_letters = 9
    # create blendshape string list
    mesh_list_tuple.append(str(mesh[:-remove_letters]))

# create a blendshape nodes for every blendshape mesh
cmds.blendShape(mesh_list_tuple, 'Louise', name="bs_node")
