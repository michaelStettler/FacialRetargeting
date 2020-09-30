import maya.cmds as cmds

# get base mesh
base_mesh = 'Louise'
# triangulate base mesh
cmds.polyTriangulate(base_mesh)

# select the folder containing all the blendshapes
bs_groupe = "Louise_bs_GRP"

# get all blendshapes' meshes
mesh_list = cmds.ls(bs_groupe, dag=1, type="mesh")  # get all blenshapes from blenshape group

# triangualte each blendshape mesh
for mesh in mesh_list:
    remove_letters = 5  # somehow maya adds "Shape" at the end of the mesh
    if 'ShapeOrig' in mesh:  # ... and sometimes "ShapeOrig"
        remove_letters = 9

    mesh_name = str(mesh[:-remove_letters])

    # triangulate mesh
    cmds.polyTriangulate(mesh_name)

    # delete history
    cmds.delete(mesh_name, ch=True)
