import maya.cmds as cmds
import json

with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# triangulate base mesh
cmds.polyTriangulate(config['maya_base_mesh_name'])

# get all blendshapes' meshes
mesh_list = cmds.ls(config['maya_bs_group'], dag=1, type="mesh")  # get all blenshapes from blenshape group

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
