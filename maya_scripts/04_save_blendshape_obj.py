import maya.cmds as cmds
import numpy as np

# define paremeters for the scenes
# scene_name = cmds.file(q=True, sn=True, shn=True)
scene_name = "louise_bs_vrts_pos"
bs_groupe = "Louise_bs_GRP"

# get all blendshapes' meshes
mesh_list = cmds.ls(bs_groupe, dag=1, type="mesh")  # get all blenshapes from blenshape group

# todo save each blendshapes as a .obj
# todo: save order!!!!!