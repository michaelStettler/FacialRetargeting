import pymesh
import numpy as np
import os
from scipy import sparse

np.set_printoptions(precision=4, linewidth=250, suppress=True)

# define parameters
mesh_path = ''
ref_mesh_name = ''
mesh_list_name = 'mesh_list.npy'

# load mesh_list
mesh_list = np.load(os.join.path(mesh_path, mesh_list_name))


def compute_deltaV(mesh, ref_mesh):
    dv = mesh.vertices - ref_mesh.vertices
    faces = ref_mesh.faces
    return pymesh.form_mesh(dv, faces)


def build_L_deltaV(mesh_list, path, ref_mesh_name):
    Laplacians = []
    deltaVs = []

    ref_mesh = pymesh.load_mesh(os.path.join(path, ref_mesh_name))

    for mesh_name in mesh_list:
        if mesh_name != ref_mesh_name:
            mesh = pymesh.load_mesh(os.path.join(path, mesh_name))
            dv_mesh = compute_deltaV(mesh, ref_mesh)
        else:
            dv_mesh = pymesh.load_mesh(os.path.join(path, mesh_name))
        deltaVs.append(dv_mesh)

        assembler = pymesh.Assembler(mesh)
        L = assembler.assemble("laplacian")
        Laplacians.append(L.todense())

    return Laplacians, deltaVs