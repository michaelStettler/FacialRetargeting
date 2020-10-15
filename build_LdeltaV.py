import pymesh
import numpy as np
import os
import time

# need it to run with python2 since pymesh was installed in my python2...
import sys
sys.path.insert(0, 'utils/')
sys.path.append('utils/')

from normalize_positions import normalize_positions

np.set_printoptions(precision=4, linewidth=250, suppress=True)

# define parameters
mesh_path = 'data/blendshapes_obj'
ref_mesh_name = 'Louise_Neutral'
mesh_list_name = "data/sorted_mesh_name_list.npy"  # important to use the sorted list!
save_path = "data/"
save_name = "LdV_louise_norm"

# load mesh_list
mesh_list = np.load(mesh_list_name).astype(str)
num_blendshapes = len(mesh_list)
print("num_blendshapes", num_blendshapes)


def compute_deltaV(mesh, ref_mesh, faces):
    dv = mesh - ref_mesh

    return pymesh.form_mesh(dv, faces)


def build_L_deltaV(mesh_list, path, ref_mesh_name):
    ref_mesh = pymesh.load_mesh(os.path.join(path, ref_mesh_name + ".obj"))
    faces = ref_mesh.faces
    ref_mesh_vertices, min_mesh, max_mesh = normalize_positions(np.copy(ref_mesh.vertices), return_min=True, return_max=True)
    n_vertices = len(ref_mesh_vertices)
    print("n_vertices:", n_vertices)

    LdVs = []
    for mesh_name in mesh_list:
        # compute dV
        if mesh_name != ref_mesh_name:
            mesh = pymesh.load_mesh(os.path.join(path, mesh_name+".obj"))
            mesh_vertices = normalize_positions(np.copy(mesh.vertices), min_pos=min_mesh, max_pos=max_mesh)

            dv_mesh = compute_deltaV(mesh_vertices, ref_mesh_vertices, faces)

            # compute Laplacians
            assembler = pymesh.Assembler(mesh)
            L = assembler.assemble("laplacian").todense()

            # compute LdV
            LdVs.append(np.dot(L, dv_mesh.vertices))
        else:
            print("[Warning] Ref blendshape found in the sorted mesh list name!")

    return np.array(LdVs)


# get LdV
start = time.time()
LdV = build_L_deltaV(mesh_list, mesh_path, ref_mesh_name)
print("Done computing in:", time.time() - start)

# reshape and save
LdV = np.reshape(LdV, (np.shape(LdV)[0], -1))
print("shape LdV", np.shape(LdV))
np.save(os.path.join(save_path, save_name), LdV)
print("Saved!")
