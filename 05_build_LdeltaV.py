import pymesh
import numpy as np
import os
import time
import json

# need it to run with python2 since pymesh was installed in my python2...
import sys
sys.path.insert(0, 'utils/')
sys.path.append('utils/')

from normalize_positions import normalize_positions

np.set_printoptions(precision=4, linewidth=250, suppress=True)

# load and define parameters
with open("C:/Users/Michael/PycharmProjects/FacialRetargeting/configs/David_to_Louise_v2.json") as f:
    config = json.load(f)

# load mesh_list
mesh_list = np.load(os.path.join(config['python_data_path'], config['sorted_maya_bs_mesh_list'])).astype(str)
num_blendshapes = len(mesh_list)
print("num_blendshapes", num_blendshapes)


def compute_deltaV(mesh, ref_mesh, faces):
    dv = mesh - ref_mesh

    return pymesh.form_mesh(dv, faces)


def build_L_deltaV(mesh_list, path, ref_mesh_name):
    ref_mesh = pymesh.load_mesh(os.path.join(path, ref_mesh_name + ".obj"))
    faces = ref_mesh.faces
    # ref_mesh_vertices, min_mesh, max_mesh = normalize_positions(np.copy(ref_mesh.vertices), return_min=True, return_max=True)
    ref_mesh_vertices = ref_mesh.vertices
    n_vertices = len(ref_mesh_vertices)
    print("n_vertices:", n_vertices)

    LdVs = []
    for mesh_name in mesh_list:
        # compute dV
        if mesh_name != ref_mesh_name:
            mesh = pymesh.load_mesh(os.path.join(path, mesh_name+".obj"))
            # mesh_vertices = normalize_positions(np.copy(mesh.vertices), min_pos=min_mesh, max_pos=max_mesh)
            mesh_vertices = mesh.vertices
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
LdV = build_L_deltaV(mesh_list, config['python_save_path_bsObj'], config['neutral_pose'])
print("Done computing in:", time.time() - start)

# reshape and save
LdV = np.reshape(LdV, (np.shape(LdV)[0], -1))
print("shape LdV", np.shape(LdV))
np.save(os.path.join(config['python_data_path'], config['LdV_name']), LdV)
print("Saved!")
