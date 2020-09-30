import pymesh as pymesh
import os

mesh_path = 'data'
mesh_name = 'simple_cube.obj'
mesh = pymesh.load_mesh(os.path.join(mesh_path, mesh_name))
print(mesh)