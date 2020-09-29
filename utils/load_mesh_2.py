import pymesh
import numpy as np
from scipy import sparse

np.set_printoptions(precision=4, linewidth=250, suppress=True)

# mesh = pymesh.load_mesh("../data/simple_cube.obj")
mesh = pymesh.load_mesh("../data/simple_strange_cube.obj")
# mesh = pymesh.load_mesh("../data/teapot.obj")  # carefull teapot seems to have double vertices! my gradient does not work for this
num_V = len(mesh.vertices)
print("num_vertices", num_V)

mesh.enable_connectivity()
neighbours = mesh.get_vertex_adjacent_vertices(0)
print(neighbours)
# print("control teapot:", 2659, 2683, 2773, 2837, 2937, 2984)
print("control simple_cube:", 1, 2, 4, 6, 7)

assembler = pymesh.Assembler(mesh)
L = assembler.assemble("laplacian")
print(type(L))
print(np.shape(L))


def build_Laplacian(mesh, num_V, anchor_weight=1, is_pymesh=False):
    """
    Build a Laplacian sparse matrix between the vertices of a triangular mesh
    This mainly follow work from:
    "Laplacian Mesh Optimization" (Andrew Nealean et al. 2006)
    and an implementation from:
    https://github.com/bmershon/laplacian-meshes/blob/master/LaplacianMesh.py

    :param mesh:
    :param num_V: num_vertices
    :return:
    """

    # declare variables to build sparse matrix
    I = []
    J = []
    V = []

    # find anchoring points (all the vertices they belong to does not have two triangles attached two it)
    anchors = []

    # built sparse Laplacian matrix with cotangent weights
    # for vertex in range(num_V):
    for vertex in range(num_V):
        # get neighbors vertices of "vertex" -> found here:
        # https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci
        if is_pymesh:
            v_neighbors = mesh.get_vertex_adjacent_vertices(vertex)
        else:
            v_neighbors = mesh.vertex_neighbor_vertices[1] \
                [mesh.vertex_neighbor_vertices[0][vertex]:mesh.vertex_neighbor_vertices[0][vertex + 1]]
        weights = []
        z = len(v_neighbors)
        I = I + ([vertex] * (z + 1))  # repeated row
        J = J + v_neighbors.tolist() + [vertex]  # column indices and this row
        is_anchor = False
        # for v_neighbor in v_neighbors:
        for v_neighbor in v_neighbors:
            if is_pymesh:
                # get faces that touches the vertex
                vertex_faces = mesh.get_vertex_adjacent_faces(vertex)
                #  get faces that touches the second vertex
                v_neigh_faces = mesh.get_vertex_adjacent_faces(v_neighbor)
                # keep only the faces that has the two vertices in common
                common_faces = vertex_faces[np.nonzero(np.in1d(vertex_faces, v_neigh_faces))[0]]
            else:
                # find all positions where the triangles have the 2 vertices of interest: vertex--v_neighbor
                tri_has_vertices = np.in1d(mesh.simplices, [vertex, v_neighbor])
                # reshape and convert bool to int
                tri_has_vertices = np.reshape(tri_has_vertices, (-1, 3)).astype(int)
                # get neighbor triangles of the edge (vertex--v_neighbor)
                # if the sum is 2, it means the triangle has the two vertices and thus the edge
                common_faces = np.array(np.where(np.sum(tri_has_vertices, axis=1) == 2))[0]

            # compute cotangents:
            cotangents = 0
            # continue only if the edge has two triangles touching it
            if len(common_faces) > 1:
                for f in range(2):
                    # get opposite vertex from the triangle f touching the edge (vertex--v_neighbor)
                    if is_pymesh:
                        p = list(filter(lambda v: v not in [vertex, v_neighbor], mesh.faces[common_faces[f]]))
                    else:
                        p = list(filter(lambda v: v not in [vertex, v_neighbor], mesh.simplices[common_faces[f]]))
                    # get u, v -> found  in line 79-80:
                    if is_pymesh:
                        (u, v) = (mesh.vertices[vertex] - mesh.vertices[p], mesh.vertices[v_neighbor] - mesh.vertices[p])
                    else:
                        (u, v) = (mesh.points[vertex] - mesh.points[p], mesh.points[v_neighbor] - mesh.points[p])  # todo compute cotangents in u, v or in xyz?....
                    (u, v) = (u[0], v[0])
                    # compute cotangents
                    cotangents += np.dot(u, v) / np.sqrt(np.sum(np.square(np.cross(u, v))))

                    # if np.cross(u, v).all() == 0:
                    #     cotangents += 1
                    #     print("Come to cross == 0 !!!!!!!!!")
                    # else:
                    #     cotangents += np.dot(u, v) / np.sqrt(np.sum(np.square(np.cross(u, v))))
            else:
                # if the vertices as one triangle attached to it, it means that the vertices is an edge and therefore,
                # the vertices is an anchor
                print("----------- is anchor")
                is_anchor = True

            # add the weights
            weights.append(-.5 * cotangents)
        V = V + weights + [(-1 * np.sum(weights))]  # n negative weights and row vertex sum

        if is_anchor:
            anchors.append(vertex)

    L = sparse.coo_matrix((V, (I, J)), shape=(num_V, num_V)).tocsr()

    print("anchors", len(anchors))

    # augment Laplacian matrix with anchor weights
    # todo: add to the already exisiting values or keep it ?
    for anchor in range(len(anchors)):
        # get anchor neighbours vertices
        if is_pymesh:
            a_neighbors = mesh.get_vertex_adjacent_vertices(0)
        else:
            a_neighbors = mesh.vertex_neighbor_vertices[1] \
                [mesh.vertex_neighbor_vertices[0][anchor]:mesh.vertex_neighbor_vertices[0][anchor + 1]]

        for a_neighbor in a_neighbors:
            L[anchor, a_neighbor] -= anchor_weight
        L[anchor, anchor] += len(a_neighbors)

    return L

L_compute = build_Laplacian(mesh, num_V, is_pymesh=True)
print(L_compute.todense())
print()
print(L.todense())

# todo: subtract a mesh by another one!