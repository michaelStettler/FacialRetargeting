import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay


def triangulate_vertices(vertices):
    """
    Built a triangular mesh from 3d points
    Uses the scipy Delaunay algorithm in a u,v parametrization
    see for more info about Delaunay:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

    :param vertices: flatten 3D points used for triangulation
    :return: mesh
    """
    vertices = np.reshape(vertices, (-1, 3))  # reshape mesh to express the xyz
    tri = Delaunay(vertices[:, :2])  # project into u,v # todo select which axis to project!
    # todo is there a way to project better into u, v ?

    return tri


def build_Laplacian(mesh, num_V, standard_weight=1):
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
    for vertex in range(num_V):
        # get neighbors vertices of "vertex" -> found here:
        # https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci
        v_neighbors = mesh.vertex_neighbor_vertices[1] \
            [mesh.vertex_neighbor_vertices[0][vertex]:mesh.vertex_neighbor_vertices[0][vertex + 1]]
        weights = []
        z = len(v_neighbors)
        I = I + ([vertex] * (z + 1))  # repeated row
        J = J + v_neighbors.tolist() + [vertex]  # column indices and this row
        is_anchor = False
        for v_neighbor in v_neighbors:
            # find all positions where the triangles have the 2 vertices of interest: vertex--v_neighbor
            tri_has_vertices = np.in1d(mesh.simplices, [vertex, v_neighbor])
            # reshape and convert bool to int
            tri_has_vertices = np.reshape(tri_has_vertices, (-1, 3)).astype(int)
            # get neighbor triangles of the edge (vertex--v_neighbor)
            # if the sum is 2, it means the triangle has the two vertices and thus the edge
            tri_neighbors = np.array(np.where(np.sum(tri_has_vertices, axis=1) == 2))[0]
            # compute cotangents:
            cotangents = 0
            # continue only if the edge has two triangles touching it
            if len(tri_neighbors) > 1:
                for f in range(2):
                    # get opposite vertex from the triangle f touching the edge (vertex--v_neighbor)
                    p = list(filter(lambda v: v not in [vertex, v_neighbor], mesh.simplices[tri_neighbors[f]]))
                    # get u, v -> found  in line 79-80:
                    (u, v) = (mesh.points[vertex] - mesh.points[p], mesh.points[v_neighbor] - mesh.points[p])
                    (u, v) = (u[0], v[0])
                    # compute cotangents
                    cotangents += np.dot(u, v) / np.sqrt(np.sum(np.square(np.cross(u, v))))

            else:
                # if the vertices as one triangle attached to it, uses standard_weights
                cotangents += standard_weight

            # add the weights
            weights.append(-.5 * cotangents)
        V = V + weights + [(-1 * np.sum(weights))]  # n negative weights and row vertex sum

        if is_anchor:
            anchors.append(vertex)

    L = sparse.coo_matrix((V, (I, J)), shape=(num_V, num_V)).tocsr()

    # # augment Laplacian matrix with anchor weights
    # for anchor in range(len(anchors)):
    #     a_neighbors = mesh.vertex_neighbor_vertices[1] \
    #         [mesh.vertex_neighbor_vertices[0][anchor]:mesh.vertex_neighbor_vertices[0][anchor + 1]]
    #     for a_neighbor in a_neighbors:
    #         L[anchor, a_neighbor] -= anchor_weight
    #     L[anchor, anchor] += len(a_neighbors)

    return L


if __name__ == '__main__':
    """
    test function to automatically triangulate a set of points
    
    run: python -m mesh.mesh
    """
    import matplotlib.pyplot as plt
    from utils.compute_delta import compute_delta

    np.random.seed(1)
    np.set_printoptions(precision=4, linewidth=250)
    print("--------- test toy example ----------")
    # declare variables
    n_k = 1  # num_blendshapes
    n_m = 5  # num markers
    n_n = n_m * 3  # num_features (num_markers * 3)
    dgk = np.random.rand(n_k, n_m, 3)
    print("dgk")
    print(dgk)

    # triangulate points
    print("triangulate vertices")
    tri_mesh = triangulate_vertices(dgk[0])
    print("triangles points construction:")
    print(tri_mesh.simplices)
    print("point 0", dgk[0, 0])

    # plot 3D created meshes
    # todo add a plot function into EMesh or in mesh?
    for i in range(n_k):
        points = dgk[i]
        tri_mesh = triangulate_vertices(points)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri_mesh.simplices)
        ax.set_title("Toy Example")

    # compute Laplacian matrix
    print("compute Laplacian")
    L = build_Laplacian(tri_mesh, n_m)
    print(L.todense())
    print()

    print("------------- Test triangulation with Vicon data -------------")
    # test with recorded Vicon
    sk = np.load('../data/louise_bs_vrts_pos.npy')  # sparse representation of the blend shapes (vk)
    ref_sk = sk[-1]  # neutral pose is the last one
    print("shape ref_sk", np.shape(ref_sk))
    delta_sk = compute_delta(sk[:-1], ref_sk)
    print("shape delta_sk", np.shape(delta_sk))
    # print(delta_sk)

    # plot 3D created meshes
    points = np.reshape(ref_sk, (-1, 3))
    print("shape points", np.shape(points))
    tri_mesh = triangulate_vertices(ref_sk)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri_mesh.simplices)
    ax.set_title("Neutral Pose")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    