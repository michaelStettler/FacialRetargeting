import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay


class EMesh:
    """
    Construct a class to compute E_Mesh as in formula 11 using a function to pass directly the personalized blendshapes
    in delta space delta_p (dp)

    k:= num_of_blendshapes
    f:= num_frames
    m:= num_markers
    n:= num_features

    """
    def __init__(self, delta_gk):
        self.delta_gk = delta_gk
        self.M = np.shape(self.delta_gk)[1]

    def _triangulate_vertices(self, vertices):
        """
        Built a triangular mesh from 3d points
        Uses the scipy Delaunay algorithm in a u,v parametrization
        see for more info about Delaunay:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

        :param vertices: 3D points used for triangulation
        :return: mesh
        """
        vertices = np.reshape(vertices, (-1, 3))  # reshape mesh to express the xyz
        tri = Delaunay(vertices[:, :2])  # project into u,v # todo select which axis to project!
        # todo is there a way to project correctly into u, v ?

        return tri

    def _build_Laplacian(self, mesh):
        """
        Build a Laplacian sparse matrix between the vertices of a triangular mesh
        This mainly follow work from:
        "Laplacian Mesh Optimization" (Andrew Nealean et al. 2006)
        and an implementation from:
        https://github.com/bmershon/laplacian-meshes/blob/master/LaplacianMesh.py

        :param mesh:
        :return:
        """
        # declare variables to build sparse matrix
        I = []
        J = []
        V = []

        # built sparse Laplacian matrix with cotangent weights
        for vertex in range(self.M):
            # get neighbors vertices of "vertex" -> found here:
            # https://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci
            v_neighbors = mesh.vertex_neighbor_vertices[1]\
                [mesh.vertex_neighbor_vertices[0][vertex]:mesh.vertex_neighbor_vertices[0][vertex+1]]
            weights = []
            z = len(v_neighbors)
            I = I + ([vertex] * (z + 1))  # repeated row
            J = J + v_neighbors.tolist() + [vertex]  # column indices and this row
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
                        (u, v) = (mesh.points[vertex] - mesh.points[p], mesh.points[v_neighbor] - mesh.points[p])  # todo compute cotangents in u, v or in xyz?....
                        (u, v) = (u[0], v[0])
                        # compute cotangents
                        cotangents += np.dot(u, v) / np.sqrt(np.sum(np.square(np.cross(u, v))))
                weights.append(-.5 * cotangents)
            V = V + weights + [(-1 * np.sum(weights))]  # n negative weights and row vertex sum

        L = sparse.coo_matrix((V, (I, J)), shape=(self.M, self.M)).tocsr()
        # todo use anchors?

        return L

    def _laplace_op(self, v):
        """
        Apply the Laplacian operator on the vertices

        m:= num of vertices (markers)

        :param v: vertices (m, 3)
        :return:
        """
        mesh = self._triangulate_vertices(v)
        L = self._build_Laplacian(mesh)
        return L.dot(v)

    def _emesh(self, dp):
        """
        Compute E_Mesh as in formula 11

        :param dp: delta p (k, n)
        :return: e_mesh
        """
        e_list = []
        for k in range(n_k):
            # todo remove the loop...
            e = np.linalg.norm(self._laplace_op(np.reshape(dp[k], (-1, 3)) - self.delta_gk[k]), axis=1)**2
            e_list.append(e)

        return np.sum(e_list) / self.M

    def get_emesh(self):
        """
        return the function emesh
        :return:
        """
        return self._emesh


if __name__ == '__main__':
    """
    test e_mesh functions
    
    1st part build a random array
    2nd part triangulate a set of markers from Vicon recording into a mesh
    
    run: python -m EMesh
    """
    import matplotlib.pyplot as plt
    from compute_delta import compute_delta

    np.random.seed(2)
    print("--------- test toy example ----------")
    # declare variables
    n_k = 1  # num_blendshapes
    n_m = 5  # num markers
    n_n = n_m * 3  # num_features (num_markers * 3)
    dgk = np.random.rand(n_k, n_m, 3)
    dp = np.random.rand(n_k, n_n)

    # create EMesh object
    e_mesh = EMesh(dgk)

    # triangulate points
    print("triangulate vertices")
    tri_mesh = e_mesh._triangulate_vertices(dp[0])

    # plot 3D created meshes
    # todo add a plot function into EMesh?
    for i in range(n_k):
        points = np.reshape(dp[i], (-1, 3))
        tri_mesh = e_mesh._triangulate_vertices(dp[i])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri_mesh.simplices)
        ax.set_title("Toy Example")

    # compute Laplacian matrix
    print("compute Laplacian")
    L = e_mesh._build_Laplacian(tri_mesh)

    # control compute e_mesh
    print("compute control e_mesh")
    emesh_list = []
    for k in range(n_k):
        dv = np.reshape(dp[k], (-1, 3)) - dgk[k]
        mesh = e_mesh._triangulate_vertices(dv)
        L = e_mesh._build_Laplacian(mesh)
        l_op = L.dot(dv)
        norm = np.linalg.norm(l_op, axis=1)**2
        emesh_list.append(norm)

    emesh_ctrl = np.sum(emesh_list) / n_m
    print("emesh_ctrl =", emesh_ctrl)

    # compute e_mesh
    print("compute e_mesh")
    e_mesh_fn = EMesh(dgk).get_emesh()
    emesh = e_mesh_fn(dp)
    print("emesh =", emesh)

    assert emesh == emesh_ctrl
    print("emesh values are equal")
    print()

    print("------------- Test triangulation with Vicon data -------------")
    # test with recorded Vicon
    sk = np.load('data/louise_bs_vrts_pos.npy')  # sparse representation of the blend shapes (vk)
    ref_sk = sk[-1]  # neutral pose is the last one
    print("shape ref_sk", np.shape(ref_sk))
    delta_sk = compute_delta(sk[:-1], ref_sk)
    print("shape delta_sk", np.shape(delta_sk))
    # print(delta_sk)

    # plot 3D created meshes
    points = np.reshape(ref_sk, (-1, 3))
    print("shape points", np.shape(points))
    tri_mesh = e_mesh._triangulate_vertices(ref_sk)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri_mesh.simplices)
    ax.set_title("Neutral Pose")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
