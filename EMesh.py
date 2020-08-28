import numpy as np

from mesh import triangulate_vertices
from mesh import build_Laplacian


class EMesh:
    """
    Construct a class to compute E_Mesh as in formula 11 using a function to pass directly the personalized blendshapes
    in delta space delta_p (dp)

    k:= num_of_blendshapes
    f:= num_frames
    m:= num_markers
    n:= num_features

    """
    def __init__(self, delta_gk, v):
        self.delta_gk = delta_gk
        self.K = np.shape(self.delta_gk)[0]
        self.M = np.shape(self.delta_gk)[1]

        mesh = triangulate_vertices(v)
        self.L = build_Laplacian(mesh, self.M)

    def _laplace_op(self, v):
        """
        Apply the Laplacian operator on the vertices

        m:= num of vertices (markers)

        :param v: vertices (m, 3)
        :return:
        """

        return self.L.dot(v)

    def _emesh(self, dp):
        """
        Compute E_Mesh as in formula 11

        :param dp: delta p (k, n)
        :return: e_mesh
        """
        e_list = []
        for k in range(self.K):
            # todo remove the loop...
            e = np.linalg.norm(self._laplace_op(np.reshape(dp[k], (-1, 3)) - self.delta_gk[k]), axis=1)**2
            e_list.append(e)

        return np.sum(e_list) / self.M

    def get_eMesh(self):
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
    v0 = dp[0]

    # create EMesh object
    e_mesh = EMesh(dgk, v0)

    # control compute e_mesh
    print("compute control e_mesh")
    mesh = triangulate_vertices(v0)
    L = build_Laplacian(mesh, n_m)
    emesh_list = []
    for k in range(n_k):
        dv = np.reshape(dp[k], (-1, 3)) - dgk[k]
        l_op = L.dot(dv)
        norm = np.linalg.norm(l_op, axis=1)**2
        emesh_list.append(norm)

    emesh_ctrl = np.sum(emesh_list) / n_m
    print("emesh_ctrl =", emesh_ctrl)

    # compute e_mesh
    print("compute e_mesh")
    e_mesh_fn = EMesh(dgk, v0).get_eMesh()
    emesh = e_mesh_fn(dp)
    print("emesh =", emesh)

    assert emesh == emesh_ctrl
    print("emesh values are equal")
    print()

