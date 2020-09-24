import numpy as np
from compute_corr_coef import compute_corr_coef

class ECEG:
    """
    Construct a class to compute E_CEG as in formula 13 using a function to pass directly the personalized blendshapes
    in delta space delta_p (dp)

    k:= num_of_blendshapes
    f:= num_frames
    m:= num_markers
    n:= num_features
    """

    def __init__(self, delta_sk):
        self.delta_sk = delta_sk
        self.K = np.shape(delta_sk)[0]
        self.N = np.shape(delta_sk)[1]
        self.M = int(self.N/3)



    def _compute_gl_operator(self, dsk):
        """
        Compute the graph Laplacian operator as formula 12

        :param dsk: (k, n)
        :return: graph Laplacian
        """
        L = self.get_graph_laplacian(dsk)

        return L @ dsk

    def get_graph_laplacian(self, dsk):
        """
        Get the Laplacian linear decomposition af formula 12 L_op(dsk) in the form L @ dsk

        L is computed by dividing the problem into a left (Sl -> a) and the right (Sk -> b) part such as L = a - b with:
            - a is the coefficient of the Ckl matrix divided by the norm over each row without the diagonal
            - b is a diagonal matrix in the form sum(dsk) / sum(|dsk|) over each row

        k:= num_of_blendshapes
        n:= num_of_features

        :param dsk: (k, n)
        :return: L (k, n)
        """
        # get ckl and remove diagonal
        ckl = compute_corr_coef(dsk, dsk)
        np.fill_diagonal(ckl, 0)  # for l!=k

        # compute decomposition of signed graph
        # compute sum(|ckl|)
        norm_ckl = np.sum(np.abs(ckl), axis=1)
        # compute sL coeff (left side)
        a = ckl / np.repeat(np.expand_dims(norm_ckl, axis=1), self.K, axis=1)
        # compute sk coeff (right side)
        sum_ckl = np.sum(ckl, axis=1)
        b = np.diag(sum_ckl / norm_ckl)
        # built L by subtracting b from a
        L = a - b

        return L

    def get_decomposition(self, dpk, dsk):
        L = self.get_graph_laplacian(dpk)
        c = -self.get_graph_laplacian(dsk)

        return L, c

    def _eceg(self, dp):
        """
        Compute E_CEG as in formula 13

        :param dp: delta p (k, n)
        :return: e_mesh
        """
        if len(np.shape(dp)) < 2:
            dp = np.reshape(dp, (self.K, self.N))

        # compute graph Laplacian
        gL = self._compute_gl_operator(dp - self.delta_sk)

        # compute norm
        norm = np.linalg.norm(gL, axis=1) ** 2

        return np.sum(norm) / self.K

    def get_eCEG(self):
        """
        return the function eceg
        :return:
        """
        return self._eceg

    def get_dECEG(self):
        print("L", np.shape(L))
        # todo...


if __name__ == '__main__':

    np.random.seed(1)
    # declare variables
    n_k = 4  # num_blendshapes
    n_m = 2  # num markers
    n_n = n_m * 3  # num_features (num_markers * 3)
    dsk = np.random.rand(n_k, n_n)
    pk = np.random.rand(n_k, n_n)
    print("shape pk", np.shape(pk))
    print(pk)

    # compute displacement and similarity
    dis_dsk = pk - dsk
    ckl = compute_corr_coef(dis_dsk, dis_dsk)

    # compute graph Laplacian
    e_ceg = 0
    for k in range(n_k):
        w = 0
        c_abs = 0
        for l in range(n_k):
            if l != k:
                ds = dis_dsk[l] - dis_dsk[k]
                w += ckl[k, l] * ds

                c_abs += np.abs(ckl[k, l])
        L = w / c_abs

        norm = np.linalg.norm(L) ** 2
        e_ceg += norm

    eceg_ctrl = e_ceg / n_k  # todo in the paper the normaization is done using M ?!?
    print("eceg_ctrl = ", eceg_ctrl)

    # compute eCEG
    e_CEG = ECEG(dsk)
    e_ceg_fn = e_CEG.get_eCEG()
    eceg = e_ceg_fn(pk)
    print("eceg =", eceg)

    assert np.around(eceg, 6) == np.around(eceg_ctrl, 6)
    print("emesh values are equal")
    print()

    print("------------- test L@pk decomposition --------------")
    Lpk_test = e_CEG._compute_gl_operator(pk)
    print("shape Lpk_test", np.shape(Lpk_test))
    print(Lpk_test)
    # build matrix L to form the L*pk
    L = e_CEG.get_graph_laplacian(pk)
    Lpk = L @ pk
    print("shape Lpk", np.shape(Lpk))
    print(Lpk)
    assert Lpk.all() == Lpk_test.all()
    print("L Decomposition works!")
    print()

    # print("------------- test L@pk + c decomposition --------------")
    # # build matrix c to get the L_op(dsk) to get the form of L_op(dpk -dsk) = L @ dpk + c
    # Lsk = e_CEG.get_graph_laplacian(dsk) @ dsk
    # print("c", np.shape(Lsk))
    # print(Lsk)
    #
    # # test id decomposition into L@pk + c works
    # dps = pk - dsk
    # gl_dps_test = e_CEG._compute_gl_operator(dps)
    # print("shape gl_dps_test", np.shape(gl_dps_test))
    # print(gl_dps_test)
    #
    # gl_dps = Lpk - Lsk
    # print("shape gl_dps", np.shape(gl_dps))
    # print(gl_dps)
    #
    # p = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # e_CEG = ECEG(np.random.rand(np.shape(p)[0], np.shape(p)[0]))
    # Lp = e_CEG.get_graph_laplacian(p) @ p
    # print("Lp")
    # print(Lp)
    # s = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    # Ls = e_CEG.get_graph_laplacian(s) @ s
    # print("Ls")
    # print(Ls)
    # ps = p - s
    # Lps = e_CEG.get_graph_laplacian(ps) @ ps
    # print("Lps")
    # print(Lps)
    # print(Lp - Ls)
    # print()

    print("try minimize")
    # try optimization
    from scipy import optimize
    opt = optimize.minimize(e_CEG.get_eCEG(), pk, method="CG")
    print(opt)

    print("try solve")
    from scipy.linalg import solve

    e_CEG.get_dECEG()
    # print("shape L", np.shape(L))
    # print(L)
    # b = np.zeros(n_k)
    # print("shape b", np.shape(b))
    # sol = solve(2*L/n_m, b)
    # print(sol)
    # print()


