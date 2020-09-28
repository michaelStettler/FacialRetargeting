import numpy as np
from src.compute_corr_coef import compute_corr_coef

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
        # test if delta_sk are separable into xyz
        if self.N % 3 != 0:
            raise ValueError("Number of features ({}) is not a multiple of 3 (xyz)".format(self.N))

        # split data into xyz coordinates
        x_indices = np.arange(start=0, stop=self.N, step=3)
        y_indices = np.arange(start=1, stop=self.N, step=3)
        z_indices = np.arange(start=2, stop=self.N, step=3)
        # split self.delta_sk
        skX = self.delta_sk[:, x_indices]
        skY = self.delta_sk[:, y_indices]
        skZ = self.delta_sk[:, z_indices]

        # compute Laplacian
        LskX = self._compute_gl_operator(skX)
        LskY = self._compute_gl_operator(skY)
        LskZ = self._compute_gl_operator(skZ)

        # build A
        AX = (2/self.M) * np.diag(np.power(LskX, 2).flatten())
        AY = (2/self.M) * np.diag(np.power(LskY, 2).flatten())
        AZ = (2/self.M) * np.diag(np.power(LskZ, 2).flatten())

        # build b
        bX = (2/self.M) * np.multiply(np.power(LskX, 2), skX).flatten()
        bY = (2/self.M) * np.multiply(np.power(LskY, 2), skY).flatten()
        bZ = (2/self.M) * np.multiply(np.power(LskZ, 2), skZ).flatten()

        return AX, AY, AZ, bX, bY, bZ


if __name__ == '__main__':
    """
    
    """

    np.random.seed(1)
    np.set_printoptions(precision=4, linewidth=200, suppress=True)

    # declare variables
    n_k = 4  # num_blendshapes
    n_m = 2  # num markers
    n_n = n_m * 3  # num_features (num_markers * 3)
    dsk = np.random.rand(n_k, n_n)
    dp = np.random.rand(n_k, n_n)
    print("shape dp", np.shape(dp))
    print(dp)

    # compute displacement and similarity
    dis_dsk = dp - np.reshape(dsk, (n_k, n_n))
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
    eceg = e_ceg_fn(dp)
    print("eceg =", eceg)

    assert np.around(eceg, 6) == np.around(eceg_ctrl, 6)
    print("emesh values are equal")
    print()

    print("------------- test L@k decomposition --------------")
    Lpk_test = e_CEG._compute_gl_operator(dp)
    print("shape Lpk_test", np.shape(Lpk_test))
    print(Lpk_test)
    # build matrix L to form the L*dp
    L = e_CEG.get_graph_laplacian(dp)
    Lpk = L @ dp
    print("shape Lpk", np.shape(Lpk))
    print(Lpk)
    assert Lpk.all() == Lpk_test.all()
    print("L Decomposition works!")
    print()

    print("----- Minimization ------")
    print("try minimize")
    import time as time
    print("try optimizer")
    from scipy import optimize
    start = time.time()
    opt = optimize.minimize(e_CEG.get_eCEG(), dsk, method="BFGS")
    print("solved in:", time.time() - start)
    print("shape opt.x", np.shape(opt.x))
    print(opt.x)

    print("try solve")
    from scipy.linalg import solve
    AX, AY, AZ, bX, bY, bZ = e_CEG.get_dECEG()
    start = time.time()
    solX = solve(AX, bX)
    solY = solve(AY, bY)
    solZ = solve(AZ, bZ)
    sol = np.vstack((solX, solY, solZ)).reshape(-1, order='F')
    print("solved in:", time.time() - start)
    print("shape sol", np.shape(sol))
    print(sol)
    print("dsk")
    print(dsk.flatten())

