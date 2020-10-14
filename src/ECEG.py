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

        self.L = self.get_graph_laplacian(self.delta_sk)

    def get_graph_laplacian(self, dsk):
        """
        Get the Laplacian linear decomposition af formula 12 L_op(dsk)


        k:= num_of_blendshapes
        n:= num_of_features

        :param dsk: (k, n)
        :return: L (k, n)
        """
        # compute coefficients
        ckl = compute_corr_coef(dsk, dsk)

        # compute laplacians
        L = []
        for k in range(np.shape(dsk)[0]):
            sum_coeff = 0
            sum_norm = 0
            for l in range(np.shape(dsk)[0]):
                if l != k:
                    sum_coeff += ckl[k, l] * (dsk[l] - dsk[k])
                    sum_norm += np.abs(ckl[k, l])
            L.append(sum_coeff/sum_norm)

        return np.array(L)

    def _eceg(self, dp):
        """
        Compute E_CEG as in formula 13

        :param dp: delta p (k, n)
        :return: e_mesh
        """
        if len(np.shape(dp)) < 2:
            dp = np.reshape(dp, (self.K, self.N))

        # compute w = Laplacian * diff
        w = self.L * (dp - self.delta_sk)

        # compute norm
        norm = np.linalg.norm(w, axis=1) ** 2

        return np.sum(norm) / self.M

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

        # split L
        LX = self.L[:, x_indices]
        LY = self.L[:, y_indices]
        LZ = self.L[:, z_indices]

        # split Lsk
        LskX = self.delta_sk[:, x_indices]
        LskY = self.delta_sk[:, y_indices]
        LskZ = self.delta_sk[:, z_indices]

        # build A
        AX = (2/self.M) * np.diag(np.power(LX, 2).flatten())
        AY = (2/self.M) * np.diag(np.power(LY, 2).flatten())
        AZ = (2/self.M) * np.diag(np.power(LZ, 2).flatten())

        # build b
        bX = (2/self.M) * np.multiply(np.power(LX, 2), LskX).flatten()
        bY = (2/self.M) * np.multiply(np.power(LY, 2), LskY).flatten()
        bZ = (2/self.M) * np.multiply(np.power(LZ, 2), LskZ).flatten()

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
    print()

    # delcare ECEG
    e_CEG = ECEG(dsk)

    # compute Laplacian
    ckl = compute_corr_coef(dsk, dsk)
    L = []
    for k in range(np.shape(dsk)[0]):
        sum_coeff = 0
        sum_norm = 0
        for l in range(np.shape(dsk)[0]):
            if l != k:
                sum_coeff += ckl[k, l] * (dsk[l] - dsk[k])
                sum_norm += np.abs(ckl[k, l])
        L.append(sum_coeff/sum_norm)
    L = np.array(L)

    # compute eceg
    e_ceg = 0
    for k in range(n_k):
        ds = dsk[l] - dsk[k]
        w = L[k] * (dp[k] - dsk[k])

        norm = np.linalg.norm(w) ** 2
        e_ceg += norm

    eceg_ctrl = e_ceg / n_m
    print("eceg_ctrl = ", eceg_ctrl)

    # compute eCEG
    e_ceg_fn = e_CEG.get_eCEG()
    eceg = e_ceg_fn(dp)
    print("eceg =", eceg)

    assert np.around(eceg, 6) == np.around(eceg_ctrl, 6)
    print("emesh values are equal")
    print()

    # print("------------- test L@k decomposition --------------")
    # Lpk_test = e_CEG._compute_gl_operator(dp)
    # print("shape Lpk_test", np.shape(Lpk_test))
    # print(Lpk_test)
    # # build matrix L to form the L*dp
    # L = e_CEG.get_graph_laplacian(dp)
    # Lpk = L @ dp
    # print("shape Lpk", np.shape(Lpk))
    # print(Lpk)
    # assert Lpk.all() == Lpk_test.all()
    # print("L Decomposition works!")
    # print()

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

