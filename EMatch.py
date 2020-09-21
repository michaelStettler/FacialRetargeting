import numpy as np


class EMatch:
    """
    Construct a class to compute E_Match as in formula 10 using a function to pass directly the personalized blendshapes
    in delta space delta_p (dp)

    k:= num_of_blendshapes
    f:= num_frames
    n:= num_features

    """
    def __init__(self, tckf, uk, daf):
        self.tilda_ckf = tckf
        self.uk = uk
        self.delta_af = daf
        self.F = np.shape(self.delta_af)[0]
        self.K = np.shape(self.uk)[0]
        self.N = np.shape(self.uk)[1]

    def _ematch(self, dp):
        """
        Compute E_Match as in formula 10

        :param dp: delta p (k, n)
        :return: e_match
        """
        # reshape dp in case it comes as a 1D array
        if len(np.shape(dp)) < 2:
            dp = np.reshape(dp, (self.K, self.N))


        # diagonalize uk
        diag_uk = np.array([np.diag(uk) for uk in self.uk])  # using diag(self.uk) would result of getting only the diagonal elements
        # compute weighted mask
        w_mask = diag_uk @ self.delta_af.T
        # duplicate dp
        dup_dp = np.repeat(np.expand_dims(dp, axis=2), self.F, axis=2)
        # compute norm
        norm = np.power(np.linalg.norm(dup_dp - w_mask, axis=1), 2)
        # compute e_match
        return np.sum(np.multiply(self.tilda_ckf, norm)) / self.F

    def get_eMatch(self):
        """
        return ematch as a function
        :return:
        """
        print("[Warning] Using this function for optimization may be very slow ")
        return self._ematch

    def get_dEmatch(self):
        """
        Compute the derivative of E_Match (formula 10) at delta_p as to minimize delta_p -> E_match' = 0
        equation: (2/F) * sum_f(c_{k,f}) * delta_p_k - (2/F) * sum_f[(c_{k,f}) * diag(u_k) * delta_a_f]

        It splits the equation in a diagonal matrix A and a vector b as to solve the equation Ax = b, with x = delta_p
        Since the equation are separable in xyz, the function splits the data and returns a system of equation for each
        dimension, resulting in 3*(kMxknM) instead of one (3kMx3kM) -> section 4.6 of the paper

        M:= num_markers = self.N / 3
        A*:= (kM x kM) diag matrix with coef = (2/F) * sum_f(c_{k,f})
        b*:= (kn,) vector with value =(2/F) * sum_f[(c_{k,f}) * diag(u_k) * delta_a_f]

        :return: AX, AY, AZ, bX, bY, bZ
        """
        # test if data are separable into xyz
        if self.N % 3 != 0:
            raise ValueError("Number of features ({}) is not a multiple of 3 (xyz)".format(self.N))
        M = int(self.N / 3)  # num markers

        # split data into xyz coordinates
        x_indices = np.arange(start=0, stop=self.N, step=3)
        y_indices = np.arange(start=1, stop=self.N, step=3)
        z_indices = np.arange(start=2, stop=self.N, step=3)
        # split self.uk
        ukX = self.uk[:, x_indices]
        ukY = self.uk[:, y_indices]
        ukZ = self.uk[:, z_indices]
        # split self.delta_af
        afX = self.delta_af[:, x_indices]
        afY = self.delta_af[:, y_indices]
        afZ = self.delta_af[:, z_indices]

        # declare variables
        bX = np.zeros((self.K, M))
        bY = np.zeros((self.K, M))
        bZ = np.zeros((self.K, M))

        # build A (kn x kn) diagonal matrix
        A = (2/self.F) * np.diag(np.repeat(np.sum(self.tilda_ckf, axis=1), M))

        # there's probably an even better way to make it all in a matrix form :)
        for k in range(self.K):
            # compute the term: tilda_c[k,:] * diag(u[k]) * delta_af[:]
            bX[k] = (2 / self.F) * self.tilda_ckf[k] @ (np.diag(ukX[k]) @ afX.T).T
            bY[k] = (2 / self.F) * self.tilda_ckf[k] @ (np.diag(ukY[k]) @ afY.T).T
            bZ[k] = (2 / self.F) * self.tilda_ckf[k] @ (np.diag(ukZ[k]) @ afZ.T).T
        bX = bX.reshape(-1)
        bY = bY.reshape(-1)
        bZ = bZ.reshape(-1)

        # A = Ax = Ay = Az
        return A, A, A, bX, bY, bZ


if __name__ == '__main__':
    """
    test E_Match function 
    
    1) test that E_Match is computer correctly
    2) test optimization of the E_Match function 
    
    run: python -m EMatch
    """
    np.random.seed(0)
    np.set_printoptions(precision=4, linewidth=200)
    # declare variables
    n_k = 2
    n_f = 3
    n_n = 12  # = 4 markers
    tckf = np.random.rand(n_k, n_f)  # (k, f)
    uk = np.random.rand(n_k, n_n)
    da = np.random.rand(n_f, n_n)
    dp = np.random.rand(n_k, n_n)

    print("----- EMatch Function -----")
    # control compute e_match
    ematch_ctrl = 0
    for f in range(n_f):
        for k in range(n_k):
            norm = np.linalg.norm(dp[k] - np.diag(uk[k]) @ da[f])
            ematch_ctrl += tckf[k, f] * norm**2
    ematch_ctrl /= n_f
    print("ematch_ctrl")
    print(ematch_ctrl)

    # compute e_match
    e_match_fn = EMatch(tckf, uk, da).get_eMatch()
    ematch = e_match_fn(dp)
    print("ematch")
    print(ematch)

    # test if value matches (up to 6 decimals)
    assert np.around(ematch, 6) == np.around(ematch_ctrl, 6)
    print("ematch values are equal")
    print()

    print("----- Minimization ------")
    import time as time
    print("try optimizer")
    from scipy import optimize
    start = time.time()
    opt = optimize.minimize(e_match_fn, dp, method="CG")
    print("solved in:", time.time() - start)
    print(opt.x[:10])  # print only 10 first

    from scipy.linalg import solve
    print("try solver")
    AX, AY, AZ, bX, bY, bZ = EMatch(tckf, uk, da).get_dEmatch()
    start = time.time()
    solX = solve(AX, bX)
    solY = solve(AY, bY)
    solZ = solve(AZ, bZ)
    sol = np.vstack((solX, solY, solZ)).reshape(-1, order='F')
    print("solved in:", time.time() - start)
    print(sol[:10])  # print only 10 first

    # test if values matches
    assert opt.x.all() == sol.all()
    print("Reached same value!")
