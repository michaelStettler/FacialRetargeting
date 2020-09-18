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
        return the function ematch
        :return:
        """
        return self._ematch


if __name__ == '__main__':
    np.random.seed(0)
    # declare variables
    n_k = 2
    n_f = 3
    n_n = 4
    tckf = np.random.rand(n_k, n_f)  # (k, f)
    uk = np.random.rand(n_k, n_n)
    da = np.random.rand(n_f, n_n)
    dp = np.random.rand(n_k, n_n)

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

    assert ematch == ematch_ctrl
    print("ematch values are equal")
    print()

    print("----- Minimization ------")
    print("try optimizer")
    print("shape dp", np.shape(dp))
    from scipy import optimize
    opt = optimize.minimize(e_match_fn, dp, method="CG")
    print(opt)

    # print("try linprog")
    # from scipy.optimize import linprog
    # print("Using slack variables")
    # # c = [8, 10, 0, 0, 0]
    # # A = np.array([[4, 15, -1, 0, 0],
    # #               [6, 4, 0, -1, 0],
    # #               [5, 2, 0, 0, -1]])
    # # b = [70, 50, 30]
    # # sol = linprog(c, A_eq=A, b_eq=b)
    # print("Using upper bound (<=)")
    # c = [8, 10]
    # A = np.array([[-4, -15],
    #               [-6, -4],
    #               [-5, -2]])
    #
    # b = [-70, -50, -30]
    # sol = linprog(c, A_ub=A, b_ub=b)
    # print(sol)

    b = []
    for f in range(n_f):
        for k in range(n_k):
            b.append(np.diag(uk[k]) @ da[f])
    # b = np.sum(b, axis=1)
    print("shape b", np.shape(b))
    a = np.diag(dp.flatten())
    print("shape a", np.shape(a))
    sol = solve(a, b)
    print(sol)
    print()
