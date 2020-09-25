import numpy as np

from mesh import triangulate_vertices
from mesh import build_Laplacian


class ERetarget():
    def __init__(self, dp, v, v0, mu=0.3, nu=0.6):
        self.mu = mu
        self.nu = nu
        self.K = np.shape(dp)[0]
        self.N = np.shape(dp)[1]
        self.M = int(self.N / 3)

        self.af = None
        self.delta_p = dp
        self.v = v
        self.v0 = v0

        # compute matrix product L_deltaV
        self.LdV = self._build_L_deltaV(v, v0)

    def set_af(self, af):
        self.af = af

    def _e_fit(self, w):

        w = np.repeat(np.expand_dims(w, axis=1), self.N, axis=1)
        w_comb = np.multiply(w, self.delta_p)
        fits = self.af - np.sum(w_comb, axis=0)

        return np.linalg.norm(fits)**2/self.M

    def _e_sparse(self, w):
        return np.linalg.norm(w, ord=1) / self.K

    def _build_L_deltaV(self, v, v0):
        LdV = []
        for k in range(self.K):
            # compute delta_V
            if np.array_equal(v[k], v0):
                # avoid the case of v[k] = v0
                dV = v[k]
                print("v[k] = v0")
            else:
                dV = v[k] - v0
            # build mesh
            mesh = triangulate_vertices(dV)
            # build Laplacian
            L = build_Laplacian(mesh, n_v)
            L = L.todense()
            L_expand = np.zeros((n_n, n_n))
            # expand L by 3
            for i in range(n_n):
                for j in range(n_n):
                    if i % 3 == 0 and j % 3 == 0:
                        L_expand[i, j] = L[int(i / 3), int(j / 3)]
                    if i % 3 == 1 and j % 3 == 1:
                        L_expand[i, j] = L[int(i / 3), int(j / 3)]
                    if i % 3 == 2 and j % 3 == 2:
                        L_expand[i, j] = L[int(i / 3), int(j / 3)]
            LdV.append(L_expand @ dV)

        return np.array(LdV)

    def _e_prior(self, w):
        prior = np.multiply(self.LdV, np.repeat(np.expand_dims(w, axis=1), self.N, axis=1))
        return np.sum(np.linalg.norm(prior)**2) / self.N

    def _e_retarget(self, w):
        return self._e_fit(w) + self.mu * self._e_prior(w) + self.nu * self._e_sparse(w)

    def get_EFit(self):
        return self._e_fit

    def get_EPrior(self):
        return self._e_prior

    def get_eRetarget(self):
        return self._e_retarget

    def get_dEFit(self):
        """
        Compute the derivative of E_fit (formula 2) and split the equation to fit the form: Ax + b
        With A a square matrix of size (kxk) and b a vector of size (k,)

        k:= num blendshapes
        M := num_markers

        :return: A, b
        """
        A = (2/self.M) * self.delta_p @ self.delta_p.T
        b = (2/self.M) * self.delta_p @ self.af

        return A, b

    def get_dEPrior(self):
        """

        :return: A, b
        """
        A = (2/self.N) * np.diag(np.sum(np.power(self.LdV, 2), axis=1))
        b = np.zeros(self.K)

        return A, b

    def get_dESparse(self):
        """

        :return: A, b
        """
        A = (1/self.K) * np.eye(self.K)
        b = np.zeros(self.K)

        return A, b

    def get_dERetarget(self):
        AFit, bFit = e_retarg.get_dEFit()
        APrior, bPrior = e_retarg.get_dEPrior()
        ASparse, bSparse = e_retarg.get_dEPrior()

        A = AFit + self.mu * APrior + self.nu * ASparse
        b = bFit + self.mu * bPrior + self.nu * bSparse

        return A, b


if __name__ == '__main__':
    import time as time
    from scipy import optimize
    from scipy.linalg import solve

    np.random.seed(1)
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    # declare variables
    n_k = 30  # num_blendshapes
    n_f = 1  # num_frames
    n_v = 40  # num_vertices (min 4 to use Delaunay)
    n_n = n_v * 3  # num_features

    af = np.random.rand(n_n)  # one single frame!
    dpk = np.random.rand(n_k, n_n)
    w = np.random.rand(n_k)  # only a single weights per blendshapes!
    v = np.random.rand(n_k, n_n)
    neutral_pose_idx = 0
    v0 = v[neutral_pose_idx]
    print("shape af", np.shape(af))
    print("shape dpk", np.shape(dpk))
    print("shape w", np.shape(w))
    print("shape v", np.shape(v))
    print()

    # declare e_retarget
    e_retarg = ERetarget(dpk, v, v0)
    e_retarg.set_af(af)

    # ------------------------------------------------------------------------------
    # ---------------------------      E Fit         -------------------------------
    # ------------------------------------------------------------------------------
    print("-------- EFit ---------")
    # test eFit
    fits = []
    for k in range(n_k):
        fit = w[k] * dpk[k]
        fits.append(fit)
    fits = af - np.sum(fits, axis=0)
    e_fit_test = np.linalg.norm(fits)**2/n_v
    print("e_fit_test", e_fit_test)

    e_fit = e_retarg._e_fit(w)
    print("e_fit", e_fit)
    assert e_fit == e_fit_test
    print("[EFit] Error values are equal")
    print()

    print("----- Minimization ------")
    print("try optimizer")
    start = time.time()
    opt = optimize.minimize(e_retarg.get_EFit(), w, method="BFGS")
    print("solved in:", time.time() - start)
    print(opt.x)

    print("try solver")
    A, b = e_retarg.get_dEFit()
    start = time.time()
    sol = solve(A, b)
    print("solved in:", time.time() - start)
    print("Sol")
    print(sol)

    # test if values matches
    # np.testing.assert_array_equal(np.around(opt.x, 4), np.round(sol, 4))
    # print("[EFit] Optimization vs. Solver reaches same values!")
    print()

    # ------------------------------------------------------------------------------
    # ---------------------------      E Prior       -------------------------------
    # ------------------------------------------------------------------------------
    print("-------- EPrior ---------")

    # test ePrior
    priors = []
    for k in range(n_k):
        # compute delta_V
        if k == neutral_pose_idx:
            # avoid the case of v[neutral_pose_idx] - v0 = 0
            dV = v[k]
        else:
            dV = v[k] - v0
        # build mesh
        mesh = triangulate_vertices(dV)
        # build Laplacian
        L = build_Laplacian(mesh, n_v)
        L = L.todense()
        L_expand = np.zeros((n_n, n_n))
        # expand L by 3
        for i in range(n_n):
            for j in range(n_n):
                if i % 3 == 0 and j % 3 == 0:
                    L_expand[i, j] = L[int(i/3), int(j/3)]
                if i % 3 == 1 and j % 3 == 1:
                    L_expand[i, j] = L[int(i / 3), int(j / 3)]
                if i % 3 == 2 and j % 3 == 2:
                    L_expand[i, j] = L[int(i / 3), int(j / 3)]
        prior = np.linalg.norm(L_expand @ dV * w[k])**2
        priors.append(prior)
    ePrior_test = np.sum(priors) / n_n
    print("EPrior_test", ePrior_test)

    # compute e_prior
    e_prior_fn = e_retarg.get_EPrior()
    ePrior = e_prior_fn(w)
    print("ePrior")
    print(ePrior)

    assert round(ePrior_test, 5) == round(ePrior, 5)
    print("[EPrior] Error values are equal")
    print()

    print("----- Minimization ------")
    print("try optimizer")
    start = time.time()
    opt = optimize.minimize(e_retarg.get_EPrior(), w, method="BFGS")
    print("solved in:", time.time() - start)
    print(opt.x)

    print("try solver")
    A, b = e_retarg.get_dEPrior()
    start = time.time()
    sol = solve(A, b)
    print("solved in:", time.time() - start)
    print("shape sol", np.shape(sol))
    print(sol)

    # test if values matches
    # np.testing.assert_array_equal(np.around(opt.x, 5), np.round(sol, 5))
    # print("Reached same value!")

    # ------------------------------------------------------------------------------
    # ---------------------------      E Retarget    -------------------------------
    # ------------------------------------------------------------------------------
    print("-------- ERetarget ---------")

    print("test eRetarget")
    start = time.time()
    opt = optimize.minimize(e_retarg.get_eRetarget(), w, method="BFGS")
    print("solved in:", time.time() - start)
    print("shape opt.x", np.shape(opt.x))
    print(opt.x)

    print("try solver")
    A, b = e_retarg.get_dERetarget()
    start = time.time()
    sol = solve(A, b)
    print("solved in:", time.time() - start)
    print("shape sol", np.shape(sol))
    print(sol)

