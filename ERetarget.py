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

    def set_af(self, af):
        self.af = af

    def _e_fit(self, w):

        w = np.repeat(np.expand_dims(w, axis=1), self.N, axis=1)
        w_comb = np.multiply(w, self.delta_p)
        fits = self.af - np.sum(w_comb, axis=0)

        return np.linalg.norm(fits)**2/self.M

    def _e_sparse(self, w):
        return np.linalg.norm(w, ord=1) / self.K

    def _e_prior(self, w):
        return np.norm(self.v_L.dot(self.dV @ w))**2 / self.N

    def _e_retarget(self, w):
        return self._e_fit(w) + self.mu * self._e_prior(w) + self.nu * self._e_sparse(w)

    def get_EFit(self):
        return self._e_fit

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


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    # declare variables
    n_k = 3  # num_blendshapes
    n_f = 1  # num_frames
    n_m = 2  # num_vertices
    n_n = n_m * 3  # num_features

    af = np.random.rand(n_n)  # one single frame!
    dpk = np.random.rand(n_k, n_n)
    w = np.random.rand(n_k)  # only a single weights per blendshapes!
    v = np.random.rand(n_k, n_n)
    v0 = v[0]
    print("shape af", np.shape(af))
    print("shape dpk", np.shape(dpk))
    print("shape w", np.shape(w))
    print("shape v", np.shape(v))
    print()

    # ------------------------------------------------------------------------------
    # ---------------------------      E Fit         -------------------------------
    # ------------------------------------------------------------------------------
    print("-------- EFit ---------")

    # declare e_retarget
    e_retarg = ERetarget(dpk, v, v0)

    # test eFit
    fits = []
    for k in range(n_k):
        fit = w[k] * dpk[k]
        fits.append(fit)
    fits = af - np.sum(fits, axis=0)
    e_fit_test = np.linalg.norm(fits)**2/n_m
    print("e_fit_test", e_fit_test)

    e_retarg.set_af(af)
    e_fit = e_retarg._e_fit(w)
    print("e_fit", e_fit)
    assert e_fit == e_fit_test
    print("E fit values are equal")
    print()

    print("----- Minimization ------")
    import time as time
    print("try optimizer")
    from scipy import optimize
    start = time.time()
    opt = optimize.minimize(e_retarg.get_EFit(), w, method="BFGS")
    print("solved in:", time.time() - start)
    print(opt.x)

    print("try solver")
    from scipy.linalg import solve
    A, b = e_retarg.get_dEFit()
    start = time.time()
    sol = solve(A, b)
    print("solved in:", time.time() - start)
    print("Sol")
    print(sol)

    # test if values matches
    np.testing.assert_array_equal(np.around(opt.x, 5), np.round(sol, 5))
    print("[EFit] Reaches same values!")
    print()

    # ------------------------------------------------------------------------------
    # ---------------------------      E Fit         -------------------------------
    # ------------------------------------------------------------------------------
    print("-------- EPrior ---------")