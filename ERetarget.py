import numpy as np

from mesh import triangulate_vertices
from mesh import build_Laplacian


class ERetarget():
    def __init__(self, dp, v, v0, mu=0.3, nu=0.6):
        self.mu = mu
        self.nu = nu
        print("shape v", np.shape(v))
        self.K = np.shape(dp)[0]
        self.N = np.shape(dp)[1]
        print("self.K", self.K)
        print("self.N", self.N)
        self.V = int(self.N / 3)
        print("self.V", self.V)

        self.af = None
        self.delta_p = dp
        # self.dV = v - np.repeat(np.expand_dims(v0, axis=1), self.K, axis=1).T
        # self.v_mesh = triangulate_vertices(v0)
        # self.v_L = build_Laplacian(self.v_mesh, self.V)  # todo expand matrix by 3!

    def set_af(self, af):
        self.af = af

    def _e_fit(self, w):
        w_comb = np.multiply(w, self.delta_p)
        fits = self.af - np.sum(w_comb, axis=0)

        return np.linalg.norm(fits)**2/self.V

    def _e_sparse(self, w):
        return np.linalg.norm(w, ord=1) / self.K

    def _e_prior(self, w):
        return np.norm(self.v_L.dot(self.dV @ w))**2 / self.N

    def _e_retarget(self, w):
        return self._e_fit(w) + self.mu * self._e_prior(w) + self.nu * self._e_sparse(w)

    def get_eRetarget(self):
        return self._e_retarget


if __name__ == '__main__':
    np.random.seed(0)
    # declare variables
    n_k = 3  # num_blendshapes
    n_f = 1  # num_frames
    n_v = 2  # num_vertices
    n_n = n_v * 3  # num_features

    af = np.random.rand(n_n)  # one single frame!
    dpk = np.random.rand(n_k, n_n)
    w = np.random.rand(n_k, n_n)
    v = np.random.rand(n_k, n_n)
    v0 = v[0]
    print("shape af", np.shape(af))
    print("shape dpk", np.shape(dpk))
    print("shape w", np.shape(w))
    print("shape v", np.shape(v))

    # declare e_reatrget
    e_retarg = ERetarget(af, dpk, v, v0)

    # test eFit
    fits = []
    for k in range(n_k):
        print("shape w[k]", np.shape(w[k]))
        print("shape dpk[k]", np.shape(dpk[k]))
        fit = w[k] * dpk[k]
        print("shape fit", np.shape(fit))
        print(fit)
        fits.append(fit)
    print("fits")
    print(np.array(fits))
    print("sum fits", np.sum(fits, axis=0))
    fits = af - np.sum(fits, axis=0)
    print("fits")
    print(fits)
    e_fit_test = np.linalg.norm(fits)**2/n_v
    print("e_fit_test", e_fit_test)

    e_retarg.set_af(af)
    e_fit = e_retarg._e_fit(w)
    print("e_fit", e_fit)