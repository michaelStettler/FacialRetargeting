import numpy as np
from EMatch import EMatch
from EMesh import EMesh
from ECEG import ECEG

class EAlign:

    def __init__(self, tilda_ckf, uk, delta_af, delta_gk, delta_sk, alpha=0.01, beta=0.1):
        self.tilda_ckf = tilda_ckf
        self.uk = uk
        self.delta_af = delta_af
        self.delta_gk = delta_gk
        self.delta_sk = delta_sk

        self.e_match = EMatch(self.tilda_ckf, self.uk, self.delta_af)
        self.e_mesh = EMesh(self.delta_gk)
        self.e_ceg = ECEG(self.delta_sk)


if __name__ == '__main__':
    """
    test E_Align Class

    run: python -m EAlign
    """
    np.random.seed(0)
    np.set_printoptions(precision=4, linewidth=200)
    # declare variables
    n_k = 2
    n_f = 3
    n_m = 4
    n_n = n_m * 3  # = 4 markers


    tilda_ckf = np.random.rand(n_k, n_f)  # (k, f)
    uk = np.random.rand(n_k, n_n)
    delta_af = np.random.rand(n_f, n_n)
    delta_gk = np.random.rand(n_k, n_m, 3)
    delta_sk = np.random.rand(n_k, n_n)
    dp = np.random.rand(n_k, n_n)

    e_align = EAlign(tilda_ckf, uk, delta_af, delta_gk, delta_sk)