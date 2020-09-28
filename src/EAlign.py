import numpy as np
from scipy.linalg import solve

from src.EMatch import EMatch
from src.EMesh import EMesh
from src.ECEG import ECEG


class EAlign:
    """
    Construct a class to compute E_Align as in formula 4 using a function to pass directly the personalized blendshapes
    in delta space delta_p (dp)

    k:= num_of_blendshapes
    f:= num_frames
    m:= num_markers
    n:= num_features (n_m * 3)

    """

    def __init__(self, tilda_ckf, uk, delta_af, delta_gk, delta_sk, alpha=0.01, beta=0.1):
        self.tilda_ckf = tilda_ckf
        self.uk = uk
        self.delta_af = delta_af
        self.delta_gk = delta_gk
        self.delta_sk = delta_sk

        self.alpha = alpha
        self.beta = beta

        # declare energy functions
        self.e_match = EMatch(self.tilda_ckf, self.uk, self.delta_af)
        self.e_mesh = EMesh(self.delta_gk)
        self.e_ceg = ECEG(self.delta_sk)

    def _eAlign(self, dp):
        return self.e_match._ematch(dp) + self.alpha * self.e_mesh._emesh(dp) + self.beta * self.e_ceg._eceg(dp)

    def get_EAlign(self):
        """
        return E Align as a function
        :return:
        """

        print("[Warning] Using 'get_EAlign()' function for optimization may be very slow ")

        return self._eAlign

    def get_dEAlign(self):
        """
        Compute E Align as the linear combination of EMatch, EMesh and ECEG as in formula 4.
        The function return the equation system to solve for the personalized blendshapes delta_p (dp) as Ax + b
        The function splits the system into xyz coordinates

        :return:
        """
        AMaX, AMaY, AMaZ, bMaX, bMaY, bMaZ = self.e_match.get_dEmatch()
        AMeX, AMeY, AMeZ, bMeX, bMeY, bMeZ = self.e_mesh.get_dEmesh()
        ACEGX, ACEGY, ACEGZ, bCEGX, bCEGY, bCEGZ = self.e_ceg.get_dECEG()

        AX = AMaX + self.alpha * AMeX + self.beta * ACEGX
        AY = AMaY + self.alpha * AMeY + self.beta * ACEGY
        AZ = AMaZ + self.alpha * AMeZ + self.beta * ACEGZ

        bX = bMaX + self.alpha * bMeX + self.beta * bCEGX
        bY = bMaY + self.alpha * bMeY + self.beta * bCEGY
        bZ = bMaZ + self.alpha * bMeZ + self.beta * bCEGZ

        return AX, AY, AZ, bX, bY, bZ

    def compute_actor_specific_blendshapes(self):
        """
        Solve EAlign to compute the personalized actor-specific blendshapes in delta space (delta_p)
        The function solve the system Ax + b for each xyz coordinates and merge the results
        :return: delta_p (n_k*n_n, )
        """

        AX, AY, AZ, bX, bY, bZ = self.get_dEAlign()
        solX = solve(AX, bX)
        solY = solve(AY, bY)
        solZ = solve(AZ, bZ)
        sol = np.vstack((solX, solY, solZ)).reshape(-1, order='F')

        return sol


if __name__ == '__main__':
    """
    test E_Align Class
    
    1) test EAlign function (minimize)
    2) test EAlign equation (solve)

    run: python -m src.EAlign
    """
    np.random.seed(0)
    np.set_printoptions(precision=4, linewidth=200)

    # declare variables
    n_k = 12  # 2
    n_f = 15  # 3
    n_m = 16  # 4
    n_n = n_m * 3  # = 4 markers

    # declare random data
    tilda_ckf = np.random.rand(n_k, n_f)  # (k, f)
    uk = np.random.rand(n_k, n_n)
    delta_af = np.random.rand(n_f, n_n)
    delta_gk = np.random.rand(n_k, n_m, 3)
    delta_sk = np.random.rand(n_k, n_n)
    dp = np.random.rand(n_k, n_n)

    # declare ERetarget
    e_align = EAlign(tilda_ckf, uk, delta_af, delta_gk, delta_sk)
    e_align_fn = e_align.get_EAlign()

    print("----- Minimization ------")
    import time as time
    print("try optimizer")
    from scipy import optimize
    start = time.time()
    opt = optimize.minimize(e_align_fn, dp, method="BFGS")
    print("solved in:", time.time() - start)
    print(opt.x[:10])  # print only 10 first

    from scipy.linalg import solve
    print("try solver")
    start = time.time()
    sol = e_align.compute_actor_specific_blendshapes()
    print("solved in:", time.time() - start)
    print(sol[:10])  # print only 10 first
    print("shape personalized blendshapes", np.shape(sol))