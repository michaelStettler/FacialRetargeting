import numpy as np


class EMatch:
    def __init__(self, tckf, uk, daf):
        self.tilda_ckf = tckf
        self.uk = uk
        self.delta_af = daf
        self.F = np.shape(self.delta_af)[0]

    def _ematch(self, dp):
        print("shape uk", np.shape(self.uk))
        print("shape daf", np.shape(self.delta_af))
        print(np.diag(self.uk))
        for uk in self.uk:
            print("uk")
            print(uk)
        diag_uk = [np.diag(uk) for uk in self.uk]
        print(np.shape(diag_uk))
        test = self.uk @ self.delta_af
        print(test)
        return 0

    def get_ematch(self):
        return self._ematch


if __name__ == '__main__':
    np.random.seed(0)
    # declare variables
    n_k = 2
    n_f = 3
    n_n = 4
    tckf = np.random.rand(n_k, n_f)  # (k, f)
    uk = np.random.rand(n_k, n_n)
    daf = np.random.rand(n_f, n_n)
    dp = np.random.rand(n_k, n_n)

    # control compute e_match
    ematch_ctrl = 0
    for f in range(n_f):
        for k in range(n_k):
            print(uk[k])
            # print(np.diag(uk[k]))
            # print(np.diag(uk[k]) @ daf[f])
            norm = np.linalg.norm(dp[k] - np.diag(uk[k]) @ daf[f])
            ematch_ctrl += tckf[k, f] * norm**2
            # print()

    ematch_ctrl /= n_f
    print("ematch_ctrl")
    print(ematch_ctrl)

    # compute e_match
    e_match_fn = EMatch(tckf, uk, daf).get_ematch()
    ematch = e_match_fn(dp)
    print(ematch)
