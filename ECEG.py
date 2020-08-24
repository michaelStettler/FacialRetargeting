import numpy as np
from compute_corr_coef import compute_corr_coef

class ECEG:

    def __init__(self, delta_sk):
        self.delta_sk = delta_sk
        self.K = np.shape(delta_sk)[0]
        self.N = np.shape(delta_sk)[1]

    def _compute_graph_laplacian(self, dsk):
        # get ckl and remove diagonal
        ckl = compute_corr_coef(dsk, dsk)
        np.fill_diagonal(ckl, 0)

        # repeat dis_dsk
        ds = np.array([dsk - d for d in dsk])
        # compute weighted signed similarity
        w_sim = np.sum(np.multiply(np.repeat(np.expand_dims(ckl, axis=2), self.N, axis=2), ds), axis=1)
        # compute sum(|ckl|)
        norm_ckl = np.sum(np.abs(ckl), axis=1)
        norm_ckl = np.repeat(np.expand_dims(norm_ckl, axis=1), self.N, axis=1)
        # compute weighted signed graph
        L = w_sim / norm_ckl

        return L

    def _eceg(self, dp):
        """
        Compute E_CEG as in formula 13

        :param dp: delta p (k, n)
        :return: e_mesh
        """

        # compute graph Laplacian
        L = self._compute_graph_laplacian(dp - self.delta_sk)

        # compute norm
        norm = np.linalg.norm(L, axis=1) ** 2

        return np.sum(norm) / self.K

    def get_eceg(self):
        """
        return the function eceg
        :return:
        """
        return self._eceg


if __name__ == '__main__':

    np.random.seed(2)
    # declare variables
    n_k = 4  # num_blendshapes
    n_m = 1  # num markers
    n_n = n_m * 3  # num_features (num_markers * 3)
    dsk = np.random.rand(n_k, n_n)
    pk = np.random.rand(n_k, n_n)

    # compute displacement and similarity
    dis_dsk = pk - dsk
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

    eceg_ctrl = e_ceg / n_k  #  todo in the paper the normaization is done using M ?!?
    print("eceg_ctrl = ", eceg_ctrl)

    # compute eCEG
    e_ceg_fn = ECEG(dsk).get_eceg()
    eceg = e_ceg_fn(pk)
    print("eceg =", eceg)