import numpy as np


def get_soft_mask(dskm):
    """
    compute soft max vector uk
    || dskm || denotes the displacement x/y/z of marker m of blendshape k in delta space
    d: delta
    s: sparse character
    k: blendshapes
    m: markers

    k:= number of blendshapes
    m:= number of markers

    :param dskm: delta_skm (k, m, xyz)
    :return:
    """
    # compute norm
    norm_dskm = np.linalg.norm(dskm, axis=2)
    #get max norm
    max_norm = np.repeat(np.expand_dims(np.amax(norm_dskm, axis=1), axis=1), np.shape(dskm)[1], axis=1)
    # compute soft max
    return np.reshape(np.repeat(norm_dskm / max_norm, 3), (np.shape(dskm)[0], np.shape(dskm)[1]*np.shape(dskm)[2]))


if __name__ == '__main__':
    """
    test get_soft_max function
    
    run: python -m src.get_soft_max
    """
    np.random.seed(0)
    # declare variables
    n_k = 4
    n_m = 2
    dsk = np.random.rand(n_k, n_m, 3)  # (k, m, xyz)

    # build ukm control using double loops
    ukm_control = np.zeros((n_k, n_m, 3))
    for k in range(n_k):
        # compute max norm
        max_norm = 0
        for m in range(n_m):
            norm_dskm = np.linalg.norm(dsk[k, m])
            if norm_dskm > max_norm:
                max_norm = norm_dskm

        # compute ukm
        for m in range(n_m):
            norm_dskm = np.linalg.norm(dsk[k, m])
            ukm_control[k, m, 0] = norm_dskm / max_norm
            ukm_control[k, m, 1] = norm_dskm / max_norm
            ukm_control[k, m, 2] = norm_dskm / max_norm
    ukm_control = np.reshape(ukm_control, (n_k, n_m*3))
    # test compute_corr_coef with 2 dims array
    ukm = get_soft_mask(dsk)

    print("ukm", np.shape(ukm))
    print(ukm)
    print("ukm_control", np.shape(ukm_control))
    print(ukm_control)

    assert (np.around(ukm, 6).all() == np.around(ukm_control, 6).all())
    print("get_soft_max function works!")