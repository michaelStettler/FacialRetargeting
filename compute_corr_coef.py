import numpy as np


def compute_corr_coef(da, ds):
    """
    Compute Pearson Correlation Coefficient between a and s
    This is formula 5 of the paper

    f:= number of frames
    k:= number of blendshapes

    :param da: delta-actor training sequence (f, n_features)
    :param ds:  delta-character sparse blendshapes (k, n_features)
    :return: correlation coefficients in delta representation
    """

    # compute norms
    a_norm = np.diag(np.power(np.linalg.norm(da, axis=1), -1))
    s_norm = np.diag(np.power(np.linalg.norm(ds, axis=1), -1))

    # compute Pearson Correlation Coefficients
    return a_norm @ da @ ds.T @ s_norm


def compute_tilda_corr_coef(ckf, tk, r=15):
    """
    compute similarity tilda_ckf from equation 8 in the paper

    k:= number of blendshapes
    f:= number of frames

    :param ckf: correlation matrix from compute_corr_coef (k, f)
    :param tk: trust values (k,)
    :param r: steepness
    :return:
    """
    # keep only positive components
    ckf = np.maximum(ckf, np.zeros(np.shape(ckf)))

    # amplify ckf (formula 7)
    b = np.exp(r*ckf) / (np.exp(r/2) + np.exp(r*ckf))

    # compute tilda_ckf
    return np.diag((1-tk)) @ ckf + np.diag(tk) @ b


if __name__ == '__main__':
    """
    Test compute_corr_coef function
    
    run: python -m compute_corr_coef
    """

    np.random.seed(0)
    print("------- test compute_corr_coef ----------")
    # test compute_corr_coef with 2 dims array
    a = np.random.rand(6, 3)  # (f, num_features)
    s = np.random.rand(4, 3)  # (k, num_features)
    print("f = num_frames, k = num_blendshapes")
    print("shape a", np.shape(a))
    print("shape s", np.shape(s))
    print()
    c = compute_corr_coef(a, s)
    print("shape c", np.shape(c))
    print(c)

    # built control
    c_control = np.zeros((np.shape(s)[0], np.shape(a)[0]))
    for f in range(np.shape(a)[0]):
        for k in range(np.shape(s)[0]):
            a_norm = np.linalg.norm(a[f])
            s_norm = np.linalg.norm(s[k])
            c_control[k, f] = s[k] @ a[f] / (a_norm * s_norm)

    assert (np.around(c, 6) == np.around(c_control, 6)).all()
    print("Compute_corr_coeff works!")
    print()

    print("------- test compute_tilda_corr_coef ----------")
    # test compute_tilda_corr_coef
    tk = np.random.rand(4)
    print("trust values tk:")
    print(tk)
    t_ckf = compute_tilda_corr_coef(c, tk)

    print("tilda_ckf", np.shape(t_ckf))
    print(t_ckf)
