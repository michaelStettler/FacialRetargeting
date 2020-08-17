import numpy as np

def compute_corr_coef(a, s):
    """
    Compute Pearson Correlation Coefficient between a and s
    This is formula 5 of the paper

    :param a:
    :param s:
    :return:
    """

    print("shape a", np.shape(a))
    print("shape s", np.shape(s))
    c = np.zeros((np.shape(a)[0], np.shape(s)[0]))
    print("shape c", np.shape(c))
    for f in range(np.shape(a)[0]):
        for k in range(np.shape(s)[0]):
            c[f, k] = a[f] @ s[k]
    return c

# todo test cases! if __name__ == '__main__':