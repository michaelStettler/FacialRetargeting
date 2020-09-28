import numpy as np

np.set_printoptions(precision=2, linewidth=200)


def rbf_kernel(k, k_prime):
    """
    compute the L2 norm between k and k_prime ||k - k_prime||
    using the fact that ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.T*y

    :param k: (n x m) vector
    :param k_prime: (n x m) vector
    :return: K (n x n) matrix
    """
    k_norm = np.sum(k ** 2, axis=-1)
    k_prime_norm = np.sum(k_prime ** 2, axis=-1)
    K = np.sqrt(np.abs((k_norm[:, None] + k_prime_norm[None, :] - 2 * np.dot(k, k.T))))
    return K


def rbf_warp(p, q):
    """
    RBF warping function to initialize the Actor Blendshape for the implementation of the paper:
    "Facial Retargeting with Automatic Range of Motion Alignment" (Ribera et al. 2017)

    The warping function follows the implementation from:
    "Transferring the Rig and Animations from a Character to Different Face Models" (Orvalho et al. 2008)
    by solving a linear function:

    ax = b
    with a = [K P; P.T 0] -> ((n+4)x(n+4))
    x = [W A].T
    b = [Q 0].T
    K is the RBF kernel U(x-p) = |x - p|

    :param p: n landmarks positions matrix (xyz) -> (nx3)
    :param q: n target positions matrix (xyz) -> (nx3)
    :return: W, A, solved matrix
    """

    # get number of lmks
    n = np.shape(p)[0]

    # declare matrices
    P = np.ones((n, 4))
    a_zero = np.zeros((4, 4))
    Q = q
    b_zero = np.zeros((4, 3))

    # build rbf kernel
    K = rbf_kernel(p, p)

    # build P
    P[:, 1:] = p

    # build final matrices
    a = np.concatenate((K, P), axis=1)
    a = np.concatenate((a, np.concatenate((P.T, a_zero), axis=1)), axis=0)
    b = np.concatenate((Q, b_zero), axis=0)

    # solve for ax = b with x = [W A].T
    x = np.linalg.solve(a, b)

    W = x[:n, :]
    A = x[n:, :]

    return W, A


def get_initial_actor_blendshapes(s0, a0, delta_sk):
    """
    Compute the initial guess actor blendshapes in delta space as explained in 4.4 Geometric Constraint of the paper

    k:= num_of_blendshapes
    m:= num_of_markers

    :param s0: neutral character expression
    :param a0: neutral actor expression
    :param delta_sk: character blendshapes in delta space
    :return: initial guess of actor blendshapes
    """
    # compute initial transform of neutral pose
    W, A = rbf_warp(s0, a0)
    # compute initial guess by transforming each character blendshapes delta_sk
    K = np.shape(delta_sk)[0]
    delta_gk = np.zeros(np.shape(delta_sk))
    for k in range(K):
        delta_gk[k] = np.multiply(delta_sk[k], W)

    return delta_gk


if __name__ == '__main__':
    """
    test the following two functions: 
        - rbf_warp
        - get_initial_actor_blendshapes
        
    run: python -m src.RBF_warp
    """
    # declare variables
    m = 5  # num_markers
    np.random.seed(0)
    print("---------- test RBF Warp ----------")
    # test RBF_warp function
    s0 = np.random.rand(m, 3)  # random landmarks population
    a0 = np.random.rand(m, 3)  # random target coordinates
    print("s0", np.shape(s0))
    print(s0)
    print("a0", np.shape(a0))
    print(a0)
    W, A = rbf_warp(s0, a0)
    print("shape W, A", np.shape(W), np.shape(A))
    print(W)
    print(A)
    print()

    print("---------- test RBF Warp ----------")
    # test get_initial_actor_blendshapes
    K = 4
    delta_sk = np.random.rand(K, m, 3)
    delta_gk = get_initial_actor_blendshapes(s0, a0, delta_sk)
    print("shape gk", np.shape(delta_gk))