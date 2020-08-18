import numpy as np


def re_order_delta(data):
    """
    re-order delta matrix data according to its norm
    the re-ordering goes from the bigger to smallest norm difference

    :param data:
    :return:
    """
    # compute total displacement
    d = np.linalg.norm(data, axis=1)

    # get sorted index from bigger to smaller
    sorted_index = np.flip(np.argsort(d))

    # sort data
    return data[sorted_index]


if __name__ == '__main__':
    """
    test re_order function
    
    run: python -m re_order_delta
    """
    np.random.seed(0)

    # test compute trust values
    delta_sk = np.random.rand(6, 1)  # (k, num_features)
    print("delta_sk", np.shape(delta_sk))

    delta_sk = re_order_delta(delta_sk)
    print("shape sorted delta_sk", np.shape(delta_sk))
    print(delta_sk)