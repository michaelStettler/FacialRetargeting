import numpy as np


def compute_delta(data, ref):
    """
    compute the delta between a vector data and a ref vector

    :param data:
    :param ref:
    :return:
    """
    deltas = []
    for d in data:
        delta = d - ref
        # check if delta is not filled by only zero -> != ref
        if np.any(delta):
            deltas.append(d - ref)

    return np.array(deltas)
