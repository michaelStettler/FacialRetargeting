import numpy as np


def compute_delta(data, ref, norm_thresh=None):
    """
    compute the delta between a vector data and a ref vector

    norm_thresh allows to remove some outsiders. When markers doesn't exit, Nexus set the value to 0, therefore applying
    the norm_thresh will set this delta to 0

    :param data:
    :param ref:
    :param norm_thresh:
    :return:
    """
    deltas = []
    for d in data:
        delta = d - ref
        if norm_thresh is not None:
            delta[np.linalg.norm(delta, axis=1) > norm_thresh] = 0

        # check if delta is not filled by only zero -> != ref
        if np.any(delta):
            deltas.append(delta)

    return np.array(deltas)
