import numpy as np


def modify_axis(pos, order='xzy', inverse_z=False):
    tmp_pos = np.copy(pos)

    if order == 'xzy':
        tmp_pos[:, 2] = -pos[:, 1]
        tmp_pos[:, 1] = pos[:, 2]
        pos = tmp_pos
        if inverse_z:
            pos[:, 2] -= np.amin(pos[:, 2])
    else:
        raise ValueError("Order {} is not implemented yet!".format(order))

    return tmp_pos