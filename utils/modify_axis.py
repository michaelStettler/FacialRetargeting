import numpy as np


def modify_axis(pos, order='xzy2xyz', inverse_z=False):
    tmp_pos = np.copy(pos)

    # get the number of dimension of the tensor
    num_dim = len(np.shape(tmp_pos))

    if order == 'xzy2xyz':
        if num_dim == 2:
            tmp_pos[:, 2] = pos[:, 1]
            tmp_pos[:, 1] = pos[:, 2]
            pos = np.copy(tmp_pos)
            if inverse_z:
                pos[:, 2] = -pos[:, 2]
        elif num_dim == 3:
            tmp_pos[:, :, 2] = pos[:, :, 1]
            tmp_pos[:, :, 1] = pos[:, :, 2]
            pos = np.copy(tmp_pos)
            if inverse_z:
                pos[:, :, 2] = -pos[:, :, 2]
        else:
            raise ValueError("[modify_axis] Num dim {} is not implemented yet!".format(num_dim))
    else:
        raise ValueError("[modify_axis] Order {} is not implemented yet!".format(order))

    return pos