import numpy as np


def normalize_positions(pos, min_pos=None, max_pos=None, return_min=False, return_max=False):
    if min_pos is None:
        min_pos = np.repeat(np.expand_dims(np.amin(pos, axis=0), axis=0), np.shape(pos)[0], axis=0)

    pos -= min_pos

    if max_pos is None:
        # max_sk = np.repeat(np.expand_dims(np.amax(ref_sk, axis=0), axis=0), np.shape(ref_sk)[0], axis=0)
        max_pos = np.amax(pos)  # normalize only but the max to keep ratio # todo check if any difference?

    pos /= max_pos

    if return_min and return_max:
        return pos, min_pos, max_pos
    elif return_min:
        return pos, min_pos
    elif return_max:
        return pos, max_pos
    else:
        return pos
