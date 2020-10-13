import numpy as np
from tqdm import tqdm


def align_to_markers(pos, ref_idx, roll_ref):
    # correct rotation
    # used this to get the roll, pitch and yaw angle, and then implement the rotation matrix
    # https://www.mathworks.com/matlabcentral/answers/298940-how-to-calculate-roll-pitch-and-yaw-from-xyz-coordinates-of-3-planar-points
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

    # 1st get 3 points  (use only markers H1, H2, and H3)
    p1 = pos[ref_idx[0], :]
    p2 = pos[ref_idx[1], :]
    p3 = pos[ref_idx[2], :]

    # build two vectors in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # compute the normal vector to the plane
    Z = np.cross(v1, v2)
    Z = Z/np.linalg.norm(Z)
    # create second and third vector
    X = (p1 + p2)/2 - p3
    X = X/np.linalg.norm(X)
    Y = np.cross(Z, X)

    # get roll, pitch and yaw angle
    roll = np.arctan2(-Z[1], Z[2])
    pitch = np.arcsin(Z[0])
    # yaw = np.arctan2(-Y[0], X[0])  # as given in the link, but I suppose I have some other "order" issue here
    yaw = np.arctan2(-Y[0], Z[0])

    # compute corrections
    dPitch = -pitch
    dYaw = (yaw - np.pi/2)
    dRoll = -(np.pi - roll) + roll_ref * np.pi/180
    # print("dPitch dYaw dRoll", dPitch, dYaw, dRoll)
    # print("dPitch dYaw dRoll", dPitch*180/np.pi, dYaw*180/np.pi, dRoll*180/np.pi)

    # compute rotation matrix
    R_pitch = np.array([[np.cos(dPitch), 0, np.sin(dPitch)],
                  [0, 1, 0],
                  [-np.sin(dPitch), 0, np.cos(dPitch)]])
    R_yaw = np.array([[np.cos(dYaw), -np.sin(dYaw), 0],
                  [np.sin(dYaw), np.cos(dYaw),  0],
                  [0, 0, 1]])
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(dRoll), -np.sin(dRoll)],
                       [0, np.sin(dRoll), np.cos(dRoll)]])
    R = R_pitch @ R_yaw @ R_roll
    # rotate positions
    pos = pos @ R

    # correct space positions
    mean_pos = np.mean(pos[[ref_idx[0], ref_idx[1]], :], axis=0)
    pos -= mean_pos
    # pos -= pos[ref_idx[0], :]

    return pos


def align_to_head_markers(positions, ref_idx, roll_ref=25):
    """
    Align the position to 3 markers (ref_idx). The function correct the rotation and position, as to fix the center
    of the 3 markers to zero and having 0 angles.

    roll_ref allows to set a desired angle

    :param pos: positions
    :param ref_idx: (3,)
    :param roll_ref: int definying what angle we want the roll to be
    :return:
    """
    num_dim = len(np.shape(positions))

    if num_dim == 2:
        positions = align_to_markers(positions, ref_idx, roll_ref)
    elif num_dim == 3:
        aligned_positions = []
        num_pos = np.shape(positions)[0]
        for p in tqdm(range(num_pos)):
            pos = positions[p]
            ap = align_to_markers(pos, ref_idx, roll_ref)
            aligned_positions.append(ap)
        positions = np.array(aligned_positions)

    else:
        raise ValueError("[Align Head Markers] Dimension {} not implemented yet".format(num_dim))

    return positions