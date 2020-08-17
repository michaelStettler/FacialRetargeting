import numpy as np

"""""
David:  - SVD to normalize position with respect to plane
        - getHead Pos to check if enough head markers available and store head markers
"""""

def RigidTransform3D(A, B):

    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np. tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       # print("Reflection detected")
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)

    t = np.dot(-R,centroid_A.T) + centroid_B.T

    return R, t

def getHeadPos(head1, head2, head3, head4):
    if sum(x is None for x in [head1,head2,head3,head4]) > 1:
        raise Exception('Not enough Markers - cut File')
    pos = []
    heads = [head1,head2,head3,head4]
    for head in heads:
        if head != None and [head.x, head.y, head.z]!=[0,0,0]:
            pos.append([head.x, head.y, head.z])
        if len(pos)==3:
            return pos
    raise Exception('Not enough Markers - cut File')
    return
