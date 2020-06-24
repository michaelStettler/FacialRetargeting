import numpy as np

def e_match(a,s):

    r = 15 # steepness for correlation amplification

    F = a.shape[0]
    K = s .shape[0]

    # calculate displacements
    d = np.linalg.norm(s,axis=0)
    s_sorted = s[np.argsort(d)]

    # claculate correlation coefficients between blendshapes and actor
    c = np.zeros((K, F))
    for f in np.arange(F):
        for k in np.arange(K):
            c[k,f]=np.dot(a[f],s_sorted[k]) / (np.linalg.norm(a[f])*np.linalg.norm(s_sorted[k]))

    c_pos = np.maximum(c,0)

    # claculate correlation coefficient within blendshapes
    c_blend = np.zeros((K, K))
    for k in np.arange(k):
        for l in np.arange(k):
            c[k,l]=np.dot(s_sorted[k],s_sorted[l]) / (np.linalg.norm(s_sorted[k])*np.linalg.norm(s_sorted[l]))

    # calculate trust values t_k
    temp = np.sum(np.maximum(c_blend,0),axis=1)
    t_k = 1- temp/np.amax(temp)

    # calculate correlation weights b
    b = np.exp(r*c_pos)/(np.exp(r/2)+np.exp(r*c_pos))

    c_tilde =  (1-np.tile(t_k,(1,F))).T*c_tilde + np.tile(t_k,(1,F)).T*b