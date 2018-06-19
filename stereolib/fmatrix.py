import math
import numpy as np

def naive_fmatrix(matches):
    A = np.matrix([[x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1] \
                   for ((x1, y1), (x2, y2)) in matches])
    U, S, VT = np.linalg.svd(A)
    f = VT[-1,:]
    F = np.reshape(f, (3, 3))
    
    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0

    return u * s * vt

def norm_eight_point(matches):
    # Computing centroids.
    n = len(matches)
    left = [np.matrix(list(p)) for p, q in matches]
    right = [np.matrix(list(q)) for p, q in matches]
    c1 = sum(left) / n
    c2 = sum(right) / n

    # Get average distance from points and centroids.
    avg1 = sum([np.linalg.norm(x - c1) for x in left]) / n
    avg2 =  sum([np.linalg.norm(x - c2) for x in right]) / n

    # Compute transform to points and merge left and right matches.
    sqrt2 = math.sqrt(2)
    T1 = np.matrix([[sqrt2 / avg1, 0, -c1[0, 0]], [0, sqrt2 / avg1, -c1[0, 1]], [0, 0, 1]])
    T2 = np.matrix([[sqrt2 / avg2, 0, -c2[0, 0]], [0, sqrt2 / avg2, -c2[0, 1]], [0, 0, 1]])

    norm_left = [T1 * np.matrix([[x[0, 0]], [x[0, 1]], [1]]) for x in left]
    norm_left[:] = [(x[0, 0] / x[2, 0], x[1, 0] / x[2, 0]) for x in norm_left]
    norm_right = [T2 * np.matrix([[x[0, 0]], [x[0, 1]], [1]]) for x in right]    
    norm_right[:] = [(x[0, 0] / x[2, 0], x[1, 0] / x[2, 0]) for x in norm_right]

    norm_matches = zip(norm_left, norm_right)

    # Get the fundamental matrix with "naive" method and denormalize its result.
    F = naive_fmatrix(norm_matches)
    return np.transpose(T2) * F * T1
