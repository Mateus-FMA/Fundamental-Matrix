import math
import numpy as np

class InconsistentMatchesException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def eight_points(matches):    
    A = np.matrix([[x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1] \
                   for ((x1, y1), (x2, y2)) in matches])

    if A.shape[0] < 8:
        raise InconsistentMatchesException("Insufficient amount of matches")

    U, S, VT = np.linalg.svd(A)
    f = VT[-1,:]
    F = np.matrix([[f[0,0], f[0,1], f[0,2]], \
                   [f[0,3], f[0,4], f[0,5]], \
                   [f[0,6], f[0,7], f[0,8]]])
    
    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0

    return u * s * vt

def norm_eight_points(matches):
    # Computing centroids.
    n = len(matches)
    left = [np.matrix(list(p)) for p, q in matches]
    right = [np.matrix(list(q)) for p, q in matches]
    c1 = sum(left) / n
    c2 = sum(right) / n

    # Get average distance from points and centroids.
    avg1 = sum([np.linalg.norm(x - c1) for x in left]) / n
    avg2 =  sum([np.linalg.norm(x - c2) for x in right]) / n

    if avg1 < 1e-6 or avg2 < 1e-6:
        raise InconsistentMatchesException("Matching points are too close (degenerate to 1 point match).")

    # Compute transform to points and merge left and right matches.
    s1 = math.sqrt(2) / avg1
    s2 = math.sqrt(2) / avg2
    T1 = np.matrix([[s1, 0, -s1 * c1[0, 0]], [0, s1, -s1 * c1[0, 1]], [0, 0, 1]])
    T2 = np.matrix([[s2, 0, -s2 * c2[0, 0]], [0, s2, -s2 * c2[0, 1]], [0, 0, 1]])

    norm_left = [T1 * np.matrix([[x[0, 0]], [x[0, 1]], [1]]) for x in left]
    norm_left[:] = [(x[0, 0] / x[2, 0], x[1, 0] / x[2, 0]) for x in norm_left]
    norm_right = [T2 * np.matrix([[x[0, 0]], [x[0, 1]], [1]]) for x in right]    
    norm_right[:] = [(x[0, 0] / x[2, 0], x[1, 0] / x[2, 0]) for x in norm_right]

    norm_matches = zip(norm_left, norm_right)

    # Get the fundamental matrix with "naive" method and denormalize its result.
    F = eight_points(norm_matches)
    return np.transpose(T2) * F * T1
