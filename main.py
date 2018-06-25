import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import os

from stereolib import stereomatch as sm
from stereolib import fmatrix as fm
from stereolib import util

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "\\data\\temple\\"

    file_params = open(data_dir + "temple_par.txt")
    N = int(next(file_params))
    lines = [[float(i) for i in line.split()[1:]] for line in file_params]
    file_params.close()

    errors_naive = []
    errors_norm = []

    for i in range(N-1):
        img1 = cv2.imread(data_dir + "temple{:04}.png".format(i+1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(data_dir + "temple{:04}.png".format(i+2), cv2.IMREAD_GRAYSCALE)

        matches = sm.get_matches(img1, img2)
        N = min(len(matches), 20)

        try:
            F_naive = fm.naive_fmatrix(matches[:N])
            F_norm = fm.norm_eight_point(matches[:N])

            F_naive = F_naive / np.max(F_naive.flatten())
            F_norm = F_norm / np.max(F_norm.flatten())

            K1 = np.matrix([[lines[i][0], lines[i][1], lines[i][2]], \
                            [lines[i][3], lines[i][4], lines[i][5]], \
                            [lines[i][6], lines[i][7], lines[i][8]]])
            R1 = np.matrix([[lines[i][9], lines[i][10], lines[i][11]], \
                            [lines[i][12], lines[i][13], lines[i][14]], \
                            [lines[i][15], lines[i][16], lines[i][17]]])
            t1 = np.matrix([[lines[i][18]], \
                            [lines[i][19]], \
                            [lines[i][20]]])
            P1 = np.concatenate((R1, t1), 1)

            K2 = np.matrix([[lines[i+1][0], lines[i+1][1], lines[i+1][2]], \
                            [lines[i+1][3], lines[i+1][4], lines[i+1][5]], \
                            [lines[i+1][6], lines[i+1][7], lines[i+1][8]]])
            R2 = np.matrix([[lines[i+1][9], lines[i+1][10], lines[i+1][11]], \
                            [lines[i+1][12], lines[i+1][13], lines[i+1][14]], \
                            [lines[i+1][15], lines[i+1][16], lines[i+1][17]]])
            t2 = np.matrix([[lines[i+1][18]], \
                            [lines[i+1][19]], \
                            [lines[i+1][20]]])
            P2 = np.concatenate((R2, t2), 1)

            e2 = P2 * np.concatenate((t1, np.matrix([1])))
            e2_cross = util.get_cross_matrix(e2)
            F_gt = e2_cross * P2 * np.linalg.pinv(P1)
            F_gt = F_gt / np.max(F_gt.flatten())

            errors_naive.append(np.linalg.norm(F_naive - F_gt))
            errors_norm.append(np.linalg.norm(F_norm - F_gt))
        except fm.InconsistentMatchesException as e:
            continue

    e1 = [x for x in errors_naive if x < 10]
    e2 = [x for x in errors_norm if x < 10]

    plt.plot(range(len(errors_naive)), errors_naive, "b--", \
             range(len(errors_norm)), errors_naive, "r+")
    plt.show()

    plt.plot(range(len(e1)), e1, "b--", range(len(e2)), e2, "r+")
    plt.show()
