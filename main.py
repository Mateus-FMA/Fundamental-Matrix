import numpy as np
import cv2
import os

from stereolib import stereomatch as sm
from stereolib import fmatrix as fm

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "\\data\\"
    test_dirs = ["Adirondack"]

    for test_dir in test_dirs:
        img1 = cv2.imread(data_dir + test_dir + "\\im0.png", cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(data_dir + test_dir + "\\im1.png", cv2.IMREAD_GRAYSCALE)

        matches = sm.get_matches(img1, img2)
        F1 = fm.naive_fmatrix(matches)
        F2 = fm.norm_eight_point(matches)

        print(F1)
        print(F2)
