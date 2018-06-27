import numpy as np
import cv2

def get_matches(img1, img2):
    # Initiate detector.
    akaze = cv2.AKAZE_create()

    # Find keypoint and descriptors.
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # Initialize matcher.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Perform matching using the ratio test described in the SIFT original article (even though
    # we're using A-KAZE descriptors):
    # Distinctive Image Features from Scale-Invariant Keypoints (David G. Lowe, 2004).
    # Also, we discard the matches with too different coordinates. Our threshold is t = 10
    # pixels.
    matches = bf.knnMatch(des1, des2, k = 2)
    # good = [m for (m, n) in matches if m.distance < 0.8 * n.distance]
    t = 10
    good = [m for (m, n) in matches if m.distance < 0.8 * n.distance and \
            abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]) <= t and \
            abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) <= t]
    # bla = cv2.drawMatches(img1, kp1, img2, kp2, good[:50], None)
    # cv2.imshow('Bla', bla)
    # cv2.waitKey()

    return [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in good]