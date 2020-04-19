import numpy as np
import cv2 as cv
import config

# calculate the essential matrix
def motion_estimation_E(pts_2d_src, pts_2d_dst):
    E_mat, inliners = cv.findEssentialMat(pts_2d_src, pts_2d_dst, config.P_MAT,
                                          method= cv.FM_RANSAC, prob= 0.8, threshold=1)
    return E_mat

# calculate the fundamental matrix
def motion_estimation_F(pts_2d_src, pts_2d_dst):
    F_mat, inliners = cv.findFundamentalMat(pts_2d_src, pts_2d_dst, method=cv.FM_RANSAC ,
                                            ransacReprojThreshold= 1, confidence=0.80)
    return F_mat

# calculate the homography matrix
def motion_estimation_H(pts_2d_src, pts_2d_dst):

    src = np.float32(pts_2d_src)
    dst = np.float32(pts_2d_dst)

    H_mat, inliners = cv.findHomography(src,dst, method=cv.FM_RANSAC,
                                        ransacReprojThreshold= 1, maxIters=2000, confidence=0.8)

    return H_mat

# calculate the motion vector based on Harris RAPiD implementation
def motion_estimation_RAPiD(pts_2d_src, pts_2d_dst, pts_3d_model):

    return 1