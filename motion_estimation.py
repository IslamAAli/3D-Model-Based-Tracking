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
def motion_estimation_harris(pts_2d_src, pts_2d_dst, pts_3d_model):

    l_init = np.subtract(pts_2d_dst, pts_2d_src)
    l_vec  = np.zeros([2*pts_2d_src.shape[0],1])
    w_mat  = np.zeros([2*pts_2d_src.shape[0],6])

    for i in range(pts_2d_src.shape[0]):

        # preparations
        div_factor = config.OBJ_T[2] + pts_3d_model[i,2]

        if div_factor == 0:
            continue

        u = pts_2d_src[i,0]
        v = pts_2d_src[i,1]

        # -- lengths vector
        l_vec[2*i] = l_init[i,0]
        l_vec[2*i + 1] = l_init[i,1]

        # -- projection / linearized matrix
        w_mat[2 * i, 0] = -u*pts_3d_model[i,1]
        w_mat[2 * i, 1] = pts_3d_model[i,2] + u*pts_3d_model[i,0]
        w_mat[2 * i, 2] = -pts_3d_model[i,1]
        w_mat[2 * i, 3] = 1
        w_mat[2 * i, 4] = 0
        w_mat[2 * i, 5] = -u

        w_mat[(2*i)+1, 0] = -pts_3d_model[i,2]-v*pts_3d_model[i,1]
        w_mat[(2*i)+1, 1] = v*pts_3d_model[i,0]
        w_mat[(2*i)+1, 2] = pts_3d_model[i,0]
        w_mat[(2*i)+1, 3] = 0
        w_mat[(2*i)+1, 4] = 1
        w_mat[(2*i)+1, 5] = -v

        w_mat[(2*i):(2*i+1), :] = np.divide(w_mat[(2*i):(2*i+1), :],div_factor)

    delta_p, residuals, rank, s = np.linalg.lstsq(w_mat, l_vec, rcond=None)

    return delta_p

def motion_estimation_harris_enhanced(pts_2d_src, pts_2d_dst, pts_3d_model):


    if pts_2d_src.shape[1] == 2:
        pts_2d_src_con = np.concatenate(pts_2d_src, np.ones([pts_2d_src.shape[0],1]), axis=0)
        pts_2d_dst_con = np.concatenate(pts_2d_dst, np.ones([pts_2d_dst.shape[0], 1]), axis=0)
    else:
        pts_2d_src_con = pts_2d_src
        pts_2d_dst_con = pts_2d_dst

    # Normalize the points
    pts_2d_src_con = np.dot(np.linalg.inv(config.K_MAT), np.transpose(pts_2d_src_con))
    pts_2d_src_con[0, :] = np.divide(pts_2d_src_con[0, :], pts_2d_src_con[2, :])
    pts_2d_src_con[1, :] = np.divide(pts_2d_src_con[1, :], pts_2d_src_con[2, :])
    pts_2d_src_con[2, :] = np.divide(pts_2d_src_con[2, :], pts_2d_src_con[2, :])
    pts_2d_src_con = np.transpose(pts_2d_src_con)

    pts_2d_dst_con = np.dot(np.linalg.inv(config.K_MAT), np.transpose(pts_2d_dst_con))
    pts_2d_dst_con[0, :] = np.divide(pts_2d_dst_con[0, :], pts_2d_dst_con[2, :])
    pts_2d_dst_con[1, :] = np.divide(pts_2d_dst_con[1, :], pts_2d_dst_con[2, :])
    pts_2d_dst_con[2, :] = np.divide(pts_2d_dst_con[2, :], pts_2d_dst_con[2, :])
    pts_2d_dst_con = np.transpose(pts_2d_dst_con)

    l_init = np.subtract(pts_2d_dst_con, pts_2d_src_con)


    l_vec = np.zeros([2 * pts_2d_src_con.shape[0], 1])
    w_mat = np.zeros([2 * pts_2d_src_con.shape[0], 6])

    # transform point to be in camera frame (rotation only)
    pts_3d_cam_R = np.transpose(np.dot((config.R_MAT),np.transpose(pts_3d_model)))
    cam_T = np.dot(-1, config.T_MAT)
    # cam_T = np.asarray(config.T_MAT)

    for i in range(pts_2d_src_con.shape[0]):

        # preparations
        div_factor = cam_T[2] + pts_3d_cam_R[i,2]

        if div_factor == 0:
            continue

        u = pts_2d_src_con[i,0]
        v = pts_2d_src_con[i,1]

        # -- lengths vector
        l_vec[2*i] = l_init[i,0]
        l_vec[2*i + 1] = l_init[i,1]

        # -- projection / linearized matrix
        w_mat[2 * i, 0] = -u*pts_3d_cam_R[i,1]
        w_mat[2 * i, 1] = pts_3d_cam_R[i,2] + u*pts_3d_cam_R[i,0]
        w_mat[2 * i, 2] = -pts_3d_cam_R[i,1]
        w_mat[2 * i, 3] = 1
        w_mat[2 * i, 4] = 0
        w_mat[2 * i, 5] = -u

        w_mat[(2*i)+1, 0] = -pts_3d_cam_R[i,2]-v*pts_3d_cam_R[i,1]
        w_mat[(2*i)+1, 1] = v*pts_3d_cam_R[i,0]
        w_mat[(2*i)+1, 2] = pts_3d_cam_R[i,0]
        w_mat[(2*i)+1, 3] = 0
        w_mat[(2*i)+1, 4] = 1
        w_mat[(2*i)+1, 5] = -v

        w_mat[(2*i):(2*i+2), :] = np.divide(w_mat[(2*i):(2*i+2), :],div_factor)

    # print(w_mat)
    # print(l_vec)

    delta_p, residuals, rank, s = np.linalg.lstsq(w_mat, l_vec, rcond=None)

    return delta_p