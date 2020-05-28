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

# calculate the motion vector based on Harris RAPiD implementation
def motion_estimation_harris_enhanced(pts_2d_src, pts_2d_dst, pts_3d_model,normal):
    if pts_2d_src.shape[1] == 2:
        ones_vec = np.ones([pts_2d_src.shape[0], 1])
        pts_2d_src_con = np.concatenate((pts_2d_src, ones_vec), axis=1)
        pts_2d_dst_con = np.concatenate((pts_2d_dst, ones_vec), axis=1)
    else:
        pts_2d_src_con = pts_2d_src
        pts_2d_dst_con = pts_2d_dst
    k=np.zeros((2,pts_3d_model.shape[0]))
    
    
   # transform point to be in camera frame (rotation only)
    pts_3d_cam_R = np.transpose(np.dot((config.R_MAT),np.transpose(pts_3d_model)))
    cam_T = np.dot(1, config.T_MAT)
#     Normalize the points
    pts_2d_src_con = np.dot(np.linalg.inv(config.K_MAT), np.transpose(pts_2d_src_con))
    # k[0,:]=np.divide(np.divide(pts_3d_cam_R[:,0]+cam_T[0],pts_3d_cam_R[:,2]+cam_T[2]).T,pts_2d_src_con[0,:])
    # k[1,:]=np.divide(np.divide(pts_3d_cam_R[:,1]+cam_T[1],pts_3d_cam_R[:,2]+cam_T[2]).T,pts_2d_src_con[1,:]) 
    # pts_2d_src_con[0, :] = np.multiply(pts_2d_src_con[0,:],k[0,:])
    # pts_2d_src_con[1, :] = np.multiply(pts_2d_src_con[1,:],k[1,:])
    pts_2d_src_con[2, :] = np.divide(pts_2d_src_con[2, :], pts_2d_src_con[2, :])
    pts_2d_src_con = np.transpose(pts_2d_src_con)
#    c1=np.mean(pts_2d_src_con[:,0])
#    c2=np.mean(pts_2d_src_con[:,1])
#    xnew=pts_2d_src_con[:,0]-c1
#    ynew=pts_2d_src_con[:,1]-c2
#    su=1/(np.std(xnew))
#    sv=1/(np.std(ynew))
#    Tform=np.dot(np.array([[su,0,0],[0,sv,0],[0,0,1]]),np.array([[1,0,-c1],[0,1,-c2],[0,0,1]]))
#    pts_2d_src_con=np.dot(Tform,pts_2d_src_con.T).T
    pts_2d_dst_con = np.dot(np.linalg.inv(config.K_MAT), np.transpose(pts_2d_dst_con))
    # pts_2d_dst_con[0, :] = np.multiply(pts_2d_dst_con[0,:],k[0,:])
    # pts_2d_dst_con[1, :] = np.multiply(pts_2d_dst_con[1,:],k[1,:])
    pts_2d_dst_con[2, :] = np.divide(pts_2d_dst_con[2, :], pts_2d_dst_con[2, :])
    pts_2d_dst_con = np.transpose(pts_2d_dst_con)
#    c1=np.mean(pts_2d_dst_con[:,0])
#    c2=np.mean(pts_2d_dst_con[:,1])
#    xnew=pts_2d_dst_con[:,0]-c1
#    ynew=pts_2d_dst_con[:,1]-c2
#    su=1/(np.std(xnew))
#    sv=1/(np.std(ynew))
#    Tform=np.dot(np.array([[su,0,0],[0,sv,0],[0,0,1]]),np.array([[1,0,-c1],[0,1,-c2],[0,0,1]]))
#    pts_2d_dst_con=np.dot(Tform,pts_2d_dst_con.T).T

    l_init = np.subtract(pts_2d_dst_con, pts_2d_src_con)


    l_vec = np.zeros([2*pts_2d_src_con.shape[0], 1])
    # l_vec = np.zeros([pts_2d_src_con.shape[0], 1])
    w_mat = np.zeros([2 * pts_2d_src_con.shape[0], 6])

 
 
    c=np.zeros((pts_2d_src.shape[0],6))

    for i in range(pts_2d_src_con.shape[0]):

        # preparations
        div_factor = cam_T[2] + pts_3d_cam_R[i,2]

        if div_factor == 0:
            continue

        u = np.divide((pts_3d_cam_R[i,0]+cam_T[0]),div_factor)
        v = np.divide((pts_3d_cam_R[i,1]+cam_T[1]),div_factor)
        # u=pts_2d_src_con[i,0]
        # v=pts_2d_src_con[i,1]

        # -- lengths vector
        # l_vec[i]=(normal[2*i]*l_init[i,0]) + (normal[2*i + 1]*l_init[i,1])
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
        # c[i,:]=(normal[2*i]*w_mat[2*i,:] + normal[2*i+1]*w_mat[2*i+1,:])
        
    # A = np.dot(c.T,c)
    # y=-1*np.dot(c.T,l_vec)
    delta_p, residuals, rank, s = np.linalg.lstsq(w_mat, l_vec, rcond=None)
    # delta_p, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    # error = l_vec + np.dot(c,delta_p)
    delta_t = np.dot(np.linalg.inv(config.R_MAT), delta_p[3:6])
    delta_r = np.dot(np.linalg.inv(config.R_MAT), delta_p[0:3])

    return delta_p, delta_t, delta_r