import cv2 as cv
import numpy as np
import config
import visual_debug

# ---------------------------------------------------------------------------
# Sampling input edges, edges represented by end points in 3D space
def sample_edges(m_edges, m_pts_per_edge):
    ctrl_pts_3d         = []
    ctrl_edge_pts_3d    = []

    for edge in m_edges:
        # extract end points
        edge_start  = edge[0:3]
        edge_end    = edge[3:6]

        ctrl_edge_pts_3d.append(edge_start)
        ctrl_edge_pts_3d.append(edge_end)

        # get increments for sampling
        increment_3d = np.divide(np.subtract(edge_end, edge_start), (m_pts_per_edge+1))

        sampled_ctrl_pt = edge_start
        # generate sampled points (doesn't have corner points)
        for i in range(config.CTRL_PTS_PER_EDGE):
            sampled_ctrl_pt = np.add(sampled_ctrl_pt, increment_3d)
            ctrl_pts_3d.append(sampled_ctrl_pt)

    # convert to numpy array
    ctrl_pts_3d = np.asarray(ctrl_pts_3d)
    ctrl_edge_pts_3d = np.asarray(ctrl_edge_pts_3d)

    # visualize output for debugging
    # visual_debug.visualize_3d_scatter(ctrl_edge_pts_3d)
    # visual_debug.visualize_3d_scatter(ctrl_pts_3d)

    # visualize edges with samples
    visual_debug.visualize_3d_lines_pts(m_edges, ctrl_pts_3d, ctrl_edge_pts_3d)

    return ctrl_pts_3d, ctrl_edge_pts_3d

# ---------------------------------------------------------------------------
# projecting 3d control points to the 2d image and do simple borders filtering
# @ m_ctrl_pts: 3d control points sampled on high contrast edges
# @ m_k: intrinsic camera matrix
# @ m_pose_r : 3d angles representing rotations (in degrees)
# @ m_pose_t : 3d translation vector
def project_ctrl_pts(m_ctrl_pts, m_k, m_pose_r, m_pose_t):

    # convert angles to radians and get the sin/cos values
    pose_r_radian = np.deg2rad(m_pose_r)
    pose_r_sin = np.sin(pose_r_radian)
    pose_r_cos = np.cos(pose_r_radian)

    # constructing the rotation matrix
    rz = [[pose_r_cos[2], -pose_r_sin[2], 0],
          [pose_r_sin[2], pose_r_cos[2],  0],
          [0, 0, 1]]

    ry = [[pose_r_cos[1], 0 , pose_r_sin[1]],
          [0, 1, 0],
          [-pose_r_sin[1], 0, pose_r_cos[1]]]

    rx = [[1, 0, 0],
          [0, pose_r_cos[0], -pose_r_sin[0]],
          [0, pose_r_sin[0], pose_r_cos[0]]]

    rot_mat = np.dot(rz, np.dot(ry, rx))

    # construct the extrinsic matrix
    ext_mat = np.concatenate((rot_mat, np.reshape(m_pose_t, (3, 1))), axis=1)
    last_row = np.reshape([0,0,0,1], (1,4))
    hom_ext_mat = np.concatenate((ext_mat, last_row), axis=0)

    # convert the 3d points to be homogeneous
    ones_col =  np.ones((np.asarray(m_ctrl_pts).shape[0], 1))
    ctrl_pts_3d_hom = np.concatenate((m_ctrl_pts,ones_col), axis=1)

    # do the projection
    ctrl_pts_2d_hom = np.dot(np.dot(m_k, hom_ext_mat), np.transpose(ctrl_pts_3d_hom))

    # do the scaling by the 3rd element
    ctrl_pts_2d_hom[0, :] = np.divide(ctrl_pts_2d_hom[0, :], ctrl_pts_2d_hom[2, :])
    ctrl_pts_2d_hom[1, :] = np.divide(ctrl_pts_2d_hom[1, :], ctrl_pts_2d_hom[2, :])
    ctrl_pts_2d_hom[2, :] = np.divide(ctrl_pts_2d_hom[2, :], ctrl_pts_2d_hom[2, :])

    ctrl_pts_2d = np.transpose(ctrl_pts_2d_hom[0:2, :])

    return ctrl_pts_2d
