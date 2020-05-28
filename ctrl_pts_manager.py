import cv2 as cv
import numpy as np
import config
import visual_debug

# ---------------------------------------------------------------------------
# Sampling input edges, edges represented by end points in 3D space
def sample_edges(m_edges, m_pts_per_edge):
    ctrl_pts_3d         = []
    ctrl_edge_pts_3d    = []
    ctrl_pts_tags       = []

    edge_id = 0
    for edge in m_edges:
        # extract end points
        edge_start  = edge[0:3]
        edge_end    = edge[3:6]

        ctrl_edge_pts_3d.append(edge_start)
        # ctrl_edge_pts_3d.append(edge_end)

        # get increments for sampling
        increment_3d = np.divide(np.subtract(edge_end, edge_start), (m_pts_per_edge+1))

        sampled_ctrl_pt = edge_start
        # generate sampled points (doesn't have corner points)
        for i in range(config.CTRL_PTS_PER_EDGE):
            sampled_ctrl_pt = np.add(sampled_ctrl_pt, increment_3d)
            ctrl_pts_3d.append(sampled_ctrl_pt)
            ctrl_pts_tags.append(edge_id)

        edge_id += 1

    # convert to numpy array
    ctrl_pts_3d = np.asarray(ctrl_pts_3d)
    ctrl_edge_pts_3d = np.asarray(ctrl_edge_pts_3d)
    ctrl_pts_tags = np.asarray(ctrl_pts_tags)

    # visualize output for debugging
    # visual_debug.visualize_3d_scatter(ctrl_edge_pts_3d)
    # visual_debug.visualize_3d_scatter(ctrl_pts_3d)

    # visualize edges with samples
    # visual_debug.visualize_3d_lines_pts(m_edges, ctrl_pts_3d, ctrl_edge_pts_3d)

    return ctrl_pts_3d, ctrl_edge_pts_3d, ctrl_pts_tags

# ---------------------------------------------------------------------------
# projecting 3d control points to the 2d image and do simple borders filtering
# @ m_ctrl_pts: 3d control points sampled on high contrast edges
# @ m_k: intrinsic camera matrix
# @ m_pose_r : 3d angles representing rotations (in degrees)
# @ m_pose_t : 3d translation vector
def project_ctrl_pts(m_ctrl_pts, m_proj_pose_r, m_proj_pose_t):

    # get the projection matrix based on the object pose in space
    proj_mat = projection_mat_gen(m_proj_pose_r, m_proj_pose_t)

    # convert the 3d points to be homogeneous
    ones_col =  np.ones((np.asarray(m_ctrl_pts).shape[0], 1))
    ctrl_pts_3d_hom = np.concatenate((m_ctrl_pts,ones_col), axis=1)
    ctrl_pts_3d_proj = project_3d_points_world_frame(ctrl_pts_3d_hom)

    # do the projection (euclidean projection)
    cam_mat = config.P_MAT
    ctrl_pts_2d_hom = np.dot(cam_mat , np.transpose(ctrl_pts_3d_proj))

    # do the scaling by the 3rd element
    ctrl_pts_2d_hom[0, :] = np.divide(ctrl_pts_2d_hom[0, :], ctrl_pts_2d_hom[2, :])
    ctrl_pts_2d_hom[1, :] = np.divide(ctrl_pts_2d_hom[1, :], ctrl_pts_2d_hom[2, :])
    ctrl_pts_2d_hom[2, :] = np.divide(ctrl_pts_2d_hom[2, :], ctrl_pts_2d_hom[2, :])

    ctrl_pts_2d = np.transpose(ctrl_pts_2d_hom[0:2, :])

    return ctrl_pts_2d

# ---------------------------------------------------------------------------
def projection_mat_gen(m_pose_r, m_pose_t):
    # convert angles to radians and get the sin/cos values
    pose_r_radian = np.deg2rad(m_pose_r)
    pose_r_sin = np.sin(pose_r_radian)
    pose_r_cos = np.cos(pose_r_radian)

    # constructing the rotation matrix
    rz = [[pose_r_cos[2], -pose_r_sin[2], 0],
          [pose_r_sin[2], pose_r_cos[2], 0],
          [0, 0, 1]]

    ry = [[pose_r_cos[1], 0, pose_r_sin[1]],
          [0, 1, 0],
          [-pose_r_sin[1], 0, pose_r_cos[1]]]

    rx = [[1, 0, 0],
          [0, pose_r_cos[0], -pose_r_sin[0]],
          [0, pose_r_sin[0], pose_r_cos[0]]]

    rot_mat = np.dot(rz, np.dot(ry, rx))

    # construct the extrinsic matrix
    ext_mat = np.concatenate((rot_mat, np.reshape(m_pose_t, (3, 1))), axis=1)
    last_row = np.reshape([0, 0, 0, 1], (1, 4))
    hom_ext_mat = np.concatenate((ext_mat, last_row), axis=0)

    return  hom_ext_mat


# ---------------------------------------------------------------------------
def filter_ctrl_pts(src_pts, dst_pts, pts_3d):
    src_pts_filtered = []
    dst_pts_filtered = []
    ctrl_pts_filtered = []

    for i in range(dst_pts.shape[0]):
        if (not (np.isinf(dst_pts[i,0]))) and (dst_pts[i,0]!=0):
            src_pts_filtered.append(src_pts[i, :])
            dst_pts_filtered.append(dst_pts[i, :])
            ctrl_pts_filtered.append(pts_3d[i,:])

    src_pts_filtered = np.asarray(src_pts_filtered)
    dst_pts_filtered = np.asarray(dst_pts_filtered)
    ctrl_pts_filtered = np.asarray(ctrl_pts_filtered)

    return src_pts_filtered, dst_pts_filtered, ctrl_pts_filtered

def flip_pts(pts):
    flipped_ctrl_pts = []
    for i in range(pts.shape[0]):
        flipped_ctrl_pts.append([pts[i, 1], pts[i, 0]])
    flipped_ctrl_pts = np.asarray(flipped_ctrl_pts).reshape([pts.shape[0],2])

    return flipped_ctrl_pts

def project_3d_points_world_frame(pts_3d_hom):

    # construct skew symmetric matrix
    omega_mat = np.asarray([
        [0, -config.OBJ_R[2], config.OBJ_R[1], config.OBJ_T[0]],
        [config.OBJ_R[2], 0, -config.OBJ_R[0], config.OBJ_T[1]],
        [-config.OBJ_R[1], config.OBJ_R[0], 0, config.OBJ_T[2]],
        [0,0,0,1]
    ])

    # get the delta to be added to the points.
    init_projection = np.dot(config.attitude_mat, np.transpose(pts_3d_hom))

    # add points to match Harris equations.
    projected_pts = init_projection + np.transpose(pts_3d_hom)

    # convert to homogeneous coordinates
    projected_pts_hom = np.transpose(np.concatenate((projected_pts[0:3,:], np.ones([1, projected_pts.shape[1]])), axis=0))

    return projected_pts_hom