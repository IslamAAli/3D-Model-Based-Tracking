import cv2 as cv
import numpy as np
import random
import visual_debug
import config

# ---------------------------------------------------------------------------
# Sampling input edges, edges represented by end points in 3D space
def read_3d_model():
    ctrl_edges = []

    ctrl_edges = artificial_pts(ctrl_edges)
    # visual_debug.visualize_3d_lines(ctrl_edges)
    return ctrl_edges

# ---------------------------------------------------------------------------

def artificial_pts(m_arr):

    m_arr.append([0, 0, 0, 3, 0, 0])
    m_arr.append([3, 0, 0, 3, 3, 0])
    m_arr.append([3, 3, 0, 0, 3, 0])
    m_arr.append([0, 3, 0, 0, 0, 0])

    m_arr.append([0, 0, 3, 3, 0, 3])
    m_arr.append([3, 0, 3, 3, 3, 3])
    m_arr.append([3, 3, 3, 0, 3, 3])
    m_arr.append([0, 3, 3, 0, 0, 3])

    m_arr.append([0, 0, 0, 0, 0, 3])
    m_arr.append([3, 0, 0, 3, 0, 3])
    m_arr.append([0, 3, 0, 0, 3, 3])
    m_arr.append([3, 3, 0, 3, 3, 3])
    return m_arr

def artificial_k_mat():
    k_mat = [[config.K_FX, 0, config.K_CX, 0],
             [0, config.K_FY, config.K_CY, 0],
             [0, 0, 1, 0]]

    k_mat = np.asarray(k_mat)

    return k_mat

def rand_rot_trans():
    rotations = [random.randrange(config.RAND_ROT_LIMIT),
                 random.randrange(config.RAND_ROT_LIMIT),
                 random.randrange(config.RAND_ROT_LIMIT)]

    translations = [random.randrange(config.RAND_TRANS_LIMIT),
                    random.randrange(config.RAND_TRANS_LIMIT),
                    random.randrange(config.RAND_TRANS_LIMIT)]

    return rotations, translations

def model_edges():
    ctrl_edges = []

    ctrl_edges.append([1, -2, 3, 1, -2, -3])
    ctrl_edges.append([1, -2, -3, 1, 2, -3])
    ctrl_edges.append([1, 2, -3, 1, 2, 3])
    ctrl_edges.append([1, -2, 3, 1, 2, 3])

    ctrl_edges.append([-1, -2, 3, -1, -2, -3])
    # ctrl_edges.append([-1, -2, -3, -1, 2, -3])
    # ctrl_edges.append([-1, 2, -3, -1, 2, 3])
    ctrl_edges.append([-1, 2, 3, -1, -2, 3])

    ctrl_edges.append([1, -2, 3, -1, -2, 3])
    ctrl_edges.append([1, -2, -3, -1, -2, -3])
    # ctrl_edges.append([1, 2, -3, -1, 2, -3])
    # ctrl_edges.append([1, 2, 3, -1, 2, 3])

    return ctrl_edges

def camera_mat_gen():
    k_mat = [[config.K_FX, 0, config.K_CX, 0],
             [0, config.K_FY, config.K_CY, 0],
             [0, 0, 1, 0]]

    k_mat = np.asarray(k_mat)

    return k_mat