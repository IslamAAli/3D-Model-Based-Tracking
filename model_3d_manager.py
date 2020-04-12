import cv2 as cv
import numpy as np
import visual_debug
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