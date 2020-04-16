import cv2
import imutils
import numpy as np

# -----------------------------------------------------------------------------
# import project modules
import ctrl_pts_manager
import model_3d_manager
import config
import visual_debug

# -----------------------------------------------------------------------------
def main():
    print('[*] RAPiD is Starting ... ')
    # ctrl_edges  = model_3d_manager.read_3d_model()
    # k_mat = model_3d_manager.artificial_k_mat()
    # rot, trans = model_3d_manager.rand_rot_trans()

    ctrl_edges  = model_3d_manager.model_edges()
    k_mat       = model_3d_manager.camera_mat_gen()
    proj_rot    = config.OBJ_R
    proj_trans  = config.OBJ_T
    ext_rot     = config.EXT_R
    ext_trans   = config.EXT_T

    ctrl_pts_3d, edge_pts_3d = ctrl_pts_manager.sample_edges(ctrl_edges, config.CTRL_PTS_PER_EDGE)
    edge_pts_2d = ctrl_pts_manager.project_ctrl_pts(edge_pts_3d, proj_rot, proj_trans)
    ctrl_pts_2d = ctrl_pts_manager.project_ctrl_pts(ctrl_pts_3d, proj_rot, proj_trans)

    visual_debug.visualize_3d_pts_img(edge_pts_2d, ctrl_pts_2d)


    ## sobel edge detector

if __name__ == "__main__":
    main()