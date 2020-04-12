import cv2
import imutils
import numpy as np

# -----------------------------------------------------------------------------
# import project modules
import ctrl_pts_manager
import model_3d_manager
import config

# -----------------------------------------------------------------------------
def main():
    print('[*] RAPiD is Starting ... ')
    ctrl_edges  = model_3d_manager.read_3d_model()
    ctrl_pts_3d, __ = ctrl_pts_manager.sample_edges(ctrl_edges, config.CTRL_PTS_PER_EDGE)
    ctrl_pts_2d = ctrl_pts_manager.project_ctrl_pts(ctrl_pts_3d, None, None, None)

if __name__ == "__main__":
    main()