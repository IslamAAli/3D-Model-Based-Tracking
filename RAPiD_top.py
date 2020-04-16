import cv2 as cv
import imutils
import numpy as np

# -----------------------------------------------------------------------------
# import project modules
import ctrl_pts_manager
import model_3d_manager
import config
import visual_debug
import edge_detection

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

    # Main loop for RAPiD Tracking
    for img_no in range(config.DATASET_SIZE):
        # read new image from the data set
        path = 'synth_data/'+ (str(img_no+1).zfill(4)+'.png')
        print('[Info] Processing image at:', path)
        img_in = cv.imread(path)

        # edge detection for the image
        sobel_img = edge_detection.detect_edges_sobel(img_in)
        # cv.imshow('Object Projection', sobel_img)
        # cv.waitKey(50)

        # extraction of correspondences

        # formulating the least square problem

        # update the object pose (rotation and translation)

        # plot the results

    cv.waitKey(0)
    visual_debug.visualize_3d_pts_img(edge_pts_2d, ctrl_pts_2d)

if __name__ == "__main__":
    main()