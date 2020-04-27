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
import controlPointMatching
import motion_estimation

# -----------------------------------------------------------------------------
def main():
    print('[*] RAPiD is Starting ... ')

    ctrl_edges  = model_3d_manager.model_edges()
    ctrl_pts_3d, edge_pts_3d, ctrl_pts_tags = ctrl_pts_manager.sample_edges(ctrl_edges, config.CTRL_PTS_PER_EDGE)

    # Main loop for RAPiD Tracking
    for img_no in range(config.DATASET_SIZE):
#    for img_no in range(2):

        # Perform 3D to 2D projection
        edge_pts_2d = ctrl_pts_manager.project_ctrl_pts(edge_pts_3d, config.OBJ_R, config.OBJ_T)
        ctrl_pts_2d = ctrl_pts_manager.project_ctrl_pts(ctrl_pts_3d, config.OBJ_R, config.OBJ_T)

        # read new image from the data set
        path = 'synth_data/'+ (str(img_no+2).zfill(4)+'.png')
        print('[Info] Processing image at:', path)
        img_in = cv.imread(path)

        # edge detection for the image
        sobel_img = edge_detection.detect_edges_sobel(img_in)

        # extraction of correspondences
        flipped_ctrl_pts = ctrl_pts_manager.flip_pts(ctrl_pts_2d)
        flipped_matched_ctrl_pts = controlPointMatching.controlPointMatching(flipped_ctrl_pts, sobel_img, ctrl_pts_tags)
        ctrl_pts_2d_matched = ctrl_pts_manager.flip_pts(flipped_matched_ctrl_pts)

        # filtering the points to remove infinity values
        ctrl_pts_src, ctrl_pts_dst, ctrl_pts_3d_filtered = ctrl_pts_manager.filter_ctrl_pts(ctrl_pts_2d, ctrl_pts_2d_matched, ctrl_pts_3d)

        # formulating the least square problem
        delta_p, delta_t, delta_r = motion_estimation.motion_estimation_harris_enhanced(ctrl_pts_src, ctrl_pts_dst, ctrl_pts_3d_filtered)
        print(delta_t)
        print(delta_r)

        # update the object pose (rotation and translation)
        config.DELTA_P += delta_p
        config.OBJ_T += delta_t
        config.OBJ_R += delta_r

        # visualize results using visual debug module
        visual_debug.visualize_2d_pts_img(sobel_img, img_in, ctrl_pts_src, ctrl_pts_dst, both=True)

if __name__ == "__main__":
    main()