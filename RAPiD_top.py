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
    # ctrl_edges  = model_3d_manager.read_3d_model()
    # k_mat = model_3d_manager.artificial_k_mat()
    # rot, trans = model_3d_manager.rand_rot_trans()

    ctrl_edges  = model_3d_manager.model_edges()
    k_mat       = model_3d_manager.camera_mat_gen()
    proj_rot    = config.OBJ_R
    proj_trans  = config.OBJ_T
    ext_rot     = config.EXT_R
    ext_trans   = config.EXT_T

    ctrl_pts_3d, edge_pts_3d, ctrl_pts_tags = ctrl_pts_manager.sample_edges(ctrl_edges, config.CTRL_PTS_PER_EDGE)
    edge_pts_2d = ctrl_pts_manager.project_ctrl_pts(edge_pts_3d, proj_rot, proj_trans)
    ctrl_pts_2d = ctrl_pts_manager.project_ctrl_pts(ctrl_pts_3d, proj_rot, proj_trans)

    # Main loop for RAPiD Tracking
#    for img_no in range(config.DATASET_SIZE):
    for img_no in range(2):   
        # read new image from the data set
        path = 'synth_data/'+ (str(img_no+1).zfill(4)+'.png')
        print('[Info] Processing image at:', path)
        img_in = cv.imread(path)

        # edge detection for the image
        sobel_img = edge_detection.detect_edges_sobel(img_in)
        # cv.imshow('Object Projection', sobel_img)
        # cv.waitKey(50)

        flipped_ctrl_pts = []
        for i in range(ctrl_pts_2d.shape[0]):
            flipped_ctrl_pts.append([ctrl_pts_2d[i,1], ctrl_pts_2d[i,0]])

        flipped_ctrl_pts = np.asarray(flipped_ctrl_pts)

        # extraction of correspondences
        ctrl_pts_2d_matched = controlPointMatching.controlPointMatching(flipped_ctrl_pts, sobel_img, ctrl_pts_tags)

        # formulating the least square problem
        ctrl_pts_src, ctrl_pts_dst = ctrl_pts_manager.filter_ctrl_pts(ctrl_pts_2d, ctrl_pts_2d_matched)
        
        from matplotlib import pyplot as plt
        mp=ctrl_pts_2d_matched
        cp=flipped_ctrl_pts 
        fig,ax=plt.subplots()
        ax.imshow(sobel_img,cmap='gray')
        ax.scatter(mp[:,1],mp[:,0],s=5,lw=1,facecolor="none",edgecolor="blue")
        ax.scatter(cp[:,1],cp[:,0],s=5,lw=1,facecolor="none",edgecolor="red")
#        visual_debug.visualize_2d_pts_img(sobel_img, ctrl_pts_src, ctrl_pts_dst)
        cv.waitKey(0)

        print('t')
        # mat = motion_estimation.motion_estimation_F(ctrl_pts_src, ctrl_pts_dst)
        # print(mat)
        # print('===============')

        # update the object pose (rotation and translation)

        # plot the results

    cv.waitKey(0)


if __name__ == "__main__":
    main()