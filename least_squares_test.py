import numpy as np
import motion_estimation
import ctrl_pts_manager
import config

P_MAT = [[641.9337, 1349.3726, -227.9790, 19295.7949],
         [268.2232, -252.8320, -1603.6173, 13286.0820],
         [-0.6516, 0.6142, -0.4453, 41.3372],
         [0,0,0,1]]

points_3d = [[1, -1, -1, 1],
             [1, -1, 1, 1],
             [-1, -1, 1, 1],
             [-1, -1, -1, 1],
             [1, 1, -1, 1],
             [1, 1, 1, 1],
             [-1, 1, 1, 1],
             [-1, 1, -1, 1]]

transform = np.asarray([[1,0,0,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
points_3d_2 = np.transpose(np.dot(transform,np.transpose(points_3d )))

projection = np.dot(P_MAT, np.transpose(points_3d))
projection[0,:] = np.divide(projection[0,:], projection[2,:])
projection[1,:] = np.divide(projection[1,:], projection[2,:])
projection[2,:] = np.divide(projection[2,:], projection[2,:])

projection_2 = np.dot(P_MAT, np.transpose(points_3d_2))
# projection_2[0,:] = np.divide(projection_2[0,:], projection_2[2,:])
# projection_2[1,:] = np.divide(projection_2[1,:], projection_2[2,:])
# projection_2[2,:] = np.divide(projection_2[2,:], projection_2[2,:])

inv_p_mat = np.linalg.inv(P_MAT)
test_pts = np.dot(inv_p_mat, projection_2)
#
test_pts[0,:] = np.divide(test_pts[0,:], test_pts[3,:])
test_pts[1,:] = np.divide(test_pts[1,:], test_pts[3,:])
test_pts[2,:] = np.divide(test_pts[2,:], test_pts[3,:])
test_pts[3,:] = np.divide(test_pts[3,:], test_pts[3,:])
print(np.transpose(test_pts))


# projection = np.asarray(projection)
# projection_2 = np.asarray(projection_2)
# points_3d = np.asarray(points_3d)
#
# src= np.transpose(projection)
# dst= np.transpose(projection_2)
# res = motion_estimation.motion_estimation_harris_enhanced(src,dst, points_3d)
# # print(res)
#
# mat = np.concatenate ((np.asarray(res[0:9]).reshape((3,3)), np.asarray(res[9:12]).reshape((3,1))), axis=1)
# mat = np.concatenate((mat, np.asarray([0,0,0,1]).reshape(1,4)), axis=0)
# # print(mat)
#
#
# trans = np.dot(mat, np.transpose(points_3d))
# # print(np.transpose(trans))
