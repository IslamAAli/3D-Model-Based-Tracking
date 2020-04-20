import numpy as np
import motion_estimation
import ctrl_pts_manager
import config

P_MAT = [[641.9337, 1349.3726, -227.9790, 19295.7949],
         [268.2232, -252.8320, -1603.6173, 13286.0820],
         [-0.6516, 0.6142, -0.4453, 41.3372]]

points_3d = [[1, -1, -1, 1],
             [1, -1, 1, 1],
             [-1, -1, 1, 1],
             [-1, -1, -1, 1],
             [1, 1, -1, 1],
             [1, 1, 1, 1],
             [-1, 1, 1, 1],
             [-1, 1, -1, 1]]

transform = np.asarray([[1,0,0,1], [0,1,0,1], [0,0,1,1], [0,0,0,1]])
points_3d_2 = np.transpose(np.dot(transform,np.transpose(points_3d )))

projection = np.dot(P_MAT, np.transpose(points_3d))
projection[0,:] = np.divide(projection[0,:], projection[2,:])
projection[1,:] = np.divide(projection[1,:], projection[2,:])
projection[2,:] = np.divide(projection[2,:], projection[2,:])

projection_2 = np.dot(P_MAT, np.transpose(points_3d_2))
projection_2[0,:] = np.divide(projection_2[0,:], projection_2[2,:])
projection_2[1,:] = np.divide(projection_2[1,:], projection_2[2,:])
projection_2[2,:] = np.divide(projection_2[2,:], projection_2[2,:])

src = np.transpose(projection)
dst = np.transpose(projection_2)

points_3d = np.asarray(points_3d)
delta_p = motion_estimation.motion_estimation_harris_enhanced(src, dst, points_3d[:, 0:3])
print(delta_p)

delta_t = np.dot(np.linalg.inv(config.R_MAT),delta_p[3:6])
delta_r = np.dot(np.linalg.inv(config.R_MAT),delta_p[0:3])
print('--------')
print(delta_p[0:3])
print(delta_r)
print('--------')
print(delta_p[3:6])
print(delta_t)

