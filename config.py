import numpy as np

# number of points to be sampled on each control edge
CTRL_PTS_PER_EDGE   = 10
DATASET_SIZE        = 800

# intrinsic camera matrix parameters
K_FX = 1422.2222    # unit: mm
K_FY = 1600         # unit: mm
K_CX = 1024/2
K_CY = 768/2

EXT_R = [63.5593, 0, 46.6919]  ## unit: degree
EXT_T = [28.3589, -24.9258, 16.9583]  ## unit: m

P_MAT = [[641.9337, 1349.3726, -227.9790, 19295.7949],
         [210.6208, -198.5349, -1444.4358, 13573.5723],
         [-0.6515, 0.6142, -0.4453, 41.3372]]

# camera matrices
K_MAT   = [[1422.2222, 0, 512], [0, 1422.2222, 384], [0,0,1]]

RT_MAT  = [[0.6859, 0.7277, 0, -1.3140],
           [0.3240, -0.3054, -0.8954, -1.6171],
           [-0.6516, 0.6142, -0.4453, 41.3372],
           [0, 0, 0, 1]]

RT_MAT_33  = [  [0.6859,     0.7277,    0,          -1.3140],
                [0.3240,    -0.3054,    -0.8954,    -1.6171],
                [-0.6516,   0.6142,     -0.4453,    41.3372]]

R_MAT = [[0.6859, 0.7277, 0],
        [0.3240, -0.3054, -0.8954],
        [-0.6516, 0.6142, -0.4453]]


OBJ_POSE    = np.identity(4)
OBJ_R       = np.zeros([3,1])
OBJ_T       = np.zeros([3,1])
DELTA_P     = np.zeros([6,1])
T_MAT = [[-1.3140],[-1.6171],[41.3372]]
# rotation and translation randomization limits
RAND_ROT_LIMIT      = 10
RAND_TRANS_LIMIT    = 5