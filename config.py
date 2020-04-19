import numpy as np

# number of points to be sampled on each control edge
CTRL_PTS_PER_EDGE   = 5
DATASET_SIZE        = 800

# intrinsic camera matrix parameters
K_FX = 1422.2222    # unit: mm
K_FY = 1600         # unit: mm
K_CX = 1024/2
K_CY = 768/2

EXT_R = [63.5593, 0, 46.6919]  ## unit: degree
EXT_T = [28.3589, -24.9258, 16.9583]  ## unit: m

P_MAT = [[641.9337, 1349.3726, -227.9790, 19295.7949],
         [268.2232, -252.8320, -1603.6173, 13286.0820],
         [-0.6516, 0.6142, -0.4453, 41.3372]]

OBJ_POSE    = np.identity(4)
OBJ_R       = [0, 0, 0]
OBJ_T       = [0, 0, 0]

# rotation and translation randomization limits
RAND_ROT_LIMIT      = 10
RAND_TRANS_LIMIT    = 5