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
Quat = np.asarray([1,0,0])

def to_quaternion(delta_r):
    alpha=np.linalg.norm(delta_r)
    if(alpha!=0):
        v=np.divide(delta_r,alpha)
    else:
        v=[0,0,0]
    q=np.asarray([np.cos(0.5*alpha),v[0]*np.sin(0.5*alpha),v[1]*np.sin(0.5*alpha),v[2]*np.sin(0.5*alpha)])
    return q

def update_quaternion(q1,q2):
    a1=q1[0]
    a2=q2[0]
    b1=q1[1]
    b2=q2[1]
    c1=q1[2]
    c2=q2[2]
    d1=q1[3]
    d2=q2[3]
    q=np.asarray([(a1*a2) - (b1*b2) - (c1*c2) - (d1*d2),
                  (a1*b2) + (b1*a2) + (c1*d2) - (d1*c2),
                  (a1*c2) - (b1*d2) + (c1*a2) + (d1*b2),
                  (a1*d2) + (b1*c2) - (c1*b2) + (d1*a2)
        ])
    alpha=2*np.arccos(q[0])
    sin=np.sqrt(1-(q[0]**2))
    if (sin < 0.001):
        dx=q[1]
        dy=q[2]
        dz=q[3]
    else:
        dx=q[1]/sin
        dy=q[2]/sin
        dz=q[3]/sin
    angles = np.multiply([dx,dy,dz],alpha)
    return angles

attitude_mat = np.zeros([4,4])
def attitude(OBJ_R,OBJ_T):
    attitude_mat = np.asarray([
        [0, -OBJ_R[2], OBJ_R[1], OBJ_T[0]],
        [OBJ_R[2], 0, -OBJ_R[0], OBJ_T[1]],
        [-OBJ_R[1], OBJ_R[0], 0, OBJ_T[2]],
        [0,0,0,1]
    ])
    return attitude_mat


# rotation and translation randomization limits
RAND_ROT_LIMIT      = 10
RAND_TRANS_LIMIT    = 5