import bpy
import bpy_extras
import numpy
from mathutils import Matrix
from mathutils import Vector


# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT, R_world2cv, T_world2cv

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT, R, T = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT, R, T

def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))
# ----------------------------------------------------------
if __name__ == "__main__":
    
    # Insert your camera name here
    cam = bpy.data.objects['Camera']
    P, K, RT, R, T= get_3x4_P_matrix_from_blender(cam)
    print("K")
    print(K)
    print("RT")
    print(RT)
    print("P")
    print(P)
    print("R")
    print(R)
    print("T")
    print(T)

    print("==== Tests ====")
    e1 = Vector((1, 0,    0, 1))
    e2 = Vector((0, 1,    0, 1))
    e3 = Vector((0, 0,    1, 1))
    O  = Vector((0, 0,    0, 1))

    p1 = P @ e1
    p1 /= p1[2]
    print("Projected e1")
    print(p1)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e1[0:3])))

    p2 = P @ e2
    p2 /= p2[2]
    print("Projected e2")
    print(p2)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e2[0:3])))

    p3 = P @ e3
    p3 /= p3[2]
    print("Projected e3")
    print(p3)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e3[0:3])))

    pO = P @ O
    pO /= pO[2]
    print("Projected world origin")
    print(pO)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(O[0:3])))

    # Bonus code: save the 3x4 P matrix into a plain text file
    # Don't forget to import numpy for this
    nP = numpy.matrix(P)
    numpy.savetxt("/tmp/P3x4.txt", nP)  # to select precision, use e.g. fmt='%.2f'