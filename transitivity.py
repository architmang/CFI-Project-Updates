#  since we have calibration between RGB(cam1) and RGB(cam2) and
#  calibration results between RGB(cam2) and Depth(cam2)
#  we can achieve calibration between RGB(cam1) and Depth(cam2)

# we use UNDISTORTION
import pickle
import cv2
import numpy as np
import itertools


def compute_transform_matrices(R_AB, T_AB, R_BC, T_BC):
    # Create transformation matrices T_AB and T_BC
    T_AB = np.hstack((R_AB, T_AB.reshape(3, 1)))
    T_AB = np.vstack((T_AB, np.array([0, 0, 0, 1])))
    T_BC = np.hstack((R_BC, T_BC.reshape(3, 1)))
    T_BC = np.vstack((T_BC, np.array([0, 0, 0, 1])))

    # Compute transformation matrix T_AC
    T_AC = np.matmul(T_AB, T_BC)

    # Extract rotation matrix R_AC and translation vector T_AC
    R_AC = T_AC[:3, :3]
    T_AC = T_AC[:3, 3]

    return R_AC, T_AC

# Define the calibration settings
col = 11
row = 8
square_size = 59
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

root_dir = "./latest_data"
# Load camera intrinsic parameters
intr_param = pickle.load(open('%s/new_intrinsic_param.pkl' % root_dir, 'rb'))
cam_list = ['azure_kinect1_2','azure_kinect1_3','azure_kinect1_4']

depth_dirs = []
rgb_dirs   = []

for cam in cam_list:
    depth_dirs.append(root_dir + "/" + cam + "_calib_snap/")
    rgb_dirs.append(root_dir + "/" + cam + "_calib_snap/")

print(depth_dirs)
print(rgb_dirs)

pairwise_extr_pickle_file = root_dir + "/pairwise_color_extrinsic.pkl"
pairwise_extr_color_params = pickle.load(open(pairwise_extr_pickle_file, 'rb'))

new_color2depth_extr_pickle_file = root_dir + "/pairwise_color2depth_extrinsic.pkl"
pairwise_color2depth_extr_param = {}

new_extr_pickle_file = root_dir + "/new_extrinsic_param.pkl"
extr_param = pickle.load(open(new_extr_pickle_file, 'rb'))
# print(f"new_extr_param are {extr_param}")

# Get all combinations of Kinect pairs
kinect_pairs = list(itertools.combinations(range(len(depth_dirs)), 2))
print(kinect_pairs)

# Iterate over each Kinect pair
for kinect_pair in kinect_pairs:
        
    print(f"kinect pair is {kinect_pair}")
    kinect1_index, kinect2_index = kinect_pair

    cam1 = cam_list[kinect1_index]
    cam2 = cam_list[kinect2_index]

    print(cam1, cam2)

    R_1c2c, T_1c2c = pairwise_extr_color_params["%s_color2_%s_color" % (cam1, cam2)]
    # print(f"R_1c2c, T_1c2c are {R_1c2c, T_1c2c}")

    R_2c2d, T_2c2d = extr_param["%s_calib_snap_d2c" % (cam2)]
    # print(f"R_2c2d, T_2c2d are {R_2c2d, T_2c2d}")

    R_1c2d, T_1c2d = compute_transform_matrices(R_1c2c, T_1c2c, R_2c2d, T_2c2d)
    pairwise_color2depth_extr_param["%s_color2_%s_depth" % (cam1, cam2)] = R_1c2d, T_1c2d

print(pairwise_color2depth_extr_param)

with open(new_color2depth_extr_pickle_file, 'wb') as f:
    pickle.dump(pairwise_color2depth_extr_param, f)