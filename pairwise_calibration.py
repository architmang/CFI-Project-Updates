# we use UNDISTORTION
import pickle
import cv2
import numpy as np
import itertools

# Define the calibration settings
col = 11
row = 8
square_size = 59
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

root_dir = "./AzureKinectRecord"
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

# Get all combinations of Kinect pairs
kinect_pairs = list(itertools.combinations(range(len(depth_dirs)), 2))
print(kinect_pairs)

extr_param = {}
new_extr_pickle_file = root_dir + "/pairwise_color_extrinsic.pkl"

# Iterate over each Kinect pair
for kinect_pair in kinect_pairs:
    # Initialize variables
    objp = np.zeros((col*row, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size
    obj_points = []
    img_points_rgb1 = []
    img_points_rgb2 = []
    
    print(f"kinect pair is {kinect_pair}")
    kinect1_index, kinect2_index = kinect_pair
    
    depth_dir_kinect1 = depth_dirs[kinect1_index]
    depth_dir_kinect2 = depth_dirs[kinect2_index]
    rgb_dir_kinect1 = rgb_dirs[kinect1_index]
    rgb_dir_kinect2 = rgb_dirs[kinect2_index]
    cam1 = cam_list[kinect1_index]
    cam2 = cam_list[kinect2_index]

    mtx_d = np.array([[intr_param['%s_calib_snap_color' % cam1][0], 0, intr_param['%s_calib_snap_color' % cam1][2]],
                    [0, intr_param['%s_calib_snap_color' % cam1][1], intr_param['%s_calib_snap_color' % cam1][3]],
                    [0, 0, 1]])
    dist_d = intr_param['%s_calib_snap_color' % cam1][6:14]
    print(mtx_d, '\n', dist_d)
    mtx_c = np.array([[intr_param['%s_calib_snap_color' % cam2][0], 0, intr_param['%s_calib_snap_color' % cam2][2]],
                    [0, intr_param['%s_calib_snap_color' % cam2][1], intr_param['%s_calib_snap_color' % cam2][3]],
                    [0, 0, 1]])
    dist_c = intr_param['%s_calib_snap_color' % cam2][6:14]
    print(mtx_c, '\n', dist_c)
    
    n_images = 25
    start_index = 0

    for i in range(n_images):
        flip=False
        
        # Read the RGB image from Kinect 1
        rgb_filename_kinect1 = rgb_dir_kinect1 + "/color%04i.jpg" % (i + start_index)
        rgb_img_kinect1 = cv2.imread(rgb_filename_kinect1)
        gray_rgb_kinect1 = cv2.cvtColor(rgb_img_kinect1, cv2.COLOR_BGR2GRAY)
        
        global gray_rgb_shape

        # Read the RGB image from Kinect 2
        rgb_filename_kinect2 = rgb_dir_kinect2 + "/color%04i.jpg" % (i + start_index)
        rgb_img_kinect2 = cv2.imread(rgb_filename_kinect2)
        gray_rgb_kinect2 = cv2.cvtColor(rgb_img_kinect2, cv2.COLOR_BGR2GRAY)
        gray_rgb_shape = gray_rgb_kinect2.shape
        
        # Find chessboard corners in depth and RGB images
        ret_rgb_kinect1, corners_rgb_kinect1 = cv2.findChessboardCorners(gray_rgb_kinect1, (col, row), None)
        ret_rgb_kinect2, corners_rgb_kinect2 = cv2.findChessboardCorners(gray_rgb_kinect2, (col, row), None)
        print(f"ret values are {ret_rgb_kinect1, ret_rgb_kinect2}")

        if ret_rgb_kinect1 and ret_rgb_kinect2:
            obj_points.append(objp)
            
            # Refine corner coordinates for color image
            corners2_cam1 = cv2.cornerSubPix(gray_rgb_kinect1, corners_rgb_kinect1, (5, 5), (-1, -1), criteria)
            corners2_cam2 = cv2.cornerSubPix(gray_rgb_kinect2, corners_rgb_kinect2, (5, 5), (-1, -1), criteria)

            cv2.drawChessboardCorners(rgb_img_kinect1, (col, row), corners2_cam1, ret_rgb_kinect1)
            cv2.imshow('rgb_img_kinect1', cv2.resize(rgb_img_kinect1, (int(gray_rgb_shape[1]/2), int(gray_rgb_shape[0]/2))))
            cv2.waitKey(50)
            
            cv2.drawChessboardCorners(rgb_img_kinect2, (col, row), corners2_cam2, ret_rgb_kinect2)
            cv2.imshow('rgb_img_kinect2', cv2.resize(rgb_img_kinect2, (int(gray_rgb_shape[1]/2), int(gray_rgb_shape[0]/2))))
            cv2.waitKey(50)

            # flip
            vec_cam1 = (corners2_cam1[0, 0, :] - corners2_cam1[-1, 0, :]) / np.linalg.norm(corners2_cam1[0, 0, :] - corners2_cam1[-1, 0, :])
            vec_cam2 = (corners2_cam2[0, 0, :] - corners2_cam2[-1, 0, :]) / np.linalg.norm(corners2_cam2[0, 0, :] - corners2_cam2[-1, 0, :])
            
            if np.dot(vec_cam1, vec_cam2) < 0:
                # flip cn_c
                corners2_cam1 = corners2_cam1[::-1, :]
                flip = True

            img_points_rgb1.append(corners2_cam1)
            img_points_rgb2.append(corners2_cam2)

        print(flip, ret_rgb_kinect1, ret_rgb_kinect2, rgb_filename_kinect1, rgb_filename_kinect2)
        
    print(len(obj_points), len(img_points_rgb1), len(img_points_rgb2))

    # Perform stereo calibration to obtain the extrinsic parameters
    retval, _, _, _, _, R, T, _, _ = \
        cv2.stereoCalibrate(objectPoints=obj_points,
                            imagePoints1=img_points_rgb1,
                            imagePoints2=img_points_rgb2,
                            imageSize=gray_rgb_shape,
                            cameraMatrix1=mtx_d,
                            distCoeffs1=dist_d,
                            cameraMatrix2=mtx_c,
                            distCoeffs2=dist_c,
                            criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 200, 1e-6),
                            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)

    # Print the extrinsic parameters
    print("Extrinsic parameters for Kinect pair:", kinect_pair)
    print("Rotation matrix (R):\n", R)
    print("Translation vector (T):\n", T)
    print("-------------------------------------------")

    extr_param['%s_color2_%s_color' % (cam1, cam2)] = (R, T)

print(extr_param)
# Store the intrinsic parameters in a pickle file
with open(new_extr_pickle_file, 'wb') as f:
    pickle.dump(extr_param, f)