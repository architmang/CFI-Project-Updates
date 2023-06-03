import pickle
import cv2
import numpy as np

# Define the calibration settings
# root_dir = "./calibration0528"
root_dir = "./AzureKinectRecord_30_05"
col = 11
row = 8
square_size = 59
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

new_intr_pickle_file = root_dir + "/intrinsic_param.pkl"
new_extr_pickle_file = root_dir + "/extrinsic_param.pkl"

# Create the object points
objp = np.zeros((col*row, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size

# Initialize variables
intr_param = {}
extr_param = {}

# Define the camera list
cam_list = ['azure_kinect1_2_calib_snap','azure_kinect1_3_calib_snap','azure_kinect2_4_calib_snap', 'azure_kinect2_5_calib_snap', 'azure_kinect3_2_calib_snap', 'azure_kinect3_3_calib_snap']
start_ind = [0, 25, 50, 75, 100, 125]

global img_d_shape, gray_c_shape

# Iterate over the cameras
for cam_ind in range(len(cam_list)):
    
    cam = cam_list[cam_ind]
    print(cam)
    dir = "%s/%s" % (root_dir, cam)

    obj_points = []
    img_points_c = []
    img_points_d = []
    
    # Get the number of images and starting index for the current camera
    # n_images = get_number_of_images(cam)  # Replace with the function to get the number of images
    # start_index = get_start_index(cam)  # Replace with the function to get the starting index
    
    n_images = 181
    start_index = 0

    for i in range(start_ind[cam_ind], start_ind[cam_ind] + n_images):
        flip=False

        # Read the color image
        filename_c = '%s/color%04i.jpg' % (dir, i + start_index)
        img_c = cv2.imread(filename_c)
        gray_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        gray_c_shape = gray_c.shape

        # Read the infrared image
        filename_d = '%s/infrared%04i.png' % (dir, i + start_index)
        img_d = cv2.imread(filename_d, -1)

        global gray_d, ret_d, corners_d
        for thr in [0.05, 0.02, 0.1]:
            gray_d = np.clip(img_d.astype(np.float32) * thr, 0, 255).astype('uint8')
            ret_d, corners_d = cv2.findChessboardCorners(gray_d, (col, row), None)
            if ret_d:
                break

        img_d_shape = img_d.shape

        # Find chessboard corners in the color image
        ret_c, corners_c = cv2.findChessboardCorners(gray_c, (col, row), None)
        
        # cv2.drawChessboardCorners(gray_d, (col, row), corners2_d, ret_d)
        # cv2.imshow('gray_d', gray_d)
        # cv2.waitKey(50)

        if ret_c and ret_d:
            obj_points.append(objp)
            
            # Refine corner coordinates for color image
            corners2_c = cv2.cornerSubPix(gray_c, corners_c, (5, 5), (-1, -1), criteria)
            corners2_d = cv2.cornerSubPix(gray_d, corners_d, (5, 5), (-1, -1), criteria)

            cv2.drawChessboardCorners(img_c, (col, row), corners2_c, ret_c)
            cv2.imshow('img_c', cv2.resize(img_c, (int(img_c.shape[1]/2), int(img_c.shape[0]/2))))
            cv2.waitKey(50)
            
            cv2.drawChessboardCorners(gray_d, (col, row), corners2_d, ret_d)
            cv2.imshow('img_d', gray_d)
            cv2.waitKey(50)

            # flip
            vec_d = (corners2_d[0, 0, :] - corners2_d[-1, 0, :]) / np.linalg.norm(corners2_d[0, 0, :] - corners2_d[-1, 0, :])
            vec_c = (corners2_c[0, 0, :] - corners2_c[-1, 0, :]) / np.linalg.norm(corners2_c[0, 0, :] - corners2_c[-1, 0, :])
            if np.dot(vec_d, vec_c) < 0:
                # flip cn_c
                corners2_d = corners2_d[::-1, :]
                flip = True

            img_points_c.append(corners2_c)
            img_points_d.append(corners2_d)

        print(flip, ret_d, ret_c, filename_c)
        
    print(len(obj_points), len(img_points_c), len(img_points_d))
    
    # Calibrate color camera intrinsic parameters
    ret_c, mtx_c, dist_c, _, _ = cv2.calibrateCamera(obj_points, img_points_c, (gray_c_shape[1], gray_c_shape[0]), None, None, flags=cv2.CALIB_RATIONAL_MODEL)

    print("Color camera:")
    print("ret:\n", ret_c)
    print("Camera matrix:\n", mtx_c)
    print("Distortion coefficients:\n", dist_c)
    
    # Calibrate depth camera intrinsic parameters
    ret_d, mtx_d, dist_d, _, _ = cv2.calibrateCamera(obj_points, img_points_d, (img_d_shape[1], img_d_shape[0]), None, None, flags=cv2.CALIB_RATIONAL_MODEL)

    print("Depth camera:")
    print("ret:\n", ret_d)
    print("Camera matrix:\n", mtx_d)
    print("Distortion coefficients:\n", dist_c)

    print(f"gray_c_shape is {gray_c_shape}, img_d_shape is {img_d_shape}")

    retval, _, _, _, _, R, T, _, _ = \
        cv2.stereoCalibrate(objectPoints=obj_points,
                            imagePoints1=img_points_d,
                            imagePoints2=img_points_c,
                            imageSize=gray_c_shape,
                            cameraMatrix1=mtx_d,
                            distCoeffs1=dist_d,
                            cameraMatrix2=mtx_c,
                            distCoeffs2=dist_c,
                            criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 200, 1e-6),
                            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)
    
    print("ret:\n", retval)
    print("R:\n", R)
    print("T:\n", T)

    params_d = (mtx_d[0, 0], mtx_d[1, 1], mtx_d[0, 2], mtx_d[1, 2], img_d_shape[0], img_d_shape[1]) + tuple(dist_d[0, 0:8])
    print(params_d)
    intr_param['%s_depth' % cam] = params_d

    params_c = (mtx_c[0, 0], mtx_c[1, 1], mtx_c[0, 2], mtx_c[1, 2], gray_c_shape[0], gray_c_shape[1]) + tuple(dist_c[0, 0:8])
    print(params_c)
    intr_param['%s_color' % cam] = params_c
    
    extr_param['%s_d2c' % cam] = (R, T)

# Store the intrinsic parameters in a pickle file
with open(new_intr_pickle_file, 'wb') as f:
    pickle.dump(intr_param, f)

# Store the intrinsic parameters in a pickle file
with open(new_extr_pickle_file, 'wb') as f:
    pickle.dump(extr_param, f)