import pickle
import cv2
import numpy as np

# Define the calibration settings
root_dir = "./demo_data"
col = 11
row = 8
square_size = 59

new_pickle_file = root_dir + "new_intrinsic_param.pkl"

# Create the object points
objp = np.zeros((col*row, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size

# Initialize variables
intr_param = {}

# Define the camera list
cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect1_4']

global img_d_shape, gray_c_shape

# Iterate over the cameras
for cam in cam_list:
    print(cam)
    dir = "%s/%s" % (root_dir, cam)

    obj_points = []
    img_points_c = []
    img_points_d = []
    
    # Get the number of images and starting index for the current camera
    # n_images = get_number_of_images(cam)  # Replace with the function to get the number of images
    # start_index = get_start_index(cam)  # Replace with the function to get the starting index
    
    n_images = 78
    start_index = 0

    for i in range(n_images):
        flip=False

        # Read the color image
        filename_c = '%s/color/color%04i.jpg' % (dir, i + start_index)
        img_c = cv2.imread(filename_c)
        gray_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        gray_c_shape = gray_c.shape

        # Read the depth image
        filename_d = '%s/depth/depth%04i.png' % (dir, i + start_index)
        img_d = cv2.imread(filename_d, -1)
        global gray_d
        for thr in [0.05, 0.02, 0.1]:
            gray_d = np.clip(img_d.astype(np.float32) * thr, 0, 255).astype('uint8')
            ret_d, corners_d = cv2.findChessboardCorners(gray_d, (col, row), None)
            if ret_d:
                break

        img_d_shape = img_d.shape
        # print(f"img_d.shape is {img_d.shape}")
        # print(f"gray_d.shape is {gray_d.shape}")

        # Find chessboard corners in the color image
        ret_c, corners_c = cv2.findChessboardCorners(gray_c, (col, row), None)
        
        if ret_c and ret_d:
            obj_points.append(objp)
            
            # Refine corner coordinates for color image
            corners2_c = cv2.cornerSubPix(gray_c, corners_c, (5, 5), (-1, -1), criteria)
            corners2_d = cv2.cornerSubPix(img_d, corners_d, (5, 5), (-1, -1), criteria)

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
    ret_c, mtx_c, dist_c, _, _ = cv2.calibrateCamera(obj_points, img_points_c, gray_c_shape[::-1], None, None)

    print("Color camera:")
    print("ret:\n", ret_c)
    print("Camera matrix:\n", mtx_c)
    print("Distortion coefficients:\n", dist_c)
    
    # Calibrate depth camera intrinsic parameters
    ret_d, mtx_d, dist_d, _, _ = cv2.calibrateCamera(obj_points, img_points_d, img_d_shape[::-1], None, None)

    print("Depth camera:")
    print("ret:\n", ret_d)
    print("Camera matrix:\n", mtx_d)
    print("Distortion coefficients:\n", dist_c)
