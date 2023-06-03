import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

root_dir = "./latest_data"

# Load camera intrinsic parameters
intr_param = pickle.load(open('%s/new_intrinsic_param.pkl' % root_dir, 'rb'))
extr_param = pickle.load(open('%s/new_extrinsic_param.pkl' % root_dir, 'rb'))
cam_list = ['azure_kinect1_2','azure_kinect1_3','azure_kinect1_4']

frame_idx = 0  # Specify the frame index for depth-to-color registration

for cam in cam_list:
    print(cam)
    dir = f"{root_dir}/{cam}"

    # Read depth and color images for the specified frame
    depth_img = cv2.imread(f"{dir}/depth/depth{frame_idx:04d}.png", -1)
    color_img = cv2.imread(f"{dir}/color/color{frame_idx:04d}.jpg")

    if depth_img is None:
        print("Error reading depth image.")
        continue

    if color_img is None:
        print("Error reading color image.")
        continue

    # Retrieve camera intrinsic parameters
    print(cam + "_color", intr_param[cam + "_calib_snap_color"])
    print(cam + "_depth", intr_param[cam + "_calib_snap_depth"])

    mtx_d = np.array([[intr_param['%s_calib_snap_depth' % cam][0], 0, intr_param['%s_calib_snap_depth' % cam][2]],
                      [0, intr_param['%s_calib_snap_depth' % cam][1], intr_param['%s_calib_snap_depth' % cam][3]],
                      [0, 0, 1]])
    dist_d = intr_param['%s_calib_snap_depth' % cam][6:14]
    # print(mtx_d, '\n', dist_d)

    mtx_c = np.array([[intr_param['%s_calib_snap_color' % cam][0], 0, intr_param['%s_calib_snap_color' % cam][2]],
                      [0, intr_param['%s_calib_snap_color' % cam][1], intr_param['%s_calib_snap_color' % cam][3]],
                      [0, 0, 1]])
    dist_c = intr_param['%s_calib_snap_color' % cam][6:14]
    # print(mtx_c, '\n', dist_c)

    h,  w = depth_img.shape[:2]
    new_mtx_d, _ = cv2.getOptimalNewCameraMatrix(mtx_d, dist_d, (w,h), 0, (w,h))
    depth_img = cv2.undistort(depth_img, mtx_d, dist_d, None, new_mtx_d)
    print(mtx_d, '\n', new_mtx_d)

    h,  w = color_img.shape[:2]
    new_mtx_c, _ = cv2.getOptimalNewCameraMatrix(mtx_c, dist_c, (w,h), 0, (w,h))
    rgb_img = cv2.undistort(color_img, mtx_c, dist_c, None, new_mtx_c)
    print(mtx_c, '\n', new_mtx_c)

    # Perform depth-to-color registration
    V, U = depth_img.shape[:2]
    V_c, U_c, _ = color_img.shape

    K_depth = np.eye(4)
    K_depth[0:3, 0:3] = new_mtx_d

    K_color = np.eye(4)
    K_color[0:3, 0:3] = new_mtx_c

    # Depth to color registration matrix
    R, T = extr_param[f"{cam}_calib_snap_d2c"]
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = np.squeeze(T)
    H = np.dot(np.dot(K_color, M), np.linalg.inv(K_depth))
    print(M)

    depth_register = np.zeros([V, U, 3], dtype=np.uint8)
    for v in range(V):
        for u in range(U):
            d = depth_img[v, u]
            if d != 0:
                u_rgb = np.dot(H[0, :], np.array([u, v, 1., 1. / d]))
                v_rgb = np.dot(H[1, :], np.array([u, v, 1., 1. / d]))
                if 0 < u_rgb < U_c and 0 < v_rgb < V_c:
                    depth_register[v, u, :] = color_img[int(v_rgb), int(u_rgb), :]

    # # Print intermediate results
    # print("Depth Image Shape:", depth_img.shape)
    # print("Color Image Shape:", rgb_img.shape)
    # print("Depth-to-Color Registration Matrix:")
    # print(H)

    # Plot depth and depth-to-color registered images
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(depth_img, cmap='gray')
    plt.title('Depth Image')

    plt.subplot(122)
    plt.imshow(depth_register)
    plt.title('Depth-to-Color Registered Image without Distortion')

    plt.tight_layout()
    plt.show()