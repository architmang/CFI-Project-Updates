import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import pickle 

# Load the point cloud from file
bg_data_dir = './AzureKinectRecord_30_05/'
bg_img_idx = 175  # 121
data_dir = './AzureKinectRecord_30_05/'
group_name = '1'
cam = 'azure_kinect1_2'
frame_idx = 20
path_to_color_image = "%s/%s/%s/color/color%04i.jpg" % (data_dir, group_name, cam, frame_idx)
cam_list = ['azure_kinect1_2','azure_kinect1_3','azure_kinect2_4', 
            'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

save_name = "%s/%s/point_cloud/pose%04i.ply" % (data_dir, group_name, frame_idx)
point_cloud = o3d.io.read_point_cloud(save_name)

# Load the intrinsic parameters
with open(data_dir + 'intrinsic_param.pkl', 'rb') as f:
    intr_params = pickle.load(f)

# load kinect extrinsic parameters
with open('%s/kinect_extrinsic_param.pkl' % data_dir, 'rb') as f:
    kinect_extr_params = pickle.load(f)
    print(kinect_extr_params.keys())

# load extrinsic parameters
with open('%s/extrinsic_param.pkl' % data_dir, 'rb') as f:
    extr_params = pickle.load(f)
    print(extr_params.keys())

# Extract the intrinsic parameters for depth
fx_d = intr_params['azure_kinect1_2_depth'][0]
fy_d = intr_params['azure_kinect1_2_depth'][1]
cx_d = intr_params['azure_kinect1_2_depth'][2]
cy_d = intr_params['azure_kinect1_2_depth'][3]
width_d = intr_params['azure_kinect1_2_depth'][5]
height_d = intr_params['azure_kinect1_2_depth'][4]

# Extract the intrinsic parameters for color
fx_c = intr_params['azure_kinect1_2_color'][0]
fy_c = intr_params['azure_kinect1_2_color'][1]
cx_c = intr_params['azure_kinect1_2_color'][2]
cy_c = intr_params['azure_kinect1_2_color'][3]
width_c = intr_params['azure_kinect1_2_color'][5]
height_c = intr_params['azure_kinect1_2_color'][4]

# Create the intrinsic matrix for depth
mtx_d = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])

# Convert the point cloud to UVD
points = np.asarray(point_cloud.points)
uvd = np.dot(points, mtx_d.T)
u = uvd[:, 0] / uvd[:, 2]
v = uvd[:, 1] / uvd[:, 2]
d = uvd[:, 2]

# Create a depth image from the UVD coordinates
depth_img = np.zeros((int(height_d), int(width_d)), dtype=np.float32)
color_img = cv2.imread(f"{data_dir}/{group_name}/{cam}/color/color{frame_idx:04d}.jpg")
depth_img[v.astype(int), u.astype(int)] = d

# Create the intrinsic matrix for color
mtx_c = np.array([[fx_c, 0, cx_c], [0, fy_c, cy_c], [0, 0, 1]])

# Convert the point cloud to UV coordinates
uv = np.dot(points, mtx_c.T)
u = uv[:, 0] / uv[:, 2]
v = uv[:, 1] / uv[:, 2]

mtx_d = np.array([[intr_params['%s_depth' % cam][0], 0, intr_params['%s_depth' % cam][2]],
                    [0, intr_params['%s_depth' % cam][1], intr_params['%s_depth' % cam][3]],
                    [0, 0, 1]])
dist_d = intr_params['%s_depth' % cam][6:14]
# print(mtx_d, '\n', dist_d)

mtx_c = np.array([[intr_params['%s_color' % cam][0], 0, intr_params['%s_color' % cam][2]],
                    [0, intr_params['%s_color' % cam][1], intr_params['%s_color' % cam][3]],
                    [0, 0, 1]])
dist_c = intr_params['%s_color' % cam][6:14]
# print(mtx_c, '\n', dist_c)

h,  w = depth_img.shape[:2]
# new_mtx_d, _ = cv2.getOptimalNewCameraMatrix(mtx_d, dist_d, (w,h), 0, (w,h))
# depth_img = cv2.undistort(depth_img, mtx_d, dist_d, None, new_mtx_d)
# print(mtx_d, '\n', new_mtx_d)

h,  w = color_img.shape[:2]
# new_mtx_c, _ = cv2.getOptimalNewCameraMatrix(mtx_c, dist_c, (w,h), 0, (w,h))
# rgb_img = cv2.undistort(color_img, mtx_c, dist_c, None, new_mtx_c)
# print(mtx_c, '\n', new_mtx_c)

# below two lines do not take into consideration the undistotion part
new_mtx_c, new_mtx_d = mtx_c, mtx_d
rgb_img = color_img

# Perform depth-to-color registration
V, U = depth_img.shape[:2]
V_c, U_c, _ = color_img.shape

K_depth = np.eye(4)
K_depth[0:3, 0:3] = new_mtx_d

K_color = np.eye(4)
K_color[0:3, 0:3] = new_mtx_c

# Depth to color registration matrix
R, T = kinect_extr_params[f"{cam}_d2c"]
M = np.eye(4)
M[0:3, 0:3] = R
M[0:3, 3] = np.squeeze(T)
H = np.dot(np.dot(K_color, M), np.linalg.inv(K_depth))
# print(M)

depth_register = np.zeros([V, U, 3], dtype=np.uint8)
for v in range(V):
    for u in range(U):
        d = depth_img[v, u]
        if d != 0:
            u_rgb = np.dot(H[0, :], np.array([u, v, 1., 1. / d]))
            v_rgb = np.dot(H[1, :], np.array([u, v, 1., 1. / d]))
            if 0 < u_rgb < U_c and 0 < v_rgb < V_c:
                depth_register[v, u, :] = color_img[int(v_rgb), int(u_rgb), :]

# Print intermediate results
print("Depth Image Shape:", depth_img.shape)
print("Color Image Shape:", rgb_img.shape)
# print("Depth-to-Color Registration Matrix:")
# print(H)

# Plot depth and depth-to-color registered images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(depth_img, cmap='gray')
plt.title('Depth Image')

plt.subplot(122)
plt.imshow(depth_register)
plt.title('Depth-to-Color Registered Image')

plt.tight_layout()
plt.show()

# # Visualize the depth image
# cv2.imshow('Depth Image', depth_img)
# cv2.waitKey(0)


# # THE BELOW CODE PLOTS COLOR IMAGES FROM 6 CAMS
# # Load the color images for all cameras
# color_images = []
# for cam in cam_list:
#     color_img = cv2.imread(f"{data_dir}/{group_name}/{cam}/color/color{frame_idx:04d}.jpg")
#     color_images.append(color_img)

# # Create a figure with six subplots
# fig, axs = plt.subplots(2, 3, figsize=(12, 8))
# fig.suptitle('Color Images - Frame 20')

# # Plot the color images in subplots
# for cam, ax in enumerate(axs.flat):
#     ax.imshow(cv2.cvtColor(color_images[cam], cv2.COLOR_BGR2RGB))
#     ax.set_title('Camera {}'.format(cam))
#     ax.axis('off')

# plt.tight_layout()
# plt.show()
