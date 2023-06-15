import sys
sys.path.append('../')
import os
import glob
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import random 
from calibration.utils import *
import pickle
import scipy

def count_points_within_radius(x, y, depth_image, radius=2):
    height, width = depth_image.shape[:2]
    count = 0

    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            new_x = x + i
            new_y = y + j

            if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
                if depth_image[new_y, new_x] > 0:
                    count += 1

    return count

def load_background(data_dir, cam_list, depth_idx=3):
    bg_imgs = {}
    for cam in cam_list:
        bg_file = '%s/%s_calib_snap/depth%04i.png' % (data_dir, cam, depth_idx)
        # bg_file = '%s/2/%s/depth/depth%04i.png' % (data_dir, cam, depth_idx)
        bg_img = cv2.imread(bg_file, -1)

        # plt.figure()
        # plt.imshow(bg_img, cmap='gray')
        # plt.show()

        bg_imgs[cam] = bg_img
    return bg_imgs

def get_tranform_mtx(r, t):
    T = np.zeros([4, 4])
    T[0:3, 0:3] = r
    T[0:3, 3] = np.squeeze(t)
    T[3, 3] = 1
    return T

def substract_foreground(bg_img, img, min_depth=500, max_depth=6000):
    bg_img = bg_img.astype(np.float32)
    img = img.astype(np.float32)

    # substraction
    img_bin = (np.abs(bg_img - img) > 60) * (img > min_depth) * (img < max_depth)
    img_bin = scipy.ndimage.median_filter(img_bin, 5)

    # connected component (useless)
    num_labels, labels_im = cv2.connectedComponents(img_bin.astype(np.uint8))
    # label 0 is the background with the largest area, second largest is the foreground
    areas = [np.sum(labels_im==(i+1)) for i in range(num_labels-1)]
    max_label = areas.index(max(areas))
    img_bin = labels_im == (max_label + 1)

    # plt.figure()
    # plt.imshow(img_bin, cmap='gray')
    # plt.show()
    return img_bin

def get_homo_point(xyz):
    assert xyz.shape[1] == 3
    one_array = np.ones([xyz.shape[0], 1])
    xyz_homo = np.concatenate([xyz, one_array], axis=1)
    return xyz_homo        

def inverse_extrinsic_params(R_12, t_12):
    """
    Compute the inverse of the extrinsic parameters.
    R_12: rotation matrix from camera 1 to camera 2
    t_12: translation vector from camera 1 to camera 2
    Returns: R_21 and t_21
    """
    # Compute the inverse rotation matrix
    R_21 = np.transpose(R_12)
    
    # Compute the inverse translation vector
    t_21 = -np.matmul(R_21, t_12)
    
    return R_21, t_21

# def twodim_2_threedim(y, x, uvd, pts, intr_params):
#     # Calculate the overall distance
#     distances = np.sqrt((uvd[:, 0] - x) ** 2 + (uvd[:, 1] - y) ** 2)
#     closest_index = np.argmin(distances)
#     closest_coordinates = uvd[closest_index]
#     print(f"for given {x} & {y}, closest coordinate as {closest_coordinates}")
#     closest_coordinates = closest_coordinates.reshape((1,3))
#     # print(f"\nshape is {closest_coordinates.shape}\n")
#     xyz = unprojection(closest_coordinates, intr_params)
#     xyz_ = get_homo_point(xyz)[0]
#     return xyz_

def transform_to_cam1(xyz, extr_params, cam):
    xyz = get_homo_point(xyz)
    # print(f"\n shape of xyz is {xyz.shape}\n")

    if cam == 'azure_kinect1_2':
        z = pts[:, 0:3]
        # print(f"\n pts shape afterwards is {z.shape}\n")
        # print(f"\n pts afterwards is {z}\n")
        return z
    
    if cam == 'azure_kinect1_3':
        r,t = extr_params['%s-%s' % ('azure_kinect1_2', cam)][0], extr_params['%s-%s' % ('azure_kinect1_2', cam)][1]
        tranform_matrix = get_tranform_mtx(r,t)
        z = np.dot(tranform_matrix, xyz.T).T[:, 0:3]
        # print(f"\n machaya shape is {z.shape}\n")
        return z

    if cam == 'azure_kinect2_4':
        # cam3 to cam2
        r,t = extr_params['%s-%s' % ('azure_kinect1_3', cam)][0], extr_params['%s-%s' % ('azure_kinect1_3', cam)][1]
        tranform_matrix = get_tranform_mtx(r,t)
        # pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]
        pts = np.dot(tranform_matrix, xyz.T).T

        # cam2 to cam1
        r,t = extr_params['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][0], extr_params['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][1]
        tranform_matrix = get_tranform_mtx(r,t)
        z = np.dot(tranform_matrix, pts.T).T[:, 0:3]
        # print(f"\n machaya shape is {z.shape}\n")
        return z
    
    if cam == 'azure_kinect2_5':
        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect2_5', 'azure_kinect3_3')][0], extr_params['%s-%s' % ('azure_kinect2_5', 'azure_kinect3_3')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, xyz.T).T

        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T

        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        z = np.dot(tranform_matrix, pts.T).T[:, 0:3]

        # print(f"\n machaya shape is {z.shape}\n")
        return z

    if cam == 'azure_kinect3_3':
        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, xyz.T).T

        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        z = np.dot(tranform_matrix, pts.T).T[:, 0:3]

        # print(f"\n machaya shape is {z.shape}\n")
        return z

    if cam == 'azure_kinect3_2':
        r,t = inverse_extrinsic_params(extr_params['%s-%s' % (cam, 'azure_kinect1_2')][0], extr_params['%s-%s' % (cam, 'azure_kinect1_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        z = np.dot(tranform_matrix, xyz.T).T[:, 0:3]
        # print(f"\n machaya shape is {z.shape}\n")
        return z

if __name__ == '__main__':

    data_dir = './AzureKinectRecord_30_05'
    group_list = ['1', '2', '3']
    cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']
    cam1 = cam_list[0]

    # # Load the color image
    # image_path = '%s/%s/%s/color/color0000.jpg' % (data_dir, group_name, cam_list[0])
    # image = plt.imread(image_path)

    bg_data_dir = './AzureKinectRecord_30_05'
    bg_img_idx = 175  # 121
    # load background image
    bg_imgs = load_background(bg_data_dir, cam_list, bg_img_idx)

    # load intrinsic parameters
    with open('%s/intrinsic_param.pkl' % data_dir, 'rb') as f:
        intr_params = pickle.load(f)
        print(intr_params.keys())

    # load pairwise extrinsic parameters
    with open('%s/extrinsic_param.pkl' % data_dir, 'rb') as f:
        extr_params = pickle.load(f)
        print(extr_params.keys())

    # load kinect extrinsic parameters
    with open('%s/kinect_extrinsic_param.pkl' % data_dir, 'rb') as f:
        kinect_extr_params = pickle.load(f)
        print(kinect_extr_params.keys())

    # Load the keypoints data
    for group_name in group_list:

        print(f"\n Current group is {group_name} \n")
        num_frames = len(glob.glob("%s/%s/azure_kinect1_2/color/color*.jpg" % (data_dir, group_name)))
        print('number of images %i' % num_frames)

        if not os.path.exists('%s/%s/point_cloud' % (data_dir, group_name)):
            print('make direction: %s/%s/point_cloud' % (data_dir, group_name))
            os.mkdir('%s/%s/point_cloud' % (data_dir, group_name))

        if group_name == '3':
            start_index = 89
        else:
            start_index = 0
        
        for cam in cam_list:

            # Depth to color registration matrix
            R, T = kinect_extr_params[f"{cam}_d2c"]
            tranform_matrix_d2c = get_tranform_mtx(R,T)

            R_inv, T_inv = inverse_extrinsic_params(R, T)
            tranform_matrix_d2c_inv = get_tranform_mtx(R_inv, T_inv)

            for frame_idx in range(num_frames):

                _frame_idx = start_index + frame_idx
                print(f"\n Current frame is {_frame_idx} \n")
                
                save_name = '%s/%s/%s/pose_json/color%04i_keypoints.json' % (data_dir, group_name, cam, _frame_idx)
                try:
                    with open(save_name, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {save_name}")
                    raise

                data_people = data['people']
                # print(data_people)

                dname = '%s/%s/%s/depth/depth%04i.png' % (data_dir, group_name, cam, _frame_idx)
                cname = '%s/%s/%s/color/color%04i.jpg' % (data_dir, group_name, cam, _frame_idx)
                
                if not os.path.exists(dname):
                    continue

                color_img = cv2.imread(cname)
                depth_img = cv2.imread(dname, -1).astype(np.float32)

                if np.mean(depth_img) < 1:
                    print('[invalid] %s' % dname)
                    continue

                if depth_img is None:
                    print("Error reading depth image.")
                    continue

                if color_img is None:
                    print("Error reading color image.")
                    continue

                img_mask = substract_foreground(bg_imgs[cam], depth_img)
                img = depth_img * img_mask
                # plt.imshow(img)
                # plt.show()
                # print("img mask is \n", img_mask)
                # new_img = color_img * img_mask[:, :, np.newaxis]
                # plt.imshow(new_img)
                # plt.show()

                # project to 3D space
                uvd = depth2uvd(img).reshape([-1, 3])
                xyz = unprojection(uvd, intr_params['%s_depth' % cam])
                xyz = get_homo_point(np.asarray(xyz))
                pts = np.dot(tranform_matrix_d2c, xyz.T).T[:, 0:3]
                uvd_c = projection(pts, intr_params['%s_color' % cam], simple_mode=False)
                # print("\n------------------uvd_c-------------------\n")
                # print(f"{uvd_c}")
                # print("\n------------------uvd_c-------------------\n")

                # print(f"\n number of people are {len(data_people)}\n")
                # Iterate over each person's keypoints
                for person_index, person in enumerate(data_people):

                    # Initialize an empty list for keypoints_3d
                    keypoints_3d = []
                    # print("\n---------------------------------------\n")
                    if 'pose_keypoints_2d' in person:
                        keypoints = person['pose_keypoints_2d']
                        keypoints = np.array(keypoints).reshape(-1, 3)
                        # print(f"\n keypoint is {keypoints} \n")
                        # print(f"\n len(keypoints) is {len(keypoints)} \n")
                        # print(keypoints)

                        # first we check if these are to be taaken into consideration or not
                        check = 0
                        for i in range(0, len(keypoints)):
                            # print("\n------------------keypoint-------------------\n")
                            confidence = keypoints[i][2]

                            # Find the point closest to keypoints[i][0],keypoints[i][1] in uvd_c
                            distances = np.sum((uvd_c[:, :2] - np.array([keypoints[i][0], keypoints[i][1]])) ** 2, axis=1)
                            closest_index = np.argmin(distances)
                            closest_point_c = uvd_c[closest_index]
                            closest_point_c = closest_point_c.reshape((-1,3))

                            # print(f"\nThe closest point to ({keypoints[i][0]},{keypoints[i][1]}) in rgb camera frame coordinates {closest_point_c}\n")
                            # print(f"shape of closest_point_c is {closest_point_c.shape}\n")
                            pts = unprojection(closest_point_c, intr_params['%s_color' % cam], simple_mode=False)
                            # print(f"\nThe corresponding point in xyz to {closest_point_c} is {pts}\n")
                            xyz = np.dot(tranform_matrix_d2c_inv, np.append(pts, 1))[:3]  # assuming transform_matrix_d2c_inv is available
                            # print(f"\nThe corresponding point in xyz to {closest_point_c} is {xyz}\n")
                            # print(f"shape of xyz is {xyz.shape}\n")
                            xyz = xyz.reshape((-1,3))
                            xyz = transform_to_cam1(xyz, extr_params, cam)

                            closest_point_d = projection(xyz, intr_params['%s_depth' % cam1])  # assuming reverse_unprojection is available
                            # print(f"\nThe closest point to ({keypoints[i][0]},{keypoints[i][1]}) in depth camera frame coordinatesis {closest_point_d[0]}\n")
                            # print(f"shape of closest_point_d[0] is {closest_point_d[0].shape}\n")
                            # Print or use the closest_point here
                            # print(f"The corresponding point in uvd to {closest_point_c} is {closest_point}")

                            # Get the depth at that pixel
                            # depth_at_uv = depth_img[v, u]
                            # uvd = np.array([u, v, depth_at_uv]).reshape((1,3))
                            # xyz = unprojection(uvd, intr_params['%s_depth' % cam])
                            # pts = get_homo_point(np.asarray(xyz))[0]

                            # print(f"color image cord are {x},{y} \n")
                            # print(f"depth image cord are {u},{v} and depth {depth_at_uv}\n")
                            # print(f"homogeneous cord are {pts}   \n")
                            a1 = int(closest_point_d[0][0])
                            b1 = int(closest_point_d[0][1])
                            num_points = count_points_within_radius(a1, b1, img)

                            # print(f"depth non zero count in vicinity of point is {num_points}")
                            if num_points > 20:
                                check = 1

                            keypoints_3d.append(closest_point_d.tolist())
                            # print("\n---------------------------------------------\n")
                    
                    if check==1:
                        data_people[person_index]['pose_keypoints_3d'] = keypoints_3d

                    else:
                        keypoints_3d = []
                        data_people[person_index]['pose_keypoints_3d'] = keypoints_3d

                    # print(f"\n keypoints_3D is {keypoints_3d} \n")
                
                data['people'] = data_people
                # save the json file
                with open(save_name, 'w') as f:
                    json.dump(data, f)
                # exit()


# # Plot the image
# plt.imshow(image)

# # Iterate over each person's keypoints
# for person_index, person in enumerate(data):
#     if 'pose_keypoints_2d' in person:
#         keypoints = person['pose_keypoints_2d']
#         keypoints = np.array(keypoints).reshape(-1, 3)

#         # Generate a random color for each person
#         color = (random.random(), random.random(), random.random())

#         # Connect the keypoints with line segments using the random color
#         for connection in connections:
#             start_part = connection[0]
#             end_part = connection[1]
#             if start_part in body_parts and end_part in body_parts:
#                 start_index = body_parts[start_part]
#                 end_index = body_parts[end_part]
#                 if start_index < keypoints.shape[0] and end_index < keypoints.shape[0]:
#                     start = keypoints[start_index]
#                     end = keypoints[end_index]
#                     plt.plot([start[0], end[0]], [start[1], end[1]], color=color)

# # Show the plot
# plt.show()