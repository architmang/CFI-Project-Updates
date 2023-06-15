import sys
sys.path.append('../')
import pickle
import numpy as np
import matplotlib.pylab as plt
import scipy.ndimage
import cv2
from calibration.utils import *
import open3d as o3d
import os
import glob


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


def substract_foreground(bg_img, img, min_depth=500, max_depth=4000):
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


def get_tranform_mtx(r, t):
    T = np.zeros([4, 4])
    T[0:3, 0:3] = r
    T[0:3, 3] = np.squeeze(t)
    T[3, 3] = 1
    return T


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

if __name__ == '__main__':

    cam_list = ['azure_kinect1_2','azure_kinect1_3','azure_kinect2_4', 
            'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']
    
    bg_data_dir = './AzureKinectRecord_30_05'
    bg_img_idx = 175  # 121
    data_dir = './AzureKinectRecord_30_05'
    group_names = ['1', '2', '3']

    global intr_params, extr_params

    # load intrinsic parameters
    with open('%s/intrinsic_param.pkl' % data_dir, 'rb') as f:
        intr_params = pickle.load(f)
        print(intr_params.keys())

    # load intrinsic parameters
    with open('%s/extrinsic_param.pkl' % data_dir, 'rb') as f:
        extr_params = pickle.load(f)
        print(extr_params.keys())

    # load background image
    bg_imgs = load_background(bg_data_dir, cam_list, bg_img_idx)

    for group_name in group_names:

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

        for frame_idx in range(num_frames):
        # for frame_idx in range(579, 1268, 1):
            save_name = "%s/%s/point_cloud/pose%04i.ply" % (data_dir, group_name, start_index + frame_idx)
            
            if os.path.exists(save_name):
                # continue
                pass

            pts_list = []
            # ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3']
            for idx, cam in enumerate(cam_list):
                # TODO: debug
                if 'azure_kinect' in cam:
                    # _frame_idx = frame_idx + 5
                    _frame_idx = start_index + frame_idx
                else:
                    _frame_idx = start_index + frame_idx
                ######

                fname = '%s/%s/%s/depth/depth%04i.png' % (data_dir, group_name, cam, _frame_idx)
                if not os.path.exists(fname):
                    continue

                depth_img = cv2.imread(fname, -1).astype(np.float32)
                if np.mean(depth_img) < 1:
                    print('[invalid] %s' % fname)
                    continue

                img_mask = substract_foreground(bg_imgs[cam], depth_img)
                img = depth_img * img_mask

                # project to 3D space
                uvd = depth2uvd(img).reshape([-1, 3])
                uvd = uvd[uvd[:, 2] > 0, :]  # get the point whose depth is greater than 0
                uvd = uvd[uvd[:, 2] < 8000, :]
                xyz = unprojection(uvd, intr_params['%s_depth' % cam])

                pts = get_homo_point(np.asarray(xyz))
                print(f"\n pts shape is {pts.shape}\n")
                print(f"\n pts is {pts}\n")
                
                if cam == 'azure_kinect1_2':
                    pts_list.append(pts[:, 0:3])
                    print(f"\n pts shape afterwards is {pts.shape}\n")
                    print(f"\n pts afterwards is {pts}\n")
                    exit()

                
                if cam == 'azure_kinect1_3':
                    r,t = extr_params['%s-%s' % ('azure_kinect1_2', cam)][0], extr_params['%s-%s' % ('azure_kinect1_2', cam)][1]
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]
                    pts_list.append(pts)
                    print(f"\n machaya shape is {pts.shape}\n")


                if cam == 'azure_kinect2_4':
                    # cam3 to cam2
                    r,t = extr_params['%s-%s' % ('azure_kinect1_3', cam)][0], extr_params['%s-%s' % ('azure_kinect1_3', cam)][1]
                    tranform_matrix = get_tranform_mtx(r,t)
                    # pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]
                    pts = np.dot(tranform_matrix, pts.T).T

                    # cam2 to cam1
                    r,t = extr_params['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][0], extr_params['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][1]
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]
                    pts_list.append(pts)
                    print(f"\n machaya shape is {pts.shape}\n")


                if cam == 'azure_kinect2_5':
                    r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect2_5', 'azure_kinect3_3')][0], extr_params['%s-%s' % ('azure_kinect2_5', 'azure_kinect3_3')][1])
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T

                    r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1])
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T

                    r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1])
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]

                    pts_list.append(pts)
                    print(f"\n machaya shape is {pts.shape}\n")


                if cam == 'azure_kinect3_3':

                    r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1])
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T

                    r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1])
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]

                    pts_list.append(pts)
                    print(f"\n machaya shape is {pts.shape}\n")


                if cam == 'azure_kinect3_2':
                    r,t = inverse_extrinsic_params(extr_params['%s-%s' % (cam, 'azure_kinect1_2')][0], extr_params['%s-%s' % (cam, 'azure_kinect1_2')][1])
                    tranform_matrix = get_tranform_mtx(r,t)
                    pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]
                    pts_list.append(pts)
                    print(f"\n machaya shape is {pts.shape}\n")

            # get rid of redundant points
            pts_prior = np.concatenate(pts_list, axis=0)
            pcd_prior = o3d.geometry.PointCloud()
            pcd_prior.points = o3d.utility.Vector3dVector(pts_prior)
            _, ind = pcd_prior.remove_radius_outlier(nb_points=30, radius=50)
            pcd_prior = pcd_prior.select_by_index(ind)

            # o3d.io.write_point_cloud(save_name, pcd_prior)
            # print('finish %04i' % frame_idx)

