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
import subprocess
import pyautogui
import open3d as o3d
import time
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the mapping of body parts to their corresponding indices
body_parts = {
    'Nose': 0,
    'Neck': 1,
    'RShoulder': 2,
    'RElbow': 3,
    'RWrist': 4,
    'LShoulder': 5,
    'LElbow': 6,
    'LWrist': 7,
    'MidHip': 8,
    'RHip': 9,
    'RKnee': 10,
    'RAnkle': 11,
    'LHip': 12,
    'LKnee': 13,
    'LAnkle': 14,
    'REye': 15,
    'LEye': 16,
    'REar': 17,
    'LEar': 18,
    'LBigToe': 19,
    'LSmallToe': 20,
    'LHeel': 21,
    'RBigToe': 22,
    'RSmallToe': 23,
    'RHeel': 24
}

# imp body points
imp_body_parts = {
    'Nose': 0,
    'Neck': 1,
    'RShoulder': 2,
    'RElbow': 3,
    'LShoulder': 5,
    'LElbow': 6,
    'MidHip': 8,
    'RHip': 9,
    'RKnee': 10,
    'RAnkle': 11,
    'LHip': 12,
    'LKnee': 13,
    'LAnkle': 14,
}


# Define the connections between keypoints for visualization
connections = [
    ('Neck', 'Nose'),
    ('Neck', 'RShoulder'),
    ('Neck', 'LShoulder'),
    ('RShoulder', 'RElbow'),
    ('RElbow', 'RWrist'),
    ('LShoulder', 'LElbow'),
    ('LElbow', 'LWrist'),
    ('Neck', 'MidHip'),
    ('MidHip', 'RHip'),
    ('RHip', 'RKnee'),
    ('RKnee', 'RAnkle'),
    ('MidHip', 'LHip'),
    ('LHip', 'LKnee'),
    ('LKnee', 'LAnkle'),
]

def get_tranform_mtx(r, t):
    T = np.zeros([4, 4])
    T[0:3, 0:3] = r
    T[0:3, 3] = np.squeeze(t)
    T[3, 3] = 1
    return T

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

# Capture the window contents as screenshots
def capture_screenshot(file_path):
    screenshot = pyautogui.screenshot()
    screenshot.save(file_path)

def convert_to_cam1(keypoint, intr_params, extr_params, cam):
    
    pts = np.append(keypoint, 1)
    # pts = np.array(keypoint).reshape(-1, 3)
    # print(f"\npoints before transformation shape: {pts.shape}")
    # print(f"\npoints is: {pts}")

    if cam == 'azure_kinect1_2':
        
        pts = pts[:-1]
        print(f"\npoints after transformation shape: {pts.shape}")
        return pts
        
    
    if cam == 'azure_kinect1_3':
        r,t = extr_params['%s-%s' % ('azure_kinect1_2', cam)][0], extr_params['%s-%s' % ('azure_kinect1_2', cam)][1]
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T
        pts = pts[:-1]

        print(f"\npoints after transformation shape: {pts.shape}")
        return pts
        

    if cam == 'azure_kinect2_4':
        # cam3 to cam2
        r,t = extr_params['%s-%s' % ('azure_kinect1_3', cam)][0], extr_params['%s-%s' % ('azure_kinect1_3', cam)][1]
        tranform_matrix = get_tranform_mtx(r,t)
        # pts = np.dot(tranform_matrix, pts.T).T[:, 0:3]
        pts = np.dot(tranform_matrix, pts.T).T

        # cam2 to cam1
        r,t = extr_params['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][0], extr_params['%s-%s' % ('azure_kinect1_2', 'azure_kinect1_3')][1]
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T
        
        # print(f"\npoints after transformation shape: {pts.shape}")
        return pts
        

    if cam == 'azure_kinect2_5':
        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect2_5', 'azure_kinect3_3')][0], extr_params['%s-%s' % ('azure_kinect2_5', 'azure_kinect3_3')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T

        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T

        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T
        
        # print(f"\npoints after transformation shape: {pts.shape}")
        return pts
        

    if cam == 'azure_kinect3_3':

        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][0], extr_params['%s-%s' % ('azure_kinect3_3', 'azure_kinect3_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T

        r,t = inverse_extrinsic_params(extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][0], extr_params['%s-%s' % ('azure_kinect3_2', 'azure_kinect1_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T
        
        # print(f"\npoints after transformation shape: {pts.shape}")
        return pts
        

    if cam == 'azure_kinect3_2':
        r,t = inverse_extrinsic_params(extr_params['%s-%s' % (cam, 'azure_kinect1_2')][0], extr_params['%s-%s' % (cam, 'azure_kinect1_2')][1])
        tranform_matrix = get_tranform_mtx(r,t)
        pts = np.dot(tranform_matrix, pts.T).T

        # print(f"\npoints after transformation shape: {pts.shape}")
        return pts


data_dir = './AzureKinectRecord_30_05'
group_list = ['1', '2', '3']  
cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

# load intrinsic parameters
with open('%s/intrinsic_param.pkl' % data_dir, 'rb') as f:
    intr_params = pickle.load(f)
    print(intr_params.keys())

# load intrinsic parameters
with open('%s/extrinsic_param.pkl' % data_dir, 'rb') as f:
    extr_params = pickle.load(f)
    print(extr_params.keys())

group_name = '1'
print(f"\n Current group is {group_name} \n")

# Load and visualize point clouds for all frames at time t
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=1920, height=1080)

num_frames = len(glob.glob("%s/%s/azure_kinect1_2/color/color*.jpg" % (data_dir, group_name)))
print('number of images %i' % num_frames)

if not os.path.exists('%s/%s/point_cloud' % (data_dir, group_name)):
    print('make direction: %s/%s/point_cloud' % (data_dir, group_name))
    os.mkdir('%s/%s/point_cloud' % (data_dir, group_name))

if group_name == '3':
    start_index = 89
else:
    start_index = 0

frame_idx = 0
_frame_idx = start_index + frame_idx

# cam = 'azure_kinect1_2'
dict - {}
for cam in cam_list:

    print(f"\n Current cam is {cam} \n")

    save_name = '%s/%s/%s/pose_json/color%04i_keypoints.json' % (data_dir, group_name, cam, _frame_idx)
    with open(save_name, 'r') as f:
        data = json.load(f)

    data_people = data['people']
    # print(data_people)

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

    # Generate the file path for the current frame
    fname = '%s/%s/depth_point_cloud/pose%04i.ply' % (data_dir, group_name, _frame_idx)
    # Load the point cloud from file
    point_cloud = o3d.io.read_point_cloud(fname)

    # Flip the point cloud along the Z-axis
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) * np.array([1, -1, -1]))

    # get rid of redundant points
    _, ind = point_cloud.remove_radius_outlier(nb_points=30, radius=50)
    point_cloud = point_cloud.select_by_index(ind)

    # visualizer.add_geometry(point_cloud)
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    print("number of people are: ", len(data_people))
    for person_index, person in enumerate(data_people):
        print(f"\n---------------{person_index}------------\n")
        keypoints_3d = []
        spheres = []
        color = color_list[person_index] # different color for each line

        if 'pose_keypoints_3d' in person and len(person['pose_keypoints_3d']) > 0:
            keypoints_3d = person['pose_keypoints_3d']
            print("\n-----------------------------\n")
            # print(f"keypoints_3d element shape is {keypoints_3d[0].shape}\n")
            # print(f"keypoints_3d element is {keypoints_3d[0]}")
            print(keypoints_3d)
            keypoints_3d = [keypoint[0] for keypoint in keypoints_3d]
            print("\n-----------------------------\n")
            keypoints_3d = keypoints_3d*np.array([1, -1, -1])
            # keypoints_3d = (np.array(keypoints_3d)).reshape(-1, 4)
            print(keypoints_3d)
            print("\n-----------------------------\n")
            keypoints_3d = np.array(keypoints_3d).reshape(-1, 3)
            # keypoints_3d = keypoints_3d*np.array([1, -1, -1, 1])
            colors = [[1, 0, 0] for _ in range(len(connections))] # red color for each line
            lines = []
            for connection in connections:
                start_part = connection[0]
                start_index = body_parts[start_part]

                end_part = connection[1]
                end_index = body_parts[end_part]

                lines.append([start_index, end_index]) # Open3D uses indices for lines
            # Create line set
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(keypoints_3d[i] for i in range(len(keypoints_3d)))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            # Then you can add the LineSet to the visualizer:
            visualizer.add_geometry(line_set)

            for keypoint in keypoints_3d[:15]:
                # Create a sphere geometry for the keypoint
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
                # translate the sphere to the keypoint
                keypoint = np.squeeze(keypoint[0:3])  # use np.squeeze to ensure keypoint is a 1D array
                sphere.translate(keypoint)
                sphere.paint_uniform_color(color)
                spheres.append(sphere)
                visualizer.add_geometry(sphere)

            # visualizer.update_geometry(point_cloud)
            for sphere in spheres:
                visualizer.update_geometry(sphere)
            visualizer.poll_events()
            visualizer.update_renderer()
            time.sleep(10)

    # delay for 10 seconds
    # visualizer.poll_events()
    # visualizer.update_renderer()
    time.sleep(15)

    # visualizer.remove_geometry(point_cloud)
    visualizer.remove_geometry(line_set)
    visualizer.destroy_window()
