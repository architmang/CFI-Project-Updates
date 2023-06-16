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
    # Compute the inverse rotation matrix
    R_21 = np.transpose(R_12)
    # Compute the inverse translation vector
    t_21 = -np.matmul(R_21, t_12)

    return R_21, t_21

# Capture the window contents as screenshots
def capture_screenshot(file_path):
    screenshot = pyautogui.screenshot()
    screenshot.save(file_path)

def avg_joints_distance(keypoints_3d_list, keypoints_3d_list_prev):
    assert keypoints_3d_list.shape == keypoints_3d_list_prev.shape, "Input arrays must have the same shape"
    
    distances = np.linalg.norm(keypoints_3d_list - keypoints_3d_list_prev, axis=1)
    avg_distance = np.mean(distances)
    return avg_distance

def id2color(person_index):
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return color_list[person_index]

data_dir = './AzureKinectRecord_30_05'
group_list = ['1', '2', '3']  
cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

# load intrinsic parameters
with open('%s/intrinsic_param.pkl' % data_dir, 'rb') as f:
    intr_params = pickle.load(f)

# load intrinsic parameters
with open('%s/extrinsic_param.pkl' % data_dir, 'rb') as f:
    extr_params = pickle.load(f)

group_name = '1'
cam = 'azure_kinect1_2'
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

dict = {}

for frame_idx in range(num_frames):

    _frame_idx = start_index + frame_idx

    save_name = '%s/%s/%s/pose_json/color%04i_keypoints.json' % (data_dir, group_name, cam, _frame_idx)
    with open(save_name, 'r') as f:
        data = json.load(f)

    data_people = data['people']

    # print(f"\n Current frame is {_frame_idx} \n")
    keypoints_3d_this_frame = []

    # print("number of people are: ", len(data_people))
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for person_index, person in enumerate(data_people):
        # print(f"\n---------------{person_index}------------\n")

        if 'pose_keypoints_3d' in person and len(person['pose_keypoints_3d']) > 0:
            keypoints_3d = person['pose_keypoints_3d']
            # print("\n-----------------------------\n")
            # print(keypoints_3d)
            keypoints_3d = [keypoint[0] for keypoint in keypoints_3d]
            # print("\n-----------------------------\n")
            keypoints_3d = keypoints_3d*np.array([1, -1, -1])
            # keypoints_3d = (np.array(keypoints_3d)).reshape(-1, 4)
            # print(keypoints_3d)
            # print("\n-----------------------------\n")
            keypoints_3d = np.array(keypoints_3d).reshape(-1, 3)
            keypoints_3d_this_frame.append((person_index, keypoints_3d))

    dict[_frame_idx] = keypoints_3d_this_frame

print("\n-----------------------------\n")
# print(dict)
# print("\n-----------------------------\n")

index_global = 0
num_frames = 20
for frame_idx in range(num_frames):

    print(f"\n Current frame is {frame_idx} \n")
    _frame_idx = start_index + frame_idx
    keypoints_3d_list = dict[_frame_idx]
    print("\n-----------------------------\n")
    print("number of people detected in this frame are: ", len(keypoints_3d_list))
    
    if frame_idx == 0:
        for index in range(len(keypoints_3d_list)):
            keypoints_3d_list[index][0] = index_global
            index_global += 1

    if frame_idx > 0:
        keypoints_3d_list_prev = dict[_frame_idx-1]
        print("\nnumber of people detected in previous frame are: ", len(keypoints_3d_list_prev))

        if len(keypoints_3d_list) == len(keypoints_3d_list_prev):
            print("\nno of people are same")
            # make a matrix of dimension (len(keypoints_3d_list), len(keypoints_3d_list_prev))
            # fill the matrix with the euclidean distance between the points of two frames
            # find the minimum distance between the points of two frames
            # the points with minimum distance are the same person
            # the points with maximum distance are the new person
            # the points with no distance are the person who left
            mat = np.zeros((len(keypoints_3d_list), len(keypoints_3d_list_prev)))
            for i in range(len(keypoints_3d_list)):
                for j in range(len(keypoints_3d_list_prev)):
                    mat[i][j] = avg_joints_distance(keypoints_3d_list[i][1], keypoints_3d_list_prev[j][1])
            
            print("\n-----------------------------\n")
            print("Matrix of distances:")
            print(mat)

            # Assign IDs to persons based on minimum distance
            assigned_ids = []
            for i in range(len(keypoints_3d_list)):
                min_distance_idx = np.argmin(mat[i])
                if min_distance_idx in assigned_ids:
                    # Person with this ID is already assigned to someone else
                    assigned_ids.append(i)
                    print("Person", i, "is a new person")
                    index_global
                else:
                    person = keypoints_3d_list_prev[min_distance_idx]
                    assigned_ids.append(min_distance_idx)
                    print("Person", i, "is the same as Person", min_distance_idx)
                    keypoints_3d_list[i][0] = person[0]

            print("Assigned IDs:", assigned_ids)

        if len(keypoints_3d_list) > len(keypoints_3d_list_prev):
            diff = len(keypoints_3d_list) - len(keypoints_3d_list_prev)
            print(f"\n{diff} new person detected")

        if len(keypoints_3d_list) < len(keypoints_3d_list_prev):
            diff = len(keypoints_3d_list_prev) - len(keypoints_3d_list)
            print(f"\n{diff} person left")

    print("\n-----------------------------\n")

for frame_idx in range(num_frames):
    keypoints_3d_list = dict[_frame_idx]
    num_people = len(keypoints_3d_list)
    
    for person_index, keypoints_3d in keypoints_3d_list:
        spheres = []
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
            sphere.paint_uniform_color(id2color(person_index))
            spheres.append(sphere)
            visualizer.add_geometry(sphere)

        # visualizer.update_geometry(point_cloud)
        for sphere in spheres:
            visualizer.update_geometry(sphere)
        visualizer.poll_events()
        visualizer.update_renderer()
    
    time.sleep(3)
    visualizer.remove_geometry(line_set)

visualizer.destroy_window()