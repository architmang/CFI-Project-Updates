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
from open3d import geometry
from open3d import visualization
from open3d.visualization import gui
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
import pyautogui
import imageio
import scipy.spatial.distance as distance
from sklearn.cluster import DBSCAN, KMeans
import subprocess
import pyautogui
import os
import glob 
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
    # 'Nose': 0,
    'Neck': 1,
    'RShoulder': 2,
    'RElbow': 3,
    'LShoulder': 5,
    'LElbow': 6,
    'MidHip': 8,
    'RHip': 9,
    'RKnee': 10,
    # 'RAnkle': 11,
    'LHip': 12,
    'LKnee': 13,
    # 'LAnkle': 14,
}

# Define the connections between keypoints for visualization
connections = [
    # ('Neck', 'Nose'),
    ('Neck', 'RShoulder'),
    ('Neck', 'LShoulder'),
    ('RShoulder', 'RElbow'),
    ('RElbow', 'RWrist'),
    ('LShoulder', 'LElbow'),
    ('LElbow', 'LWrist'),
    ('Neck', 'MidHip'),
    ('MidHip', 'RHip'),
    ('RHip', 'RKnee'),
    # ('RKnee', 'RAnkle'),
    ('MidHip', 'LHip'),
    ('LHip', 'LKnee'),
    # ('LKnee', 'LAnkle'),
]

# Capture the window contents as screenshots
def capture_screenshot(file_path):
    screenshot = pyautogui.screenshot()
    screenshot.save(file_path)

def calculate_average_distance(point_cloud, human_points):
    point_cloud = np.array(point_cloud.points)
    human_points = np.array(human_points)
    
    distances = np.linalg.norm(point_cloud[:, np.newaxis, :] - human_points, axis=2)
    print("------------------")
    print(distances.shape)
    print(distances)
    print("------------------")

    average_distances = np.min(distances, axis=1)
    
    return average_distances

def id2color(person_index):
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # red, green, blue correct
    # color_list = [[1, 0, 0], [0, 1, 0], [0, 1, 0]]  # red, green, green incorrect

    if person_index<3:
        return color_list[person_index]
    
    return color_list[-1]

video_frames = []
data_dir = './AzureKinectRecord_30_05'
group_list = ['1', '2', '3']  
cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

# Specify the time t and duration to show each point cloud
time_t = 0.1  # Replace with the desired time t
duration_secs = 0.05  # Replace with the desired duration in seconds

# load intrinsic parameters
with open('%s/intrinsic_param.pkl' % data_dir, 'rb') as f:
    intr_params = pickle.load(f)

# load intrinsic parameters
with open('%s/extrinsic_param.pkl' % data_dir, 'rb') as f:
    extr_params = pickle.load(f)

group_name = '2'
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

with open('dict.pickle', 'rb') as handle:
    dict = pickle.load(handle)
print(dict[0])

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=1920, height=1080)

# Create a folder to store the rendered frames
frames_folder_human0 = f'{data_dir}/{group_name}/point_cloud_frames_human0'
frames_folder_human1 = f'{data_dir}/{group_name}/point_cloud_frames_human1'

for frame_idx in range(num_frames):

    _frame_idx = start_index + frame_idx

    # b = next(item[1] for item in x if item[0] == 1)
    human0 = next(item[1] for item in dict[_frame_idx] if item[0] == 0)
    human1 = next(item[1] for item in dict[_frame_idx] if item[0] == 1)

    print(f'human0: {human0}')
    print(f'human1: {human1}')

    save_name = "%s/%s/point_cloud/pose%04i.ply" % (data_dir, group_name, frame_idx)

    # Load the point cloud from file
    point_cloud = o3d.io.read_point_cloud(save_name)

    # Flip the point cloud along the Z-axis
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) * np.array([1, -1, -1]))
    all_points = np.asarray(point_cloud.points)
    # ------------------------------------------------------------------------OLD-------------------------------------------------------------------------------
    # Calculate the average distance of each point to the human and assign a label

    # distance_human0 = calculate_average_distance(point_cloud, human0)
    # distance_human1 = calculate_average_distance(point_cloud, human1)

    # labels = np.zeros(len(distance_human0))
    # labels[distance_human0 > distance_human1] = 1
    # print("--------labels----------")
    # print(labels)
    # values, counts = np.unique(labels, return_counts=True)
    # print(values, counts)
    # print("--------labels----------")

    # # keep only points which have label 0
    # points_human0 = o3d.utility.Vector3dVector(np.asarray(point_cloud.points)[labels == 0])
    # points_human1 = o3d.utility.Vector3dVector(np.asarray(point_cloud.points)[labels == 1])

    # ------------------------------------------------------------------------NEW--------------------------------------------------------------------------------
    
    # Set the DBSCAN parameters
    eps = 0.1  # The maximum distance between two samples to be considered as part of the same cluster
    min_samples = 5  # The minimum number of points in a neighborhood to form a core point

    # Create and fit the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(all_points)

    print(all_points.shape)
    print("labels")
    print(labels.shape)
    values, counts = np.unique(labels, return_counts=True)
    print(values, counts)
    # exit()

    # Extract the segmented points of each human
    human0_points = all_points[labels == 0]
    human1_points = all_points[labels == 1]

    point_cloud.points = human1_points # remove LATER

    # Customize the visualization settings
    visualizer.add_geometry(point_cloud)
    visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background color to black
    visualizer.get_render_option().point_size = 1.0  # Set point size
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))  # Add coordinate frame
    visualizer.update_geometry(point_cloud)
    visualizer.poll_events()
    visualizer.update_renderer()

    # Save the current frame as an image
    capture_screenshot("%s/%s/point_cloud_frames_human0/frame_%04d.png" % (data_dir, group_name, frame_idx))
    image_path = os.path.join(frames_folder_human0, f'frame_{frame_idx:04d}.png')
    # visualizer.capture_screen_image(image_path)
    video_frames.append(imageio.imread(image_path))

    # Wait for the specified duration
    time.sleep(duration_secs)

    # Clear the previous point cloud from the visualization
    visualizer.remove_geometry(point_cloud)

# Save the frames as a video using imageio
video_path = "%s/%s/point_cloud_human_1_DBSCAN.mp4"  % (data_dir, group_name)
imageio.mimsave(video_path, video_frames, fps=2)