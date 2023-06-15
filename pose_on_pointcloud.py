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
        
    
if __name__ == '__main__':

    data_dir = './AzureKinectRecord_30_05'
    group_list = ['1', '2', '3']
    cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3']

    # Specify the time t and duration to show each point cloud
    time_t = 0.1  # Replace with the desired time t
    duration_secs = 0.2  # Replace with the desired duration in seconds

    # Specify video output settings
    fps = 10  # Frames per second

    # load intrinsic parameters
    with open('%s/intrinsic_param.pkl' % data_dir, 'rb') as f:
        intr_params = pickle.load(f)
        print(intr_params.keys())

    # load intrinsic parameters
    with open('%s/extrinsic_param.pkl' % data_dir, 'rb') as f:
        extr_params = pickle.load(f)
        print(extr_params.keys())

    # Load the keypoints data
    for group_name in group_list:

        print(f"\n Current group is {group_name} \n")

        # Load and visualize point clouds for all frames at time t
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=1920, height=1080)

        # Create a folder to store the rendered frames
        frames_folder = f'{data_dir}/{group_name}'

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
            _frame_idx = start_index + frame_idx

            cam = 'azure_kinect1_2'
            print(f"\n Current cam is {cam} \n")
            save_name = '%s/%s/%s/pose_json/color%04i_keypoints.json' % (data_dir, group_name, cam, _frame_idx)
            with open(save_name, 'r') as f:
                data = json.load(f)

            data_people = data['people']
            print(data_people)

            # Generate the file path for the current frame
            fname = '%s/%s/point_cloud/pose%04i.ply' % (data_dir, group_name, _frame_idx)
            if not os.path.exists(fname):
                continue

            # Load the point cloud from file
            point_cloud = o3d.io.read_point_cloud(fname)

            # Flip the point cloud along the Z-axis
            point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) * np.array([1, -1, -1]))

            # # get rid of redundant points
            _, ind = point_cloud.remove_radius_outlier(nb_points=30, radius=50)
            point_cloud = point_cloud.select_by_index(ind)

            # Customize the visualization settings
            visualizer.add_geometry(point_cloud)
            visualizer.get_render_option().background_color = np.asarray([0, 0, 0])  # Set background color to black
            visualizer.get_render_option().point_size = 1.0  # Set point size
            visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))  # Add coordinate frame
            visualizer.update_geometry(point_cloud)
            visualizer.poll_events()
            visualizer.update_renderer()

            spheres = []
            # Define the RGB values for red
            color = np.array([1.0, 0.0, 0.0])  # [R, G, B]

            # Iterate over each person's keypoints
            for person_index, person in enumerate(data_people):
                
                keypoints_3d = []
                # Generate a random color for each person
                # color = (random.random(), random.random(), random.random())

                if 'pose_keypoints_3d' in person and len(person['pose_keypoints_3d']) > 0:
                    keypoints_3d = person['pose_keypoints_3d']
                    # keypoints_3d = np.array(keypoints_3d).reshape(-1, 3)
                    print(keypoints_3d)
                    
                    # Visualize the keypoints
                    for keypoint_index, keypoint in enumerate(keypoints_3d):
                        # print(f"\n keypoint before is {keypoint} \n")
                        # keypoint = convert_to_cam1(keypoint, intr_params, extr_params, cam)[0]
                        # keypoint*= np.array([1, -1, -1])
                        # print(f"\n converted keypoint before is {keypoint} \n")
                        # Create a sphere geometry for the keypoint
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=30)
                        
                        # Translate the sphere to the keypoint position
                        sphere.translate(keypoint)
                        
                        # Set the color of the sphere
                        sphere.paint_uniform_color(color)
                        # Add the sphere to the list
                        spheres.append(sphere)
                        # Add the sphere to the visualizer
                        visualizer.add_geometry(sphere)

                    # Create an empty line set
                    # line_set = o3d.geometry.LineSet()

                    # Set the points and lines for the line set
                    # line_set.points = o3d.utility.Vector3dVector(keypoints_3d)
                    # line_set.lines = o3d.utility.Vector2iVector([(body_parts[start], body_parts[end]) for start, end in connections if start in body_parts and end in body_parts])

                    # Set the line color
                    # line_set.colors = o3d.utility.Vector3dVector([(1.0, 0.0, 0.0)] * len(line_set.lines))

                    # Customize the visualization settings
                    # visualizer.add_geometry(line_set)
                    # visualizer.get_render_option().line_width = 5.0  # Set the line width

                    visualizer.poll_events()
                    visualizer.update_renderer()
            
            # Clear the previous point cloud and line set from the visualization
            visualizer.remove_geometry(point_cloud)

            # if line_set is not None:
                # visualizer.remove_geometry(line_set)

            # Update the geometries in the visualizer
            visualizer.poll_events()
            visualizer.update_renderer()

            # # Update the geometries in the visualizer
            # for sphere in spheres:
            #     visualizer.remove_geometry(sphere)

    # Destroy the visualization window
    visualizer.destroy_window()


            # Save the current frame as an image
            # capture_screenshot("%s/%s/point_cloud_frames_group/frame_%04d.png" % (data_dir, group_name, frame_idx))
            # time.sleep(duration_secs)