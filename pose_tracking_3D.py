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
    
    if person_index<3:
        return color_list[person_index]
    
    return color_list[-1]

def text_3d(text, pos, direction=None, degree=0.0, font='Arial.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

# # Create a custom callback function for adding labels
# def add_labels(vis):
#     # Clear existing labels
#     vis.clear_geometries("labels")

#     # Iterate over the spheres and add labels
#     for idx, sphere in enumerate(spheres):
#         center = sphere.get_center()

#         # Create a Label3D object
#         label = o3d.visualization.rendering.Label3D()
#         label.set_text(str(idx))
#         label.set_position(center)
#         label.set_color(np.random.uniform(0, 1, size=3))
#         label.set_font_size(np.random.uniform(10, 20))

#         # Add the label to the visualizer
#         vis.add_geometry(label, "labels")

#     return False

video_frames = []
data_dir = './AzureKinectRecord_30_05'
group_list = ['1', '2', '3']  
cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

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

dict = {}

for frame_idx in range(num_frames):
    
    _frame_idx = start_index + frame_idx
    keypoints_3d_this_frame = []

    for cam in cam_list:
        cnt=0
        save_name = '%s/%s/%s/pose_json/color%04i_keypoints.json' % (data_dir, group_name, cam, _frame_idx)
        with open(save_name, 'r') as f:
            data = json.load(f)
        data_people = data['people']

        for person_index, person in enumerate(data_people):
            # print(f"\n---------------{person_index}------------\n")

            if 'pose_keypoints_3d' in person and len(person['pose_keypoints_2d']) > 0 and len(person['pose_keypoints_3d']) > 0:
                keypoints_3d = person['pose_keypoints_3d']
                keypoints_3d = [keypoint[0] for keypoint in keypoints_3d]
                keypoints_3d = keypoints_3d*np.array([1, -1, -1])
                keypoints_3d = np.array(keypoints_3d).reshape(-1, 3)
                keypoints_3d_this_frame.append((person_index, keypoints_3d))
                cnt+=1
        print("\n This frame has ", cnt, " people detected in camera ", cam)    
    
    dict[_frame_idx] = keypoints_3d_this_frame
    print("\n frame number is ", _frame_idx, "total number of 3d keypoints are ", len(keypoints_3d_this_frame))
    print( "\n-----------------------------\n")
    # print(keypoints_3d_this_frame)

    # supppose 8 persons are detected in this frame across all cameras
    # form a matrix of dimension (8, 8)
    # fill the matrix with the euclidean distance between the points of two persons
    # find the minimum distance between the points of two persons
    # the points with minimum distance are the same person
    # keep doing this until 2 persons are left
    # Perform person matching
    # while len(keypoints_3d_this_frame) > 2:

    #     num_persons = len(keypoints_3d_this_frame)
    #     distances = np.zeros((num_persons, num_persons))

    #     # Calculate pairwise Euclidean distances between keypoints of each person
    #     for i in range(num_persons):
    #         # print("\n")
    #         for j in range(num_persons):
    #             keypoints_3d_a = keypoints_3d_this_frame[i][1]
    #             keypoints_3d_b = keypoints_3d_this_frame[j][1]
    #             keypoints_2d_a = keypoints_3d_a[:, :2]  # Extract only x and y coordinates
    #             keypoints_2d_b = keypoints_3d_b[:, :2]  # Extract only x and y coordinates
    #             distances[i, j] = avg_joints_distance(keypoints_2d_a, keypoints_2d_b)
    #             # dist = distance.cdist(keypoints_2d_a, keypoints_2d_b, metric='euclidean')
    #             # distances[i, j] = np.min(dist)
    #             # print(distances[i, j], end=" ")

    #     min_distance = float('inf')
    #     min_distance_indices = None

    #     # Iterate over each element in the matrix
    #     for i in range(num_persons):
    #         for j in range(num_persons):
    #             if i != j and distances[i, j] != 0:  # Exclude diagonal elements and check for non-zero values
    #                 if distances[i, j] < min_distance:
    #                     min_distance = distances[i, j]
    #                     min_distance_indices = (i, j)

    #     # Print the pair with the minimum non-zero distance and its indices
    #     # print("Pair with minimum non-zero distance:", min_distance_indices)
    #     # print("Minimum non-zero distance:", min_distance)

    #     # Find the pair of persons with the minimum distance
    #     old_persons = (keypoints_3d_this_frame[min_distance_indices[0]] , keypoints_3d_this_frame[min_distance_indices[1]])
    #     new_person = (old_persons[0][0],(keypoints_3d_this_frame[min_distance_indices[0]][1] + keypoints_3d_this_frame[min_distance_indices[1]][1])/2)

    #     # Print the pair of persons with the minimum distance
    #     # print("Persons with minimum distance:", old_persons)
    #     # print("new Person:", new_person)

    #     # Create a new list without the matched persons
    #     keypoints_3d_new = []
    #     for i in range(num_persons):
    #         if i not in min_distance_indices:
    #             keypoints_3d_new.append(keypoints_3d_this_frame[i])
    #     keypoints_3d_this_frame = keypoints_3d_new
    #     keypoints_3d_this_frame.append(new_person)
    #     # print("Number of persons left:", len(keypoints_3d_this_frame))
    #     # print(keypoints_3d_this_frame)

    # # mat = np.zeros((len(keypoints_3d_this_frame), len(keypoints_3d_this_frame)))
    # # for i in range(len(keypoints_3d_this_frame)):
    # #     print("\n")
    # #     for j in range(len(keypoints_3d_this_frame)):
    # #         mat[i][j] = avg_joints_distance(keypoints_3d_this_frame[i][1], keypoints_3d_this_frame[j][1])
    # #         print(mat[i][j], end=" ")
    
    dict[_frame_idx] = keypoints_3d_this_frame

exit()    
index_global = 0
for frame_idx in range(num_frames):

    print(f"\n Current frame is {frame_idx} \n")
    _frame_idx = start_index + frame_idx
    keypoints_3d_list = dict[_frame_idx]
    print("\n-----------------------------\n")
    print("number of people detected in this frame are: ", len(keypoints_3d_list))
    
    if frame_idx == 0:
        for index in range(len(keypoints_3d_list)):
            keypoints_3d_list[index] = list(keypoints_3d_list[index])
            keypoints_3d_list[index][0] = index_global
            keypoints_3d_list[index] = tuple(keypoints_3d_list[index])
            index_global += 1

    if frame_idx > 0:
        keypoints_3d_list_prev = dict[_frame_idx-1]
        print("\nnumber of people detected in previous frame are: ", len(keypoints_3d_list_prev))

        # make a matrix of dimension (len(keypoints_3d_list), len(keypoints_3d_list_prev))
        # fill the matrix with the euclidean distance between the points of two frames

        mat = np.zeros((len(keypoints_3d_list), len(keypoints_3d_list_prev)))
        for i in range(len(keypoints_3d_list)):
            for j in range(len(keypoints_3d_list_prev)):
                mat[i][j] = avg_joints_distance(keypoints_3d_list[i][1], keypoints_3d_list_prev[j][1])
        
        print("\n-----------------------------\n")
        print("Matrix of distances:")
        print(mat)

        if len(keypoints_3d_list) == len(keypoints_3d_list_prev):
            print("\nno of people are same")

            # find the minimum distance between the points of two frames
            # the points with minimum distance are the same person
            # the points with maximum distance are the new person
            # the points with no distance are the person who left the frame
            # Assign IDs to persons based on minimum distance

            assigned_ids = []
            assigned_old_indices = set()  # Keep track of old person indices that have been assigned to new persons

            for j in range(len(keypoints_3d_list_prev)):
                
                min_distance_idx = np.argmin(mat[:, j])
                if min_distance_idx in assigned_ids:
                    # Person with this ID is already assigned to someone else
                    continue
                
                else:
                    keypoints_3d_list[min_distance_idx] = list(keypoints_3d_list[min_distance_idx])
                    keypoints_3d_list[min_distance_idx][0] = keypoints_3d_list_prev[j][0]
                    keypoints_3d_list[min_distance_idx] = tuple(keypoints_3d_list[min_distance_idx])
                    assigned_ids.append(min_distance_idx)
                    assigned_old_indices.add(min_distance_idx)
                    print("Person", min_distance_idx, "is the same as new Person", j)

            for i in range(len(keypoints_3d_list)):
                
                if i not in assigned_ids:
                    # Assign a new ID to the new person
                    assigned_ids.append(i)
                    print("Person", i, "is a new person")
                    keypoints_3d_list[i] = list(keypoints_3d_list[i])
                    keypoints_3d_list[i][0] = index_global
                    keypoints_3d_list[i] = tuple(keypoints_3d_list[i])
                    index_global += 1

            print("Assigned IDs:", assigned_ids)

        if len(keypoints_3d_list) > len(keypoints_3d_list_prev):

            # Assign IDs to new persons based on minimum distance
            assigned_ids = []
            assigned_old_indices = set()  # Keep track of old person indices that have been assigned to new persons

            for j in range(len(keypoints_3d_list_prev)):
                min_distance_idx = np.argmin(mat[:, j])
                if min_distance_idx in assigned_ids:
                    # Person with this ID is already assigned to someone else
                    continue
                else:
                    keypoints_3d_list[min_distance_idx] = list(keypoints_3d_list[min_distance_idx])
                    keypoints_3d_list[min_distance_idx][0] = keypoints_3d_list_prev[j][0]
                    keypoints_3d_list[min_distance_idx] = tuple(keypoints_3d_list[min_distance_idx])
                    assigned_ids.append(min_distance_idx)
                    assigned_old_indices.add(min_distance_idx)
                    print("Person", min_distance_idx, "is the same as new Person", j)
                
            for i in range(len(keypoints_3d_list)):
                if i not in assigned_ids:
                    # Assign a new ID to the new person
                    assigned_ids.append(i)
                    print("Person", i, "is a new person")
                    keypoints_3d_list[i] = list(keypoints_3d_list[i])
                    keypoints_3d_list[i][0] = index_global
                    keypoints_3d_list[i] = tuple(keypoints_3d_list[i])
                    index_global += 1

            print("Assigned IDs:", assigned_ids)
            print("New person count:", len(keypoints_3d_list) - len(keypoints_3d_list_prev))

        if len(keypoints_3d_list) < len(keypoints_3d_list_prev):

            # Assign IDs to new persons based on minimum distance
            assigned_ids = []
            assigned_old_indices = set()  # Keep track of old person indices that have been assigned to new persons

            for i in range(len(keypoints_3d_list)):
                min_distance_idx = np.argmin(mat[i, :])
                if min_distance_idx in assigned_ids or min_distance_idx in assigned_old_indices:
                    # Person with this ID is already assigned to someone else
                    continue
                else:
                    person = keypoints_3d_list_prev[min_distance_idx]
                    assigned_ids.append(i)
                    assigned_old_indices.add(min_distance_idx)
                    print("Person", i, "is the same as Person", min_distance_idx)
                    keypoints_3d_list[i] = list(keypoints_3d_list[i])
                    keypoints_3d_list[i][0] = person[0]
                    keypoints_3d_list[i] = tuple(keypoints_3d_list[i])

            for i in range(len(keypoints_3d_list_prev)):
                if i not in assigned_old_indices:
                    print("Person", i, "has left")
                    index_global -= 1

            print("Assigned IDs:", assigned_ids)
            print("Person left count:", len(keypoints_3d_list_prev) - len(keypoints_3d_list))

    print("\n-----------------------------\n")

visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=1920, height=1080)

for frame_idx in range(num_frames):
    keypoints_3d_list = dict[frame_idx]
    num_people = len(keypoints_3d_list)
    
    universal_line_set_list = []
    universal_sphere_list = []
    universal_bbox_line_set_list = []
    universal_label_pcd_list = []

    for keypoints_3d_tuple in keypoints_3d_list:
        
        person_index = keypoints_3d_tuple[0]
        keypoints_3d = keypoints_3d_tuple[1]

        spheres = []
        colors = [id2color(person_index) for _ in range(len(connections))] # color according to index label for each line
        lines = []

        print("\n------------keypoints_3d-----------------\n")
        print(keypoints_3d)
        print("\n------------keypoints_3d-----------------\n")

        for connection in connections:
            start_part = connection[0]
            start_index = body_parts[start_part]

            end_part = connection[1]
            end_index = body_parts[end_part]

            lines.append([start_index, end_index]) # Open3D uses indices for lines

        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(keypoints_3d[:, :3])  # Use keypoints_3d[:, :3] to get the (x, y, z) coordinates
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        visualizer.add_geometry(line_set)
        universal_line_set_list.append(line_set)

        for index, keypoint in enumerate(keypoints_3d[:15]):
            if index == 0 or index == 11 or index == 14:
                continue
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
            keypoint = np.squeeze(keypoint[:3])  # use np.squeeze to ensure keypoint is a 1D array
            sphere.translate(keypoint)
            sphere.paint_uniform_color(id2color(person_index))
            spheres.append(sphere)
            visualizer.add_geometry(sphere)

        universal_sphere_list.append(spheres)
        # Calculate the minimum and maximum coordinates of all the sphere vertices
        all_vertices = np.concatenate([np.asarray(sphere.vertices) for sphere in spheres], axis=0)
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)

        # Create a custom bounding box using LineSet
        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]]

        bbox_line_set = o3d.geometry.LineSet()
        bbox_line_set.points = o3d.utility.Vector3dVector([[min_coords[0], min_coords[1], min_coords[2]],
                                                        [max_coords[0], min_coords[1], min_coords[2]],
                                                        [max_coords[0], min_coords[1], max_coords[2]],
                                                        [min_coords[0], min_coords[1], max_coords[2]],
                                                        [min_coords[0], max_coords[1], min_coords[2]],
                                                        [max_coords[0], max_coords[1], min_coords[2]],
                                                        [max_coords[0], max_coords[1], max_coords[2]],
                                                        [min_coords[0], max_coords[1], max_coords[2]]])
        bbox_line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
        bbox_line_set.colors = o3d.utility.Vector3dVector([id2color(person_index) for _ in range(len(bbox_lines))])  # Green color for bounding box
        visualizer.add_geometry(bbox_line_set)
        universal_bbox_line_set_list.append(bbox_line_set)

        # visualizer.register_animation_callback(add_labels)
        # Create a custom callback function for adding labels
        center = [ max_coords[0], max_coords[1], max_coords[2] ]
        # Create a Label3D object
        label_text = f"person {person_index}"
        label_pos = center
        label_direction = None
        label_degree = 0.0
        label_font = ".\Arial.ttf"  # Replace with the path to your font file
        label_font_size = 15

        label_pcd = text_3d(label_text, label_pos, label_direction, label_degree, label_font, label_font_size)
        universal_label_pcd_list.append(label_pcd)

        # Add the label point cloud to the visualizer
        visualizer.add_geometry(label_pcd)

    visualizer.poll_events()
    visualizer.update_renderer()
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    video_frames.append(frame)

    for line_set in universal_line_set_list:
        visualizer.remove_geometry(line_set)

    for spheres in universal_sphere_list:    
        for sphere in spheres:
            visualizer.remove_geometry(sphere)
        spheres.clear()

    for bbox_line_set in universal_bbox_line_set_list:
        visualizer.remove_geometry(bbox_line_set)

    for label_pcd in universal_label_pcd_list:
        visualizer.remove_geometry(label_pcd)

visualizer.destroy_window()

# Save the frames as a video using imageio
video_path = "object_tracking_3D_multi_view.mp4"
imageio.mimsave(video_path, video_frames, fps=2)