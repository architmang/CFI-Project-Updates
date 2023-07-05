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

if __name__ == '__main__':

    data_dir = './AzureKinectRecord_30_05'
    group_list = ['1', '2', '3']
    cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

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

                # Plot the image
                plt.imshow(color_img)

                for person_index, person in enumerate(data_people):

                    if 'pose_keypoints_2d' in person:
                        keypoints = person['pose_keypoints_2d']
                        keypoints = np.array(keypoints).reshape(-1, 3)

                        # Generate a random color for each person
                        color = (random.random(), random.random(), random.random())

                        # Plot the keypoints using the random color
                        for i in range(keypoints.shape[0]):
                            if i in imp_body_parts.values():
                                x, y, _ = keypoints[i]
                                plt.scatter(x, y, color=color)

                        # Calculate vertical distance between topmost and bottommost keypoints
                        distance = np.abs(keypoints[1][1] - keypoints[8][1])

                        text = f"{distance:0.3f}"
                        # torso to ancle distance

                        if text != '0.000':
                            plt.text(keypoints[1][0], keypoints[1][1], text, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

                # Show the plot
                # plt.show()
                # Save the plot with the keypoints and distance
                save_path = f"{data_dir}/{group_name}/{cam}/bounding_boxes/{_frame_idx}.jpg"
                plt.savefig(save_path)
                plt.close()
                # print(f"Saved image: {save_path}")