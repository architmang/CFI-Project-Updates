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
import open3d as o3d
import time
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from open3d import geometry
# from open3d import visualization
# from open3d.visualization import gui
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
# import pyautogui
import imageio

data_dir = './AzureKinectRecord_30_05'
group_list = ['1', '2', '3']  
cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

for group_name in group_list[1:2]:

    print(f"\n\t\t Current group is {group_name} \n")
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

    for cam in cam_list:

        print(f"\n\t\t Current cam is {cam} \n")
        for frame_idx in range(num_frames):

            _frame_idx = start_index + frame_idx
            save_name = '%s/%s/%s/pose_json/color%04i_keypoints.json' % (data_dir, group_name, cam, _frame_idx)
            with open(save_name, 'r') as f:
                data = json.load(f)

            data_people = data['people']
            keypoints_3d_this_frame = []

            color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

            cnt=0
            for person_index, person in enumerate(data_people):

                if 'pose_keypoints_3d' in person and len(person['pose_keypoints_3d']) > 0:
                    cnt+=1

            if cnt==0 or cnt>1:
                print(f"number of people in frame {frame_idx} is {cnt}\n")                