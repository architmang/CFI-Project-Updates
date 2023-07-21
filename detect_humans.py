
# NOT IN USE

# import os
# import cv2
# import numpy as np
# import pickle
# import torch
# from pathlib import Path

# def detect_humans(image_path):
#     # Load YOLOv5 model (yolov5x)
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

#     # Load the image
#     image = cv2.imread(image_path)

#     # Resize the image to the expected input size (640x640 for yolov5x)
#     image = cv2.resize(image, (640, 640))
    
#     # Convert the image to the expected format (batch_size=1, channels, height, width)
#     image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).float()

#     # Run the model on the image and get the predictions
#     results = model(image)

#     # Get the detections from results
#     detections = results[0]  # Shape: (25200, 85)

#     # Filter out human detections (class 0 corresponds to humans)
#     humans = []
#     for detection in detections:
#         class_index = int(detection[-1])
#         confidence = detection[4]
#         if class_index == 0 and confidence > 0.3:  # You can adjust the confidence threshold as needed
#             bbox = detection[:4].cpu().numpy().tolist()
#             humans.append(bbox)

#     return humans

# def main(input_dir):
#     all_points_labels = []

#     # Get a list of image file names in the input directory
#     image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     print(f"Found {len(image_files)} image files in {input_dir}.")

#     for image_file in image_files:
#         image_path = os.path.join(input_dir, image_file)
#         humans = detect_humans(image_path)

#         print(f"Detected {len(humans)} humans in {image_file}.")
#         exit()
#         if len(humans) >= 2:
#             # Sort the detected humans based on their x-coordinate (left to right)
#             sorted_humans = sorted(humans, key=lambda x: x[0])

#             # Assign label 0 to the leftmost human and label 1 to the rightmost human
#             leftmost_human = sorted_humans[0]
#             rightmost_human = sorted_humans[-1]

#             # Save the points and labels in a list
#             points_labels = [
#                 (leftmost_human, 0),
#                 (rightmost_human, 1)
#             ]

#             # Append the points and labels to the list
#             all_points_labels.append(points_labels)

#             # Display the image with bounding boxes around detected humans
#             image_with_boxes = cv2.imread(image_path)
#             for (x, y, w, h), label in points_labels:
#                 color = (0, 255, 0) if label == 0 else (0, 0, 255)  # Green for label 0, Red for label 1
#                 cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)

#             cv2.imshow('Detected Humans', image_with_boxes)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

#     return all_points_labels

# if __name__ == "__main__":

#     data_dir = './AzureKinectRecord_30_05'
#     group_list = ['1', '2', '3']  
#     cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

#     group_name = '2'
#     if group_name == '3':
#         start_index = 89
#     else:
#         start_index = 0

#     dict = {}    
#     for cam in cam_list[:1]:
#         print(cam)
#         input_directory = "%s/%s/%s/color" % (data_dir, group_name, cam)
#         all_points_labels = main(input_directory)
#         dict[cam] = all_points_labels

#     # dictionary(all cameras) of list(all images in dir) of list(2 humans) of tuples(labels)
#     # save the dictionary in a file
#     with open('%s/%s/color_image_labels.pickle' % (data_dir, group_name), 'wb') as handle:
#         pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

