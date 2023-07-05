import os
import cv2
import numpy as np

def load_yolo():
    # Load YOLOv3 pre-trained weights and configuration
    net = cv2.dnn.readNetFromDarknet("yolov3/yolov3.cfg", "yolov3/yolov3.weights")
    output_layers = net.getUnconnectedOutLayersNames()
    return net, output_layers


def detect_persons(directory_path, output_directory, net, output_layers):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Read the image file
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path)

        # Detect persons in the image using YOLOv3
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Person class ID is 0
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Apply non-maximum suppression to limit the number of detections to 3 at max
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            indices = indices.flatten()[:3]  # Limit the number of people detected to 3

            # Draw bounding boxes and print corner coordinates
            for i in indices:
                x, y, w, h = boxes[i]
                confidence = confidences[i]

                # Calculate the coordinates of the corners
                x1, y1 = x, y
                x2, y2 = x + w, y
                x3, y3 = x, y + h
                x4, y4 = x + w, y + h

                # Draw the bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Print the corner coordinates
                text = f"({x1},{y1}),({x4},{y4}),({h},{w})"
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the processed image in the output directory
        output_path = os.path.join(output_directory, image_file)
        cv2.imwrite(output_path, image)

        print(f"Processed: {image_file}")

    print("Processing complete.")


# Provide the directory path and output directory path
data_dir = './AzureKinectRecord_30_05'
group_list = ['1', '2', '3']
cam_list = ['azure_kinect1_2', 'azure_kinect1_3', 'azure_kinect2_4', 'azure_kinect2_5', 'azure_kinect3_3', 'azure_kinect3_2']

for group_name in group_list:
    for cam in cam_list:
        directory_path = os.path.join(data_dir, group_name, cam, 'color')
        output_directory = os.path.join(data_dir, group_name, cam, 'bounding_boxes')
        net, output_layers = load_yolo()
        detect_persons(directory_path, output_directory, net, output_layers)
